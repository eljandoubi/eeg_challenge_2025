from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import Labram
from configs import VQVAETrainingLosses, GumbelConfig


class VectorQuantizerEmbedding(nn.Module):
    def __init__(
        self,
        n_chans: int,
        vocab_size: int,
        embedding_dim: int,
        temperature: float = 1.0,
    ):
        assert embedding_dim >= vocab_size, (
            f"We need embedding_dim >= vocab_size but got : {embedding_dim} < {vocab_size}."
        )
        super().__init__()
        self.n_chans = n_chans
        self.codebook_dim = embedding_dim
        self.vocab_size = vocab_size
        self.temperature = temperature

        self.codebook = nn.Parameter(torch.empty(1, n_chans, vocab_size, embedding_dim))
        nn.init.trunc_normal_(self.codebook, std=0.02)

    def compute_codeloss(self) -> torch.Tensor:
        norm = torch.matmul(self.codebook, self.codebook.transpose(-1, -2))
        norm = norm - torch.eye(self.vocab_size, device=self.codebook.device).reshape(
            1, 1, self.vocab_size, self.vocab_size
        )
        norm = torch.linalg.norm(norm, dim=(-1, -2))
        return norm.mean()

    def forward(self, hidden_states: torch.Tensor, hard: bool = True):
        codevector_probs = F.gumbel_softmax(
            hidden_states.float(), tau=self.temperature, hard=hard
        )
        codevector = codevector_probs.reshape(
            codevector_probs.size(0), self.n_chans, self.vocab_size, 1
        ) * self.codebook.type_as(codevector_probs)

        return codevector.sum(-2)


class VQVAE(nn.Module):
    def __init__(
        self,
        n_times: int,
        n_chans: int,
        vocab_size: int,
        embedding_dim: int,
        temperature: float = 1.0,
        qkv_bias: bool = True,
        init_values: float = 0.1,
        do_qk_norm: bool = True,
        dorp: float = 0.1,
        gumbel_cfg: GumbelConfig = GumbelConfig(),
    ):
        assert embedding_dim >= n_times, (
            f"We need embedding_dim >= n_times but got : {embedding_dim} < {n_times}."
        )
        super().__init__()
        self.gumbel_cfg = gumbel_cfg
        self.embedding_dim = embedding_dim
        if do_qk_norm:
            qk_norm = nn.LayerNorm
        else:
            qk_norm = None

        self.encoder = Labram(
            n_times=n_times,
            n_chans=n_chans,
            qkv_bias=qkv_bias,
            n_outputs=vocab_size,
            init_values=init_values,
            qk_norm=qk_norm,
            drop_prob=dorp,
            drop_path_prob=dorp,
            attn_drop_prob=dorp,
        )

        self.vq = VectorQuantizerEmbedding(
            n_chans, vocab_size, embedding_dim, temperature
        )

        self.decoder = Labram(
            n_times=embedding_dim,
            n_chans=n_chans,
            patch_size=embedding_dim,
            qkv_bias=qkv_bias,
            n_outputs=n_times,
            init_values=init_values,
            qk_norm=qk_norm,
            drop_prob=dorp,
            drop_path_prob=dorp,
            attn_drop_prob=dorp,
        )

        self.phase_back = nn.Sequential(
            nn.Linear(n_times, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, n_times),
        )
        self.phase_head = nn.Sequential(
            nn.LayerNorm(n_times),
            nn.Linear(n_times, n_times),
        )

        self.magnitude_back = nn.Sequential(
            nn.Linear(n_times, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, n_times),
        )

        self.magnitude_head = nn.Sequential(
            nn.LayerNorm(n_times),
            nn.Linear(n_times, n_times),
        )

        self.apply(Labram._init_weights)

    @torch.no_grad()
    def _compute_target(self, x: torch.Tensor):
        X = torch.fft.fft(x, dim=-1)
        return X.abs(), X.angle() / torch.pi

    def _encoder_perplexity(self, h: torch.Tensor):
        return (
            torch.tensor(self.embedding_dim, device=h.device).log()
            + (h.softmax(dim=-1) * h.log_softmax(dim=-1)).sum(dim=-1)
        ).mean()

    def forward(self, x: torch.Tensor, hard: bool = True):
        h = self.encoder(x, return_patch_tokens=True)
        v = self.vq(h, hard)
        h_r = self.decoder(v, return_patch_tokens=True)
        phi = self.phase_head(self.phase_back(h_r) + h_r)
        ampl = self.magnitude_head(self.magnitude_back(h_r) + h_r)

        tr_ampl, tr_phi = self._compute_target(x)
        phi_loss = F.huber_loss(phi, tr_phi, reduction="mean")
        ampl_loss = F.huber_loss(ampl, tr_ampl, reduction="mean")
        orth_loss = self.vq.compute_codeloss()
        encode_loss = self._encoder_perplexity(h)

        return VQVAETrainingLosses(
            amplitude_loss=ampl_loss,
            phase_loss=phi_loss,
            orthogonality_loss=orth_loss,
            encoder_perplexity=encode_loss,
        )

    def update_gumbel_temperature(self, steps: Optional[int] = None):
        args = self.gumbel_cfg
        if steps is None:
            steps = args.step
            args.step += 1
        self.vq.temperature = max(
            args.max_gumbel_temperature * args.gumbel_temperature_decay**steps,
            args.min_gumbel_temperature,
        )
        return self.vq.temperature


if __name__ == "__main__":
    from configs import VQVALossesWeights, VQVAEConfig

    cfg = VQVAEConfig()
    print(cfg)
    vqvae = VQVAE(**cfg.to_dict()).cuda()
    x = torch.randn(2, 128, 200).cuda()
    y = vqvae(x)
    print(y)
    w = VQVALossesWeights()
    print(w)
    z = w * y
    print(z)
    z.backward()
    n = vqvae.vq.codebook.grad.norm()
    print(n)
    w.update(y)
    print(w)
