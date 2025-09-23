import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEmbedding(nn.Module):
    def __init__(
        self,
        n_chans: int,
        vocab_size: int,
        embedding_dim: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.codebook_dim = embedding_dim
        self.vocab_size = vocab_size
        self.temperature = temperature

        self.codebook = nn.Parameter(torch.empty(1, n_chans, vocab_size, embedding_dim))
        nn.init.trunc_normal_(self.codebook)

    def compute_codeloss(self) -> torch.Tensor:
        norm = torch.matmul(self.codebook, self.codebook.transpose(-1, -2))
        norm = norm - torch.eye(self.vocab_size, device=self.codebook.device).reshape(
            1, 1, self.vocab_size, -1
        )
        norm = torch.linalg.norm(norm, dim=(-1, -2))
        return norm.mean()

    def forward(self, hidden_states: torch.Tensor, hard: bool = True):
        codevector_probs = F.gumbel_softmax(
            hidden_states.float(), tau=self.temperature, hard=self.training or hard
        )
        codevector = codevector_probs.reshape(
            -1, self.n_chans, self.vocab_size, 1
        ) * self.codebook.type_as(codevector_probs)

        return codevector


if __name__ == "__main__":
    embed = VectorQuantizerEmbedding(128, 2048, 16).cuda()
    ls = embed.compute_codeloss()
    print(ls)
    x = torch.randn(3, 128, 2048).cuda()
    y = embed(x)
    s = y.sum()
    print(s)
    a = s + ls
    a.backward()
    print(embed.codebook.grad.sum())
