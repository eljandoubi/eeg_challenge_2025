from typing import Optional, Literal, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
from accelerate.utils import LoggerType


@dataclass
class VQVAETrainingLosses:
    amplitude_loss: torch.Tensor
    phase_loss: torch.Tensor
    orthogonality_loss: torch.Tensor
    encoder_perplexity: torch.Tensor

    def to_dict(self, prefix: str = "", item: bool = False):
        if not item:
            return {
                f"{prefix}amplitude_loss": self.amplitude_loss.detach(),
                f"{prefix}phase_loss": self.phase_loss.detach(),
                f"{prefix}orthogonality_loss": self.orthogonality_loss.detach(),
                f"{prefix}encoder_perplexity": self.encoder_perplexity.detach(),
            }

        return {
            f"{prefix}amplitude_loss": self.amplitude_loss.item(),
            f"{prefix}phase_loss": self.phase_loss.item(),
            f"{prefix}orthogonality_loss": self.orthogonality_loss.item(),
            f"{prefix}encoder_perplexity": self.encoder_perplexity.item(),
        }

    def map(self, func: Callable[[torch.Tensor], torch.Tensor], prefix: str = ""):
        res_dict = {}
        for k, v in self.to_dict(prefix, item=False).items():
            res_dict[k] = func(v)
        return VQVAETrainingLosses(**res_dict)


@dataclass
class Basic:
    def to_dict(self):
        return asdict(self)


@dataclass
class VQVALossesWeights(Basic):
    w_amplitude_loss: float = 1.0
    w_phase_loss: float = 1.0
    w_orthogonality_loss: float = 1.0
    w_encoder_perplexity: float = 1.0

    def __mul__(self, other: VQVAETrainingLosses) -> torch.Tensor:
        return (
            self.w_amplitude_loss * other.amplitude_loss
            + self.w_phase_loss * other.phase_loss
            + self.w_orthogonality_loss * other.orthogonality_loss
            + self.w_encoder_perplexity * other.encoder_perplexity
        )

    @torch.no_grad()
    def update(self, other: VQVAETrainingLosses):
        other_dict = other.to_dict()
        keys = tuple(other_dict.keys())
        self_dict = self.to_dict()
        weights = torch.tensor([self_dict[f"w_{k}"] for k in keys])
        weights.add_(torch.tensor([-other_dict[k] for k in keys]).softmax(dim=0)).div_(
            weights.sum()
        )
        for k in keys:
            setattr(self, f"w_{k}", weights[keys.index(k)].item())


@dataclass
class VQVAEConfig(Basic):
    n_times: int = 200
    n_chans: int = 128
    vocab_size: int = 1024
    embedding_dim: int = 1024
    temperature: float = 1.0
    qkv_bias: bool = True
    init_values: float = 0.1
    do_qk_norm: bool = True
    dorp: float = 0.1
    eps: float = 1e-7


@dataclass
class DataConfig(Basic):
    data_path: Path = "/home/a/startkit/data"
    tasks: str = "all"
    train_inter: str = "1"
    eval_inter: str = "5"
    num_workers: int = 4
    download: bool = False

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.tasks = self.tasks.split(",")
        if len(self.tasks) == 1:
            self.tasks = self.tasks[0]
        self.train_inter = list(map(int, self.train_inter.split(",")))
        self.eval_inter = list(map(int, self.eval_inter.split(",")))


@dataclass
class TrainingConfig(Basic):
    train_batch_size: int = 2
    eval_batch_size: int = 4
    num_epochs: int = 10
    seed: Optional[int] = 42
    lr: float = 1e-5
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "cosine"
    num_warmup_steps: Optional[int] = 10000
    num_training_steps: Optional[int] = None
    log_with: LoggerType = "tensorboard"
    project_dir: str = "./work"
    log_folder: str = "vqvae_eeg"
    max_grad_norm: float = 1.0
    logging_steps: int = 1000


@dataclass
class GumbelConfig(Basic):
    max_gumbel_temperature: float = 2.0
    min_gumbel_temperature: float = 0.5
    gumbel_temperature_decay: float = 0.999995
    step: Optional[int] = 0
