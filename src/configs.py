from typing import Optional, Callable, Literal
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
from accelerate.utils import LoggerType
from transformers import SchedulerType


@dataclass
class VQVAETrainingLosses:
    amplitude_loss: torch.Tensor = torch.tensor(0.0)
    phase_loss: torch.Tensor = torch.tensor(0.0)
    orthogonality_loss: torch.Tensor = torch.tensor(0.0)
    encoder_perplexity: torch.Tensor = torch.tensor(0.0)

    def to_dict(self, prefix: str = "", post: Literal[None, "detach", "item"] = None):
        if post == "detach":
            return {
                f"{prefix}amplitude_loss": self.amplitude_loss.detach(),
                f"{prefix}phase_loss": self.phase_loss.detach(),
                f"{prefix}orthogonality_loss": self.orthogonality_loss.detach(),
                f"{prefix}encoder_perplexity": self.encoder_perplexity.detach(),
            }
        elif post == "item":
            return {
                f"{prefix}amplitude_loss": self.amplitude_loss.item(),
                f"{prefix}phase_loss": self.phase_loss.item(),
                f"{prefix}orthogonality_loss": self.orthogonality_loss.item(),
                f"{prefix}encoder_perplexity": self.encoder_perplexity.item(),
            }

        return {
            f"{prefix}amplitude_loss": self.amplitude_loss,
            f"{prefix}phase_loss": self.phase_loss,
            f"{prefix}orthogonality_loss": self.orthogonality_loss,
            f"{prefix}encoder_perplexity": self.encoder_perplexity,
        }

    def map(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        prefix: str = "",
        post: Literal[None, "detach", "item"] = None,
    ):
        res_dict = {}
        for k, v in self.to_dict(prefix, post).items():
            res_dict[k] = func(v)
        return VQVAETrainingLosses(**res_dict)

    def __add__(self, other: "VQVAETrainingLosses") -> "VQVAETrainingLosses":
        self_dict = self.to_dict(post="detach")
        other_dict = other.to_dict(post="detach")
        res_dict = {}
        for k in self_dict:
            res_dict[k] = self_dict[k] + other_dict[k]
        return VQVAETrainingLosses(**res_dict)

    def to(self, somthing):
        def _to(x: torch.Tensor):
            return x.to(somthing)

        return self.map(_to)


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

    @torch.inference_mode()
    def update(self, other: VQVAETrainingLosses):
        other_dict = other.to_dict(post="detach")
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
    train_inter: Optional[str] = None
    eval_inter: str = "5"
    num_workers: int = 4
    prefetch_factor: int = 8
    cache_size: int = 1_000
    download: bool = False
    process: bool = True
    train_data: str = "./work/train_data"
    eval_data: str = "./work/eval_data"

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.tasks = self.tasks.split(",")
        if self.process:
            self.download = False
        if len(self.tasks) == 1:
            self.tasks = self.tasks[0]
        if isinstance(self.train_inter, str):
            self.train_inter = list(map(int, self.train_inter.split(",")))
        else:
            inter = list(range(1, 12))
            inter.remove(5)
            self.train_inter = inter
        self.eval_inter = list(map(int, self.eval_inter.split(",")))
        self.train_data = Path(self.train_data)
        self.train_data.mkdir(parents=True, exist_ok=True)
        self.eval_data = Path(self.eval_data)
        self.eval_data.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig(Basic):
    train_batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 10_000
    seed: Optional[int] = 42
    lr: float = 1e-4
    weight_decay: float = 0.01
    lr_scheduler_type: SchedulerType = "cosine"
    num_warmup_steps: Optional[int] = 10000
    num_training_steps: Optional[int] = None
    log_with: LoggerType = "tensorboard"
    project_dir: str = "./work"
    log_folder: str = "vqvae_eeg"
    max_grad_norm: float = 10.0
    logging_steps: int = 1_000
    eval_steps: int = 50_000
    resume_from_checkpoint: bool = True


@dataclass
class GumbelConfig(Basic):
    max_gumbel_temperature: float = 2.0
    min_gumbel_temperature: float = 0.5
    gumbel_temperature_decay: float = 0.999995
    step: Optional[int] = 0


@dataclass
class EarlyStoppingConfig(Basic):
    patience: int = 5
    min_delta: float = 0.0
    save_path: str = "best_checkpoint.pt"
