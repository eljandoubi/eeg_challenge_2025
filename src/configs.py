from dataclasses import dataclass, asdict
import torch


@dataclass
class VQVAETrainingLosses:
    amplitude_loss: torch.Tensor
    phase_loss: torch.Tensor
    orthogonality_loss: torch.Tensor
    encoder_perplexity: torch.Tensor

    def to_dict(self):
        return {
            "amplitude_loss": self.amplitude_loss.detach(),
            "phase_loss": self.phase_loss.detach(),
            "orthogonality_loss": self.orthogonality_loss.detach(),
            "encoder_perplexity": self.encoder_perplexity.detach(),
        }


@dataclass
class VQVALossesWeights:
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

    def to_dict(self):
        return asdict(self)

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
