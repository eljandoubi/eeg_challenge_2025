from pathlib import Path

import torch.nn as nn
from accelerate import Accelerator


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        save_path: str = "best_checkpoint.pt",
    ):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum improvement to qualify as an improvement.
            save_path (str): File to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float, model: nn.Module, accelerator: Accelerator):
        """
        Args:
            score (float): The validation metric (e.g. loss or accuracy).
            model (torch.nn.Module): The model being trained.
            accelerator (Accelerator): HuggingFace Accelerate object.
        """
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            path_to_checkpoint = Path(accelerator.project_dir) / self.save_path
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.save(accelerator.get_state_dict(model), path_to_checkpoint)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
