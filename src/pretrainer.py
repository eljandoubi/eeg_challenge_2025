import re
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from transformers import get_scheduler, set_seed, HfArgumentParser
from data_utils import data_download, fixed_windows
from model import VQVAE
import configs as cfg

parser = HfArgumentParser(
    (cfg.TrainingConfig, cfg.VQVAEConfig, cfg.DataConfig, cfg.VQVALossesWeights)
)

(training_cfg, vqvae_cfg, data_cfg, loss_weights) = parser.parse_args_into_dataclasses()

set_seed(training_cfg.seed)

model = VQVAE(**vqvae_cfg.to_dict())

optimizer = optim.Adam(model.parameters(), lr=training_cfg.lr)

if data_cfg.download:
    data_download(training_cfg.data_path)

# train_set = fixed_windows(
#     tasks=data_cfg.tasks,
#     inter=data_cfg.train_inter,
#     cache_dir=data_cfg.data_path,
#     wind=vqvae_cfg.n_times,
# )

train_set = torch.randn(10, 129, 200)

train_dataloader = DataLoader(
    train_set,
    batch_size=training_cfg.train_batch_size,
    shuffle=False,
    num_workers=data_cfg.num_workers,
    pin_memory=True,
)

# eval_set = fixed_windows(
#     tasks=data_cfg.tasks,
#     inter=data_cfg.eval_inter,
#     cache_dir=data_cfg.data_path,
#     wind=vqvae_cfg.n_times,
# )

# eval_dataloader = DataLoader(
#     eval_set,
#     batch_size=training_cfg.train_batch_size,
#     shuffle=False,
#     num_workers=data_cfg.num_workers,
#     pin_memory=True,
# )

training_cfg.num_training_steps = len(train_dataloader) * training_cfg.num_epochs

scheduler = get_scheduler(
    name=training_cfg.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=training_cfg.num_warmup_steps,
    num_training_steps=training_cfg.num_training_steps,
)

accelerator = Accelerator(
    log_with=training_cfg.log_with, project_dir=training_cfg.project_dir
)

accelerator.init_trackers(training_cfg.log_folder)

model, optimizer, scheduler, train_dataloader, *eval_dataloader = accelerator.prepare(
    model,
    optimizer,
    scheduler,
    train_dataloader,  # eval_dataloader
)


def gather(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(accelerator.gather_for_metrics(x)) / (
        training_cfg.train_batch_size * accelerator.state.num_processes
    )


checkpoint_step = -1
if training_cfg.resume_from_checkpoint:
    checkpoints = [
        p
        for p in Path(accelerator.project_dir).glob("checkpoint_*")
        if re.fullmatch(r"checkpoint_\d+", p.name)
    ]
    path_to_checkpoint = max(checkpoints, key=lambda p: int(p.name.split("_")[-1]))
    checkpoint_step = int(path_to_checkpoint.name.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {checkpoint_step}")
    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)

step = 0
for batch, *_ in tqdm(train_dataloader, desc="Training"):
    if step > checkpoint_step:
        x = batch[:, :-1, ...] * 1e5
        optimizer.zero_grad()
        losses = model(x)
        loss = loss_weights * losses
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), training_cfg.max_grad_norm
            )
        optimizer.step()
        scheduler.step()
        if hasattr(model, "module"):
            temperature = model.module.update_gumbel_temperature()
        else:
            temperature = model.update_gumbel_temperature()

        if step % training_cfg.logging_steps == 0:
            gathered_losses = losses.map(gather)
            loss_weights.update(gathered_losses)
            state = gathered_losses.to_dict(prefix="train/", item=True)
            state["temperature"] = temperature
            state["lr"] = scheduler.get_last_lr()[0]
            state["grad_norm"] = grad_norm.item()
            accelerator.log(state, step)
            path_to_checkpoint = Path(accelerator.project_dir) / f"checkpoint_{step}"
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                accelerator.save_state(output_dir=path_to_checkpoint)
    step += 1
    break

print(loss_weights)
accelerator.end_training()
