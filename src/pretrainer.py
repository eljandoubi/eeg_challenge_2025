import re
import logging
from pathlib import Path

import torch
from tqdm import tqdm
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from braindecode.datautil import load_concat_dataset
from transformers import get_scheduler, set_seed, HfArgumentParser

import configs as cfg
from model import VQVAE
from early_stop import EarlyStopping
from data_utils import data_download, fixed_windows

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


parser = HfArgumentParser(
    (
        cfg.TrainingConfig,
        cfg.VQVAEConfig,
        cfg.DataConfig,
        cfg.VQVALossesWeights,
        cfg.EarlyStoppingConfig,
    )
)

logging.info("Parse configs")

(training_cfg, vqvae_cfg, data_cfg, loss_weights, stp_cfg) = (
    parser.parse_args_into_dataclasses()
)

set_seed(training_cfg.seed)


if data_cfg.process:
    logging.info("Process Data")
    if data_cfg.download:
        logging.info("Download Data")
        data_download(training_cfg.data_path)

    logging.info("create fixed windows training data")
    train_set = fixed_windows(
        tasks=data_cfg.tasks,
        inter=data_cfg.train_inter,
        cache_dir=data_cfg.data_path,
        wind=vqvae_cfg.n_times,
    )
    logging.info("Save training data")
    train_set.save(training_cfg.train_data)
    logging.info("create fixed windows eval data")
    eval_set = fixed_windows(
        tasks=data_cfg.tasks,
        inter=data_cfg.eval_inter,
        cache_dir=data_cfg.data_path,
        wind=vqvae_cfg.n_times,
    )
    logging.info("Save eval data")
    eval_set.save(training_cfg.eval_data)
else:
    logging.info("Load fixed windows training data")
    train_set = load_concat_dataset(training_cfg.train_data, n_jobs=-1)
    train_dataloader = DataLoader(
        train_set,
        batch_size=training_cfg.train_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )
    logging.info("Load fixed windows eval data")
    eval_set = load_concat_dataset(training_cfg.eval_data, n_jobs=-1)
    eval_dataloader = DataLoader(
        eval_set,
        batch_size=training_cfg.eval_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )

    nb_eval_batch = len(eval_dataloader)

    training_cfg.num_training_steps = len(train_dataloader) * training_cfg.num_epochs

    logging.info("Create model")
    model = VQVAE(**vqvae_cfg.to_dict())

    logging.info("Create optimizer")
    optimizer = optim.AdamW(
        model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay
    )
    logging.info("Create scheduler")
    scheduler = get_scheduler(
        name=training_cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_cfg.num_warmup_steps,
        num_training_steps=training_cfg.num_training_steps,
    )
    logging.info("Create Accelerator")
    accelerator = Accelerator(
        log_with=training_cfg.log_with, project_dir=training_cfg.project_dir
    )

    accelerator.init_trackers(training_cfg.log_folder)
    logging.info("Prepare with accelerator")
    model, optimizer, scheduler, train_dataloader, eval_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, eval_dataloader
        )
    )

    def gather_batch(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(accelerator.gather_for_metrics(x)) / (
            training_cfg.train_batch_size * accelerator.state.num_processes
        )

    def gather_eval(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(accelerator.gather_for_metrics(x)) / (
            training_cfg.train_batch_size
            * nb_eval_batch
            * accelerator.state.num_processes
        )

    logging.info("Create early_stopping")
    early_stopping = EarlyStopping(**stp_cfg.to_dict())

    checkpoint_step = -1
    if training_cfg.resume_from_checkpoint:
        checkpoints = [
            p
            for p in Path(accelerator.project_dir).glob("checkpoint_*")
            if re.fullmatch(r"checkpoint_\d+", p.name)
        ]
        if len(checkpoints) > 0:
            path_to_checkpoint = max(
                checkpoints, key=lambda p: int(p.name.split("_")[-1])
            )
            checkpoint_step = int(path_to_checkpoint.name.split("_")[-1])
            accelerator.print(f"Resuming from Iteration: {checkpoint_step}")
            with accelerator.main_process_first():
                accelerator.load_state(path_to_checkpoint)

    step = 0
    model.train()
    progress_bar = tqdm(
        range(max(checkpoint_step, 0), training_cfg.num_training_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )
    for _ in range(training_cfg.num_epochs):
        for batch, *_ in train_dataloader:
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

                if step > 0 and step % training_cfg.logging_steps == 0:
                    if hasattr(model, "module"):
                        temperature = model.module.update_gumbel_temperature()
                    else:
                        temperature = model.update_gumbel_temperature()
                    gathered_losses = losses.map(gather_batch, post="detach")
                    state = gathered_losses.to_dict(prefix="train/", post="item")
                    state.update(loss_weights.to_dict())
                    state["temperature"] = temperature
                    state["lr"] = scheduler.get_last_lr()[0]
                    state["grad_norm"] = grad_norm.item()
                    state["train/loss"] = gather_batch(loss.detach()).item()
                    accelerator.log(state, step)

                if step > 0 and step % training_cfg.eval_steps == 0:
                    eval_losses = cfg.VQVAETrainingLosses().to(accelerator.device)
                    eval_loss = torch.tensor(0.0, device=accelerator.device)
                    model.eval()
                    for batch, *_ in tqdm(eval_dataloader, desc="Evaluation"):
                        with torch.inference_mode():
                            x = batch[:, :-1, ...] * 1e5
                            losses = model(x)
                            eval_losses += losses
                            loss = loss_weights * losses
                            eval_loss += loss

                    gathered_eval = eval_losses.map(gather_eval, post="detach")
                    loss_weights.update(gathered_eval)
                    eval_state = gathered_eval.to_dict(prefix="eval/", post="item")
                    eval_state["eval/loss"] = gather_eval(eval_loss.detach()).item()
                    accelerator.log(eval_state, step)
                    model.train()

                    path_to_checkpoint = (
                        Path(accelerator.project_dir) / f"checkpoint_{step}"
                    )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        accelerator.save_state(output_dir=path_to_checkpoint)

                    early_stopping.step(eval_state["eval/loss"], model, accelerator)
                progress_bar.update(1)
            step += 1
            if early_stopping.should_stop:
                accelerator.print("Early stopping triggered!")
                break
        if early_stopping.should_stop:
            break

    accelerator.end_training()
