import logging
import re
from datetime import timedelta
from itertools import cycle
from pathlib import Path

import torch
import torch.optim as optim
from accelerate import Accelerator, PartialState
from accelerate.utils import release_memory
from accelerate.utils.memory import clear_device_cache
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, get_scheduler, set_seed

import configs as cfg
from data_utils import data_download, fixed_windows
from early_stop import EarlyStopping
from model import VQVAE
from torch_dataset import SubjectsDataset

main_state = PartialState(timeout=timedelta(days=1))
idx = main_state.process_index
FORMAT = f"process {idx} %(levelname)s %(asctime)s,%(msecs)d %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = HfArgumentParser(
        (
            cfg.TrainingConfig,
            cfg.VQVAEConfig,
            cfg.DataConfig,
            cfg.VQVALossesWeights,
            cfg.EarlyStoppingConfig,
        )
    )

    logger.info("Parse configs")

    cfgs = parser.parse_args_into_dataclasses()

    for conf in cfgs:
        logger.info(conf)

    (training_cfg, vqvae_cfg, data_cfg, loss_weights, stp_cfg) = cfgs

    set_seed(training_cfg.seed)

    if data_cfg.process:
        logger.info("Process Data")
        if data_cfg.download:
            logger.info("Download Data")
            data_download(data_cfg.data_path)

        logger.info("create fixed windows training data")
        train_set = fixed_windows(
            tasks=data_cfg.tasks,
            inter=data_cfg.train_inter,
            cache_dir=data_cfg.data_path,
            wind=vqvae_cfg.n_times,
        )
        logger.info("Save training data")
        train_set.save(data_cfg.train_data)
        logger.info("create fixed windows eval data")
        eval_set = fixed_windows(
            tasks=data_cfg.tasks,
            inter=data_cfg.eval_inter,
            cache_dir=data_cfg.data_path,
            wind=vqvae_cfg.n_times,
        )
        logger.info("Save eval data")
        eval_set.save(data_cfg.eval_data)
    else:
        logger.info("Create Accelerator")
        accelerator = Accelerator(
            log_with=training_cfg.log_with, project_dir=training_cfg.project_dir
        )

        accelerator.init_trackers(training_cfg.log_folder)

        accelerator.print(accelerator.distributed_type)

        logger.info("Load fixed windows training data")
        train_set = SubjectsDataset(
            data_cfg.train_data,
            nb_subjects=data_cfg.nb_train_sub,
            cache_size=data_cfg.cache_size,
        )

        logger.info("Load fixed windows eval data")
        eval_set = SubjectsDataset(
            data_cfg.eval_data,
            nb_subjects=data_cfg.nb_eval_sub,
            cache_size=data_cfg.cache_size,
        )

        if accelerator.is_main_process:
            logger.info("Start Train initialization")
            sizes, subjects = train_set.subject_and_intervals()
            train_sizes = torch.from_numpy(sizes).to(
                dtype=torch.long, device=accelerator.device
            )
            train_subjects = torch.from_numpy(subjects).to(
                dtype=torch.long, device=accelerator.device
            )
            logger.info("Finish Train initialization")
            logger.info("Start Eval initialization")
            sizes, subjects = eval_set.subject_and_intervals()
            eval_sizes = torch.from_numpy(sizes).to(
                dtype=torch.long, device=accelerator.device
            )
            eval_subjects = torch.from_numpy(subjects).to(
                dtype=torch.long, device=accelerator.device
            )
            logger.info("Finish Eval initialization")
        else:
            logger.info("Wait main initialization")
            sizes, subjects = None, None
            train_sizes = torch.zeros(
                data_cfg.nb_train_sub, dtype=torch.long, device=accelerator.device
            )
            train_subjects = torch.zeros(
                data_cfg.nb_train_sub, dtype=torch.long, device=accelerator.device
            )
            eval_sizes = torch.zeros(
                data_cfg.nb_eval_sub, dtype=torch.long, device=accelerator.device
            )
            eval_subjects = torch.zeros(
                data_cfg.nb_eval_sub, dtype=torch.long, device=accelerator.device
            )

        accelerator.wait_for_everyone()
        train_set.syn_data(
            sizes=accelerator.reduce(train_sizes).cpu().numpy(),
            subjects=accelerator.reduce(train_subjects).cpu().numpy(),
        )
        eval_set.syn_data(
            sizes=accelerator.reduce(eval_sizes).cpu().numpy(),
            subjects=accelerator.reduce(eval_subjects).cpu().numpy(),
        )
        accelerator.wait_for_everyone()
        sizes, subjects, train_sizes, train_subjects, eval_sizes, eval_subjects = (
            release_memory(
                sizes, subjects, train_sizes, train_subjects, eval_sizes, eval_subjects
            )
        )
        logger.info(
            "Train and Eval size shapes: %s, %s",
            train_set.sizes.shape,
            eval_set.sizes.shape,
        )

        train_dataloader = DataLoader(
            train_set,
            batch_size=training_cfg.train_batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            prefetch_factor=data_cfg.prefetch_factor,
            persistent_workers=data_cfg.persistent_workers,
        )
        eval_dataloader = DataLoader(
            eval_set,
            batch_size=training_cfg.eval_batch_size,
            shuffle=False,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            prefetch_factor=data_cfg.prefetch_factor,
            persistent_workers=data_cfg.persistent_workers,
        )

        nb_eval_batch = len(eval_dataloader)

        training_cfg.num_training_steps = (
            len(train_dataloader) * training_cfg.num_epochs
        )

        logger.info("Create model")
        model = VQVAE(**vqvae_cfg.to_dict())

        logger.info("Create optimizer")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_cfg.lr,
            weight_decay=training_cfg.weight_decay,
        )
        logger.info("Create scheduler")
        scheduler = get_scheduler(
            name=training_cfg.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_cfg.num_warmup_steps,
            num_training_steps=training_cfg.num_training_steps,
        )

        logger.info("Prepare with accelerator")
        model, optimizer, scheduler, train_dataloader, eval_dataloader = (
            accelerator.prepare(
                model, optimizer, scheduler, train_dataloader, eval_dataloader
            )
        )

        def gather_batch(x: torch.Tensor) -> torch.Tensor:
            return (
                torch.mean(accelerator.gather_for_metrics(x))
                / training_cfg.logging_steps
            )

        def gather_eval(x: torch.Tensor) -> torch.Tensor:
            return torch.mean(accelerator.gather_for_metrics(x)) / nb_eval_batch

        logger.info("Create early_stopping")
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

        model.train()
        progress_bar = tqdm(
            range(max(checkpoint_step, 0), training_cfg.num_training_steps),
            disable=not accelerator.is_local_main_process,
            desc="Training",
        )
        train_iter = cycle(train_dataloader)
        with torch.inference_mode():
            train_losses = cfg.VQVAETrainingLosses().to(accelerator.device)
            train_loss = torch.tensor(0.0, device=accelerator.device)
            train_grad_norm = torch.tensor(0.0, device=accelerator.device)

        for step in progress_bar:
            batch, *_ = next(train_iter)
            x = batch[:, :-1, ...] * 1e5
            optimizer.zero_grad()
            losses = model(x)
            loss = loss_weights * losses
            accelerator.backward(loss)
            with torch.inference_mode():
                train_losses += losses.detach()
                train_loss += loss.detach()
                if accelerator.sync_gradients:
                    train_grad_norm += accelerator.clip_grad_norm_(
                        model.parameters(), training_cfg.max_grad_norm
                    )
            optimizer.step()
            scheduler.step()
            clear_device_cache(garbage_collection=False)

            if step > 0 and step % training_cfg.logging_steps == 0:
                with torch.inference_mode():
                    if hasattr(model, "module"):
                        temperature = model.module.update_gumbel_temperature()
                    else:
                        temperature = model.update_gumbel_temperature()

                    gathered_losses = train_losses.map(gather_batch, post="detach")
                    state = gathered_losses.to_dict(prefix="train/", post="item")
                    state.update(loss_weights.to_dict())
                    state["temperature"] = temperature
                    state["lr"] = scheduler.get_last_lr()[0]
                    state["grad_norm"] = gather_batch(train_grad_norm).item()
                    state["train/loss"] = gather_batch(train_loss).item()
                    accelerator.log(state, step)
                    accelerator.print(f"Step {step}, train: {state}")
                    accelerator.wait_for_everyone()
                    (
                        train_losses,
                        train_loss,
                        train_grad_norm,
                        state,
                        gathered_losses,
                    ) = release_memory(
                        train_losses,
                        train_loss,
                        train_grad_norm,
                        state,
                        gathered_losses,
                    )
                    train_losses = cfg.VQVAETrainingLosses().to(accelerator.device)
                    train_loss = torch.tensor(0.0, device=accelerator.device)
                    train_grad_norm = torch.tensor(0.0, device=accelerator.device)

            if step > 0 and step % training_cfg.eval_steps == 0:
                model.eval()
                batch, x, losses, loss = release_memory(batch, x, losses, loss)
                with torch.inference_mode():
                    eval_losses = cfg.VQVAETrainingLosses().to(accelerator.device)
                    eval_loss = torch.tensor(0.0, device=accelerator.device)
                    for batch, *_ in tqdm(eval_dataloader, desc="Evaluation"):
                        x = batch[:, :-1, ...] * 1e5
                        losses = model(x)
                        eval_losses += losses
                        loss = loss_weights * losses
                        eval_loss += loss

                    batch, x, losses, loss = release_memory(batch, x, losses, loss)
                    gathered_eval = eval_losses.map(gather_eval, post="detach")
                    loss_weights.update(gathered_eval)
                    eval_state = gathered_eval.to_dict(prefix="eval/", post="item")
                    eval_state["eval/loss"] = gather_eval(eval_loss.detach()).item()
                    accelerator.log(eval_state, step)
                    accelerator.print(f"Step {step}, eval: {eval_state}")
                    accelerator.wait_for_everyone()
                    eval_losses, eval_loss, gathered_eval, eval_state = release_memory(
                        eval_losses, eval_loss, gathered_eval, eval_state
                    )
                model.train()

                path_to_checkpoint = (
                    Path(accelerator.project_dir) / f"checkpoint_{step}"
                )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.save_state(output_dir=path_to_checkpoint)

                early_stopping.step(eval_state["eval/loss"], model, accelerator)

            if early_stopping.should_stop:
                accelerator.print("Early stopping triggered!")
                break

        accelerator.end_training()


if __name__ == "__main__":
    main()
