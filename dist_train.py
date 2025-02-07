import os
import time
import logging
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Union

import deepspeed
from deepspeed.runtime.utils import see_memory_usage

# Example placeholder for your custom dataset loaders
# from your_dataset import MultiStepLoader, AutoregressiveLoader

# Your existing directory-creation function
def create_training_directory(base_dir: str):
    """
    Creates a unique subdirectory in `base_dir`.
    If `base_dir`/`model exists`, append _1, _2, etc.
    Returns (directory_path, name_counter).
    """
    os.makedirs(base_dir, exist_ok=True)

    name_counter = 0
    output_dir = os.path.join(base_dir, f"model")
    while os.path.exists(output_dir):
        name_counter += 1
        output_dir = os.path.join(base_dir, f"model_{name_counter}")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir, name_counter

def train_model_deepseek(
    model,
    train_loader: Union["MultiStepLoader", "AutoregressiveLoader"],
    epochs=1,
    train_mode="incremental",
    lr=1e-4,
    save_every=15,
    verbose=False,
    base_dir="checkpoints",
    base_model_name="model",
    eta_min=1e-6,
    warmup_steps=100,
    run_name='tsfm',
    gradient_accumulation_steps=2
):
    """
    DeepSpeed-based multi-GPU training function.

    Args:
        model (nn.Module): The model to train.
        train_loader (Union[MultiStepLoader, AutoregressiveLoader]): Training data loader.
        epochs (int): Number of epochs to train.
        train_mode (str, optional): 'incremental' or 'multi-step'.
        lr (float, optional): Initial learning rate.
        save_every (int, optional): Save a checkpoint every N mini-batches.
        verbose (bool, optional): Whether to log detailed timing info.
        base_dir (str, optional): Directory where checkpoints will be saved.
        base_model_name (str, optional): Base name for saved checkpoints.
        eta_min (float, optional): Minimum learning rate for CosineAnnealing.
        warmup_steps (int, optional): Number of warmup steps.
        run_name (str, optional): Base name for wandb runs.
        gradient_accumulation_steps (int, optional): Steps to accumulate grads before syncing.
    """

    # ----------------------------------------------------------------------
    # 1) Initialize Distributed + Parse local/global rank
    # ----------------------------------------------------------------------
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # ----------------------------------------------------------------------
    # 2) Configure Logging
    #    - Master rank logs to console + wandb
    #    - All ranks log to rank-specific file
    # ----------------------------------------------------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Each rank writes to its own file
    log_file = f"train_rank_{local_rank}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Only rank 0 logs to console (optional)
    if local_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    # ----------------------------------------------------------------------
    # 3) Prepare output directory & wandb (only on master rank)
    # ----------------------------------------------------------------------
    output_dir, name_counter = create_training_directory(base_dir)
    run_name = f"{run_name}_{name_counter}"

    if local_rank == 0:
        logger.info(f"[INFO] Checkpoints will be saved to: {output_dir}")
        wandb.init(project="Time-Series-FMs", name=run_name)
    else:
        # If not rank 0, do not init wandb (avoid conflicts)
        wandb.run = None  # Safe guard

    # ----------------------------------------------------------------------
    # 4) Define DeepSpeed config and Initialize
    # ----------------------------------------------------------------------
    #   - We use manual CosineAnnealingLR, so we won't specify a DS scheduler
    #   - We'll do partial gradient sync via `gradient_accumulation_steps`
    #   - We'll do FP16 by default; can disable if you prefer full precision
    # ----------------------------------------------------------------------
    ds_config = {
        "train_micro_batch_size_per_gpu": train_loader.batch_size,  # Must match your loader’s batch_size
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16": {
            "enabled": True
        },
        # This is the total *effective* batch = micro_batch * accumulation * world_size
        "train_batch_size": train_loader.batch_size * gradient_accumulation_steps * world_size,
        "zero_optimization": {
            "stage": 0  # If you want ZeRO stage 1/2/3, change here
        },
        "gradient_clipping": 1.0
    }

    # Move model to device before DeepSpeed init
    model.to(device)

    # NOTE: We'll define a standard torch optimizer below and pass it to DS.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # If you want AdamW, etc., just replace it.

    # DeepSpeed Initialize
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Manually define a PyTorch Cosine LR Scheduler (per-rank)
    # We'll handle warmup logic in the loop
    total_steps = epochs * len(train_loader)
    # Because we’re using gradient accumulation, the # of optimizer steps is total_steps / gradient_accumulation_steps
    effective_steps = total_steps // gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=effective_steps, eta_min=eta_min)

    # ----------------------------------------------------------------------
    # 5) Training Loop
    # ----------------------------------------------------------------------
    logger.info("Starting training...")
    global_step = 0
    model_engine.train()

    # For time measurements
    t_cycle_end_prev = None
    time_load_next_batch_prev = 0.0

    for epoch in range(epochs):
        # If you need data sampler epoch setting:
        # if hasattr(train_loader.sampler, 'set_epoch'):
        #     train_loader.sampler.set_epoch(epoch)

        # local steps for gradient accumulation
        accumulated_steps = 0

        for x, y, attn_mask, padding_mask in train_loader:
            # --------------- (A) Measure time to load next batch ---------------
            t_cycle_start = time.time()
            if t_cycle_end_prev is None:
                time_load_next_batch_current = 0.0
            else:
                time_load_next_batch_current = t_cycle_start - t_cycle_end_prev

            # ------------------------ Data to Device ---------------------------
            t0 = time.time()
            x = x.to(device)
            y = y.to(device)
            attn_mask = attn_mask.to(device)
            padding_mask = padding_mask.to(device)
            t_data_to_device = time.time() - t0

            # ------------------------ Forward Pass -----------------------------
            t1 = time.time()
            output = model_engine(x, attn_mask=attn_mask, padding_mask=padding_mask)
            t_forward = time.time() - t1

            # ---------------------- Compute Loss -------------------------------
            t2 = time.time()
            criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.preprocessor.PAD_TOKEN)
            if train_mode == "incremental":
                # Use only last step for loss
                last_logits = output[:, -1, :]       # [batch_size, vocab_size]
                last_target = y[:, -1]              # [batch_size]
                loss = criterion(last_logits, last_target)
            else:  # "multi-step"
                # Compute CE across entire sequence
                logits = output.transpose(1, 2)     # [batch, classes, seq_len]
                loss = criterion(logits, y)
            t_loss = time.time() - t2

            # ---------------------- Backward + Grad Sync -----------------------
            t3 = time.time()
            # Measure time spent in backward for possible gradient sync
            start_grad_sync = time.time()
            model_engine.backward(loss)  
            # By default, DS won't all-reduce grads until we call step() or
            # if gradient_accumulation_steps=1 in the config (or we can measure it here).
            torch.cuda.synchronize()
            grad_sync_time = time.time() - start_grad_sync

            t_backward = time.time() - t3

            # --------------- Conditionally Step (accumulation) -----------------
            step_time_start = time.time()
            accumulated_steps += 1
            # Only update model & scheduler after enough local steps
            if accumulated_steps % gradient_accumulation_steps == 0:
                model_engine.step()
                # Manually step the PyTorch LR scheduler
                # Warmup logic: only step if past warmup
                if global_step >= warmup_steps:
                    scheduler.step()
                # Zero grads are automatically handled by DS engine if not using ZeRO stage 3
                # but we can do model_engine.zero_grad() if needed

            t_optimizer_step = time.time() - step_time_start

            # --------------- Clear references, measure memory cleanup ----------
            t5 = time.time()
            loss_val = loss.item()
            del loss, output, x, y, attn_mask, padding_mask
            torch.cuda.empty_cache()
            t_clear_mem = time.time() - t5

            # --------------- Update global step & possible checkpoint ----------
            global_step += 1
            t_save_model = None
            if (global_step % save_every) == 0 and local_rank == 0:
                t_save = time.time()
                # Use DeepSpeed's save_checkpoint to properly save partitioned weights if using ZeRO
                # If using stage=0, you could also just do standard torch.save
                ckpt_name = f"{base_model_name}_{global_step}"
                model_engine.save_checkpoint(output_dir, ckpt_name)
                t_save_model = time.time() - t_save
                logger.info(f"[INFO] Saved checkpoint at step {global_step} -> {ckpt_name}")

            # --------------- Gather logs ---------------------------------------
            log_dict = {
                'global_step': global_step,
                'train_loss': loss_val,
                'lr': optimizer.param_groups[0]['lr'],

                # Current iteration's critical times
                'time/data_to_device': t_data_to_device,
                'time/forward_pass': t_forward,
                'time/loss_computation': t_loss,
                'time/backward_pass': t_backward,
                'time/grad_sync': grad_sync_time,        # measure of DS backward sync
                'time/optimizer_step': t_optimizer_step,
                'time/memory_clear': t_clear_mem,
                # Lagged logs
                'time/load_next_batch': time_load_next_batch_prev
            }
            if t_save_model is not None:
                log_dict['time/model_save_to_disk'] = t_save_model

            # Master process logs to wandb
            if local_rank == 0 and wandb.run is not None:
                wandb.log(log_dict)

            # Every process logs to its local file
            logger.info(
                f"[Epoch {epoch+1}/{epochs}] "
                f"GlobalStep {global_step} | "
                f"Loss {loss_val:.4f} | LR {optimizer.param_groups[0]['lr']:.6f} | "
                f"data->dev {t_data_to_device:.4f}s, fw {t_forward:.4f}s, loss {t_loss:.4f}s, "
                f"back {t_backward:.4f}s, sync {grad_sync_time:.4f}s, opt {t_optimizer_step:.4f}s"
            )

            if verbose and local_rank == 0:
                # Only rank 0 might print more detailed breakdown
                logger.info(
                    f"[VERBOSE] data->dev: {t_data_to_device:.4f}s | "
                    f"forward: {t_forward:.4f}s | loss: {t_loss:.4f}s | "
                    f"backward: {t_backward:.4f}s | sync: {grad_sync_time:.4f}s | "
                    f"optimizer: {t_optimizer_step:.4f}s"
                )

            time_load_next_batch_prev = time_load_next_batch_current
            t_cycle_end_prev = time.time()

        # End of epoch
        # If you want an epoch-level checkpoint, do so here
        # if local_rank == 0:
        #     model_engine.save_checkpoint(output_dir, f"{base_model_name}_epoch_{epoch+1}")

    # ----------------------------------------------------------------------
    # 6) Final Save + Cleanup
    # ----------------------------------------------------------------------
    if local_rank == 0:
        final_ckpt_name = f"{base_model_name}_{global_step}"
        model_engine.save_checkpoint(output_dir, final_ckpt_name)
        logger.info(f"[INFO] Saved final checkpoint -> {final_ckpt_name}")

        if wandb.run is not None:
            wandb.finish()
    dist.barrier()  # Ensure all ranks finish before returning
