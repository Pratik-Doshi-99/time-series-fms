# train.py
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Union
from data.dataset import MultiStepLoader, AutoregressiveLoader
import wandb
import csv
from tqdm import tqdm
from utils import create_training_directory, log_to_csv
from models.model import DecoderOnlyTransformer
from training.training_extension import MetricsAggregation, calculate_accuracy
import signal
import sys


def validation_step(model, val_loader, criterion, device, train_mode="incremental"):
    """
    Execute validation loop and return aggregated validation metrics.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
        train_mode: 'incremental' or 'multi-step'

    Returns:
        Dictionary of aggregated validation metrics
    """
    model.eval()
    metrics_aggregator = MetricsAggregation()

    with torch.no_grad():
        for x, y, attn_mask, padding_mask in tqdm(val_loader, desc="Validation", leave=False):
            # Move data to device
            x = x.to(device)
            y = y.to(device)
            attn_mask = attn_mask.to(device)
            padding_mask = padding_mask.to(device)

            # Forward pass
            output = model(x, attn_mask=attn_mask, padding_mask=padding_mask)

            # Compute loss
            if train_mode == "incremental":
                last_logits = output[:, -1, :]
                last_target = y[:, -1]
                loss = criterion(last_logits, last_target)
            else:  # "multi-step"
                logits = output.transpose(1, 2)
                loss = criterion(logits, y)

            # Accumulate metrics
            val_metrics = {
                'val_loss': loss.cpu().item(),
                'val_acc': calculate_accuracy(logits, y, padding_mask)
            }
            metrics_aggregator.accumulate(val_metrics)

            # Clean up memory
            del loss, output, x, y, attn_mask, padding_mask
            torch.cuda.empty_cache()

    # Aggregate and return metrics
    aggregated_val_metrics = metrics_aggregator.aggregate('mean')
    model.train()

    return aggregated_val_metrics


def train_model(
    model: DecoderOnlyTransformer,
    train_loader: Union["MultiStepLoader", "AutoregressiveLoader"],
    val_loader: Union["MultiStepLoader", "AutoregressiveLoader"],
    device,
    epochs=1,
    train_mode="incremental",
    # Additional parameters for improved training loop
    lr=1e-4,
    save_every=15,
    val_every=15,
    verbose=False,
    base_dir="checkpoints",
    base_model_name="model",
    eta_min=1e-6,
    warmup_steps = 100,
    run_name = 'tsfm',
    verbose_acts = False,
    weight_decay = 1e-4,
    optimizer_type = 'adam',
    start_global_step = 0
):
    """
    Train the given model, with options for incremental or multi-step training modes.

    Args:
        model (nn.Module): The model to train.
        train_loader (Union[MultiStepLoader, AutoregressiveLoader]): Training data loader.
        epochs (int): Number of epochs to train.
        device: The device to train on.
        train_mode (str, optional): 'incremental' or 'multi-step'.
        lr (float, optional): Initial learning rate.
        save_every (int, optional): Save a checkpoint every N mini-batches.
        verbose (bool, optional): Whether to print detailed timing info each iteration.
        base_dir (str, optional): Directory where checkpoints will be saved.
        base_model_name (str, optional): Base name for saved checkpoint files.
        eta_min (float, optional): Minimum learning rate for CosineAnnealingLR.
    """
    # Ensure wandb is initialized (if you're using wandb)
    # wandb.init(project="your_project_name")  # Uncomment or modify as needed

    # Create a directory to store checkpoints
    #os.makedirs(base_dir, exist_ok=True)
    #output_dir = base_dir
    output_dir, name_counter = create_training_directory(base_dir)
    print(f"[INFO] Checkpoints will be saved to: {output_dir}")
    run_name = f"{run_name}_{name_counter}"
    wandb.init(project="Time-Series-FMs", name=run_name)

    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n[INFO] Received signal {signum}. Finishing WandB run and exiting gracefully...")
        try:
            wandb.finish(exit_code=1, quiet=True)
        except Exception as e:
            # Suppress WandB errors during shutdown (e.g., BrokenPipeError)
            print(f"[WARNING] WandB finish failed (expected during forced shutdown): {type(e).__name__}")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Move model to device
    model.to(device)

    # Optimizer & Cosine LR Scheduler
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Using total training steps = epochs * number_of_batches for T_max
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs * len(train_loader), 
        eta_min=eta_min
    )

    # Loss function
    # We use the same criterion as in the original code
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.preprocessor.PAD_TOKEN)

    print(f'Starting training from step {start_global_step}...')
    global_step = start_global_step

    # ----------------------------------------------------------------------
    #  Variables to track lagged time measurements (similar to second sample)
    # ----------------------------------------------------------------------
    t_cycle_end_prev = None        # End time of the previous iteration
    time_load_next_batch_prev = 0.0
    model.train()

    # Initialize metrics aggregation
    metrics_aggregator = MetricsAggregation()

    for epoch in range(epochs):

        for x, y, attn_mask, padding_mask in train_loader:
            # ---------------------------------------
            # (A) Measure time to load next batch
            # ---------------------------------------
            t_cycle_start = time.time()
            if t_cycle_end_prev is None:
                # First iteration (no previous cycle)
                time_load_next_batch_current = 0.0
            else:
                time_load_next_batch_current = t_cycle_start - t_cycle_end_prev

            # ---------------------------------------
            # (B) Critical training steps
            # ---------------------------------------
            t0 = time.time()
            x = x.to(device)
            y = y.to(device)
            attn_mask = attn_mask.to(device)
            padding_mask = padding_mask.to(device)
            t_data_to_device = time.time() - t0

            t1 = time.time()
            output = model(x, attn_mask=attn_mask, padding_mask=padding_mask, act_dir=output_dir)
            t_forward = time.time() - t1

            t2 = time.time()
            if train_mode == "incremental":
                # Use only the last step for loss
                last_logits = output[:, -1, :]
                last_target = y[:, -1]
                loss = criterion(last_logits, last_target)
            else:  # "multi-step"
                # Compute cross-entropy across entire sequence
                logits = output.transpose(1, 2)  # [batch, classes, seq]
                loss = criterion(logits, y)
            t_loss = time.time() - t2

            t3 = time.time()
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            # print("--->encoder:")
            # for name, params in model.named_parameters():
            #     print("-->name:",name, "-->max_grad:", params.grad.max(), "-->min_grad:", params.grad.min())
            if verbose_acts:
                csv_log_dict = {}
                for name, params in model.named_parameters():
                    csv_log_dict[f"{name}_max"] = params.grad.max().item()
                    csv_log_dict[f"{name}_min"] = params.grad.min().item()
                log_to_csv(os.path.join(output_dir,"gradients.csv"),csv_log_dict)
            t_backward = time.time() - t3

            t4 = time.time()
            optimizer.step()
            t_optimizer_step = time.time() - t4

            t5 = time.time()
            #memory clean up
            loss_val = loss.cpu().item()
            accuracy = calculate_accuracy(logits, y, padding_mask)
            del loss, output, x, y, attn_mask, padding_mask, logits
            torch.cuda.empty_cache()
            t_clear_mem = time.time() - t5


            # Update step count
            global_step += 1


            t_save_model = None
            if (global_step % save_every) == 0:
                t_save = time.time()
                checkpoint_name = f"{base_model_name}_{global_step}.pt"
                checkpoint_path = os.path.join(output_dir, checkpoint_name)
                torch.save(model.state_dict(), checkpoint_path)
                t_save_model = time.time() - t_save
                print(f"[INFO] Saved checkpoint at: {checkpoint_path}")

            # Prepare metrics for accumulation
            step_metrics = {
                'train_loss': loss_val,
                'train_acc': accuracy,
                'lr': scheduler.get_last_lr()[0],
                'time/data_to_device': t_data_to_device,
                'time/forward_pass': t_forward,
                'time/loss_computation': t_loss,
                'time/backward_pass': t_backward,
                'time/optimizer_step': t_optimizer_step,
                "time/memory_clear": t_clear_mem,
                'time/load_next_batch': time_load_next_batch_prev
            }
            if t_save_model is not None:
                step_metrics['time/model_save_to_disk'] = t_save_model

            # Accumulate metrics
            metrics_aggregator.accumulate(step_metrics)

            # Log aggregated metrics every val_every steps
            if (global_step % val_every) == 0:
                # Get aggregated training metrics
                aggregated_train_metrics = metrics_aggregator.aggregate('mean')

                # Run validation and get validation metrics
                print(f"[INFO] Running validation at step {global_step}...")
                val_metrics = validation_step(model, val_loader, criterion, device, train_mode)

                # Combine training and validation metrics
                combined_metrics = {**aggregated_train_metrics, **val_metrics}
                combined_metrics['global_step'] = global_step

                # Log to wandb with explicit step
                wandb.log(combined_metrics, step=global_step)

                # Clear training metrics accumulator
                metrics_aggregator.clear()

                print(f"[INFO] Logged aggregated metrics at step {global_step}: "
                      f"train_loss={aggregated_train_metrics['train_loss']:.4f}, "
                      f"val_loss={val_metrics['val_loss']:.4f}")
            
            if global_step > warmup_steps:
                scheduler.step()

            print(
                f"[Global Step {global_step}] "
                f"Loss: {loss_val:.6f} | "
                f"Acc: {accuracy:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} "
                f"Epoch: {epoch+1}/{epochs} "
            )
            if verbose:
                print(
                    f"Data->Device: {t_data_to_device:.4f}s | "
                    f"Forward: {t_forward:.4f}s | "
                    f"Loss: {t_loss:.4f}s | "
                    f"Backward: {t_backward:.4f}s | "
                    f"Optimizer Step: {t_optimizer_step:.4f}s"
                )

            time_load_next_batch_prev = time_load_next_batch_current
            # if global_step >= training_steps:
            #     break
            t_cycle_end_prev = time.time()


    wandb.finish()
    # Final save after all epochs
    checkpoint_name = f"{base_model_name}_{global_step}.pt"
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] Saved final checkpoint at: {checkpoint_path}")
    




def train_model_multi_gpu(model, train_loader, epochs, train_mode="incremental"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    train_model(model, train_loader, epochs, device, train_mode=train_mode)

