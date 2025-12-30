import argparse
import random
import torch
import os

# DeepSpeed
import deepspeed

# Local imports from your project
from data import (generate_time_series, TSPreprocessor, MultiTimeSeriesDataset,
                  AutoregressiveLoader, MultiStepLoader)
from model import DecoderOnlyTransformer
from dist_train import train_model_deepseek  # The new deepseek training function you wrote


'''
TODO:
1. Move all training configurations to a yaml file
'''

def main_worker(args):
    """
    Worker function invoked on each GPU process. Here, we:
      1) Build the dataset, data loader, and model.
      2) Call train_model_deepseek for multi-GPU training.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Each rank writes to its own file
    log_file = f"{args.run_name}_{args.local_rank}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Log the device for debugging (one line per GPU process)
    #print(f"[Worker] Using device: cuda (local rank = {os.environ.get('LOCAL_RANK', '0')})")
    print(f"[Worker] Using device: cuda (local rank = {local_rank})")

    # 1) Create dataset & loader
    dataset = MultiTimeSeriesDataset(
        data_dir=args.data_dir,
        num_samples_per_file=args.samples_per_file,
        max_training_length=args.max_training_length
    )

    # Autoregressive loader for training
    loader = AutoregressiveLoader(dataset, batch_size=args.batch_size)

    # 2) Create model
    model = DecoderOnlyTransformer(
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        quantized_classes=dataset.preprocessor.vocab_size,
        padding_idx=dataset.preprocessor.PAD_TOKEN
    )

    # 3) Call deepseek training loop
    train_model_deepseek(
        model,
        logger=logger,
        train_loader=loader,
        epochs=args.epochs,
        train_mode=args.train_mode,
        lr=args.lr,
        save_every=args.save_every,
        verbose=args.verbose,
        base_dir=args.base_dir,
        base_model_name=args.base_model_name,
        eta_min=args.eta_min,
        warmup_steps=args.warmup_steps,
        run_name=args.run_name,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


def main():
    """
    Main entry point.
    Parses arguments, then launches distributed training with DeepSpeed.
    """
    parser = argparse.ArgumentParser(description="Train a Decoder-Only Transformer model.")

    # Model arguments
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--model_dim", type=int, default=32, help="Dimensionality of the model.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of the feed-forward layer.")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--train_mode", type=str, default="incremental", choices=["incremental", "multi-step"],
                        help="Training mode for the loop.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Min LR for cosine annealing.")
    parser.add_argument("--warmup_steps", type=int, default=250, help="Number of warmup steps for LR.")
    parser.add_argument("--save_every", type=int, default=100, help="Save a checkpoint every N steps.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10,
                        help="Local steps of gradient accumulation before syncing grads.")

    # Checkpoint & logging arguments
    parser.add_argument("--base_dir", type=str, default="checkpoints", help="Directory for saving checkpoints.")
    parser.add_argument("--base_model_name", type=str, default="model", help="Base name for checkpoint files.")
    parser.add_argument("--run_name", type=str, default="tsfm", help="W&B Run Name.")

    # Dataset arguments
    parser.add_argument("--max_training_length", type=int, default=64, help="Max sequence length for training data.")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the preprocessed tensor file.")
    parser.add_argument("--samples_per_file", type=int, default=1000,
                        help="Number of samples to create from a single .pt file in the data_dir")
    parser.add_argument("--autoreg_expansion_factor", type=int, default=50, help="When using autoregressive training, the factor by which the total samples will expand")

    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')

    # Additional multi-GPU argument
    #parser.add_argument("--data_parallel_workers", type=int, default=2,
    #                    help="Number of GPUs to use for data-parallel training")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # We pass the entire `args` to the DeepSpeed launcher
    # This spawns multiple processes, each calling `main_worker(args)`
    #deepspeed.run(
    #    main_worker,
    #    args=(args,),           # Must pass as tuple
    #    num_gpus=args.data_parallel_workers
    #)
    main_worker(args)


if __name__ == "__main__":
    main()
