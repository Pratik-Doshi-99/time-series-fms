# main.py

from data import (generate_time_series, TSPreprocessor, MultiTimeSeriesDataset, 
                  AutoregressiveLoader, MultiStepLoader)
from model import DecoderOnlyTransformer
from train import train_model
import random
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Decoder-Only Transformer model.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--model_dim", type=int, default=32, help="Dimensionality of the model.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of the feed-forward layer.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument("--max_training_length", type=int, default=64, help="Max sequence length for training data.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--save_every", type=int, default=100, help="Save a checkpoint every N steps.")
    parser.add_argument("--train_mode", type=str, default="incremental", choices=["incremental", "multi-step"],
                        help="Training mode for the loop.")
    parser.add_argument("--base_dir", type=str, default="checkpoints", help="Directory where checkpoints are saved.")
    parser.add_argument("--base_model_name", type=str, default="model", help="Base name for checkpoint files.")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Min LR for cosine annealing.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to the preprocessed tensor file.")
    parser.add_argument("--samples_per_file", type=int, default=1000, help="The number of samples to create from a single .pt file in the data_dir")
    parser.add_argument("--warmup_steps", type=int, default=250, help="Number of warmup steps for LR.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu or cuda). If None, auto-detect.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--autoreg_expansion_factor", type=int, default=50, help="When using autoregressive training, the factor by which the total samples will expand")
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create dataset & loader
    dataset = MultiTimeSeriesDataset(
        data_dir=args.data_dir,
        num_samples_per_file=args.samples_per_file,
        max_training_length=args.max_training_length
    )
    loader = AutoregressiveLoader(dataset, batch_size=args.batch_size)
    print(dataset.preprocessor.vocab_size, dataset.preprocessor.PAD_TOKEN)
    # Create model
    model = DecoderOnlyTransformer(
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        quantized_classes=dataset.preprocessor.vocab_size,
        padding_idx=dataset.preprocessor.PAD_TOKEN
    )

    

    # Train
    train_model(
        model,
        loader,
        device=device,
        epochs=args.epochs,
        train_mode=args.train_mode,
        lr=args.lr,
        save_every=args.save_every,
        verbose=args.verbose,
        base_dir=args.base_dir,
        base_model_name=args.base_model_name,
        eta_min=args.eta_min,
        warmup_steps=args.warmup_steps,
        
    )
