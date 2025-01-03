# main.py

from data import (generate_time_series, TSPreprocessor, MultiTimeSeriesDataset, 
                  AutoregressiveLoader, MultiStepLoader)
from model import DecoderOnlyTransformer
from train import train_model
import random
import torch

if __name__ == "__main__":
    # CHOOSE TRAIN MODE
    # multi-step or incremental
    train_mode = "incremental"
    #train_mode = "multi-step"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_series = 1000
    series_length = 128
    max_training_length = 64
    num_classes = 100

    # Generate synthetic data
    series_list = [
        generate_time_series(
            series_length,
            drift=random.uniform(0, 0.2),
            cycle_amplitude=random.uniform(0.5, 1.5),
            noise_std=random.uniform(0.1, 0.3),
            trend_slope=random.uniform(0, 0.05),
            frequency=random.uniform(0.5, 2.0),
            bias=random.uniform(-10, 10)
        )
        for _ in range(num_series)
    ]

    # Preprocess data
    preprocessor = TSPreprocessor(num_classes=num_classes, add_bos=True, add_eos=True)
    preprocessed_tensors = []
    metadata = []

    for series in series_list:
        tensor, meta = preprocessor.preprocess_series(series)
        preprocessed_tensors.append(tensor)
        metadata.append(meta)

    preprocessor.save_preprocessed(preprocessed_tensors, metadata, "preprocessed_data.pt")

    # Create Dataset
    dataset = MultiTimeSeriesDataset(
        tensor_file_path="preprocessed_data.pt", 
        max_training_length=max_training_length
    )

    # Depending on train_mode, choose loader
    if train_mode == "incremental":
        print("Using incremental step-by-step approach (AutoregressiveLoader).")
        train_loader = AutoregressiveLoader(dataset, batch_size=64)
    else:
        print("Using multi-step approach (MultiStepLoader).")
        train_loader = MultiStepLoader(dataset, batch_size=64)

    # Build model
    model = DecoderOnlyTransformer(
        num_layers=4, 
        model_dim=32, 
        num_heads=4, 
        hidden_dim=128, 
        quantized_classes=preprocessor.vocab_size,
        padding_idx=preprocessor.PAD_TOKEN
    )

    # Train
    print(f"Training model with train_mode={train_mode}...")
    train_model(model, train_loader, epochs=2, device=device, train_mode=train_mode)
