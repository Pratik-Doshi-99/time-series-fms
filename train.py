import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, attn_mask, padding_mask in train_loader:
            x, y = x.to(device), y.to(device)
            attn_mask, padding_mask = attn_mask.to(device), padding_mask.to(device)

            optimizer.zero_grad()
            output = model(x, attn_mask=attn_mask, padding_mask=padding_mask)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}")

def train_model_multi_gpu(model, train_loader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    train_model(model, train_loader, epochs, device)

## main.py ##
from data import generate_time_series, TSPreprocessor, MultiTimeSeriesDataset, AutoregressiveLoader
from model import DecoderOnlyTransformer
from train import train_model_multi_gpu
import random

if __name__ == "__main__":
    num_series = 100
    series_length = 500
    max_training_length = 1024
    num_classes = 100

    series_list = [generate_time_series(series_length, drift=random.uniform(0, 0.2),
                                        cycle_amplitude=random.uniform(0.5, 1.5),
                                        noise_std=random.uniform(0.1, 0.3),
                                        trend_slope=random.uniform(0, 0.05),
                                        frequency=random.uniform(0.5, 2.0),
                                        bias=random.uniform(-10, 10)) for _ in range(num_series)]

    preprocessor = TSPreprocessor(num_classes=num_classes, add_bos=True, add_eos=True)
    preprocessed_tensors = []
    metadata = []

    for series in series_list:
        tensor, meta = preprocessor.preprocess_series(series)
        preprocessed_tensors.append(tensor)
        metadata.append(meta)

    preprocessor.save_preprocessed(preprocessed_tensors, metadata, "preprocessed_data.pt")

    dataset = MultiTimeSeriesDataset(tensor_file_path="preprocessed_data.pt", max_training_length=max_training_length)
    train_loader = AutoregressiveLoader(dataset, batch_size=16)

    model = DecoderOnlyTransformer(num_layers=4, model_dim=64, num_heads=4, hidden_dim=128)

    print("Training model:")
    train_model_multi_gpu(model, train_loader, epochs=10)
