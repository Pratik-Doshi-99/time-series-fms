import numpy as np
import torch
import random
import os
import math

def generate_time_series(length, drift=0.0, cycle_amplitude=0.0, noise_std=0.1, trend_slope=0.0, frequency=1.0, bias=0.0):
    time = np.arange(length)
    base_series = np.full(length, bias)
    trend = trend_slope * time
    cycle = cycle_amplitude * np.sin(2 * np.pi * frequency * time / length)
    noise = np.random.normal(0, noise_std, length)
    drift_array = drift * time
    return base_series + trend + cycle + noise + drift_array

class TSPreprocessor:
    BOS_TOKEN = -101  # Beginning of Sequence
    EOS_TOKEN = -102  # End of Sequence
    PAD_TOKEN = -103  # Padding Token

    # NOTE: DO NOT USE BOS OR EOS TOKENS.
    def __init__(self, num_classes, add_bos=False, add_eos=False):
        self.num_classes = num_classes
        self.add_bos = False
        self.add_eos = False
        self.range_min = -100
        self.range_max = 100
        self.range_precision = (self.range_max - self.range_min) / num_classes

    def normalize_vector(self, vector):
        min_val = np.min(vector)
        max_val = np.max(vector)
        mean_val = np.mean(vector)
        std_val = np.std(vector)
        normalized_vector = (vector - min_val) / (max_val - min_val)
        return normalized_vector, min_val, max_val, mean_val, std_val

    def quantize_vector(self, normalized_vector):
        quantized_levels = np.arange(self.range_min, self.range_max + self.range_precision, self.range_precision)
        quantized_vector = self.range_min + np.round((normalized_vector * (self.range_max - self.range_min)) / self.range_precision) * self.range_precision
        return np.clip(quantized_vector, self.range_min, self.range_max)

    def preprocess_series(self, series):
        normalized_segment, min_val, max_val, mean_val, std_val = self.normalize_vector(series)
        quantized_segment = self.quantize_vector(normalized_segment)

        sequence = quantized_segment.tolist()
        if self.add_bos:
            sequence.insert(0, self.BOS_TOKEN)
        if self.add_eos:
            sequence.append(self.EOS_TOKEN)

        metadata = {
            "min_val": min_val,
            "max_val": max_val,
            "mean_val": mean_val,
            "std_val": std_val
        }
        return torch.tensor(sequence, dtype=torch.float32), metadata
    
    def dequantize_series(self, quantized_series, metadata):
        normalized_vector = (quantized_series - self.range_min) / (self.range_max - self.range_min)
        original_series = normalized_vector * (metadata["max_val"] - metadata["min_val"]) + metadata["min_val"]
        return original_series

    def save_preprocessed(self, tensors, metadata, file_path):
        data = {
            "tensors": tensors,
            "metadata": metadata
        }
        torch.save(data, file_path)

class MultiTimeSeriesDataset:
    def __init__(self, tensor_file_path, max_training_length=1024):
        if not os.path.exists(tensor_file_path):
            raise FileNotFoundError(f"File not found: {tensor_file_path}")
        data = torch.load(tensor_file_path)
        self.tensors = data["tensors"]
        self.metadata = data["metadata"]
        self.max_training_length = max_training_length

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        full_sequence = self.tensors[index]
        start_idx = random.randint(0, len(full_sequence) - 1)
        return full_sequence[start_idx:start_idx + self.max_training_length], self.metadata[index]

class AutoregressiveLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.cache = []
        self.current_index = 0

    def __iter__(self):
        return self
    
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __next__(self):
        batched_x, batched_y = [], []

        while len(batched_x) < self.batch_size:
            if self.cache:
                cached_x, cached_y = self.cache.pop(0)
                batched_x.append(cached_x)
                batched_y.append(cached_y)
            else:
                if self.current_index >= len(self.dataset):
                    if batched_x:
                        break
                    self.current_index = 0
                    raise StopIteration

                sample, _ = self.dataset[self.current_index]
                self.current_index += 1

                for i in range(1, len(sample)):
                    if sample[0] == TSPreprocessor.BOS_TOKEN:
                        continue
                    x = sample[:i]
                    y = sample[i:i+1]
                    x_tensor = torch.tensor(x, dtype=torch.float32)
                    y_tensor = torch.tensor(y, dtype=torch.float32)

                    if len(batched_x) < self.batch_size:
                        batched_x.append(x_tensor)
                        batched_y.append(y_tensor)
                    else:
                        self.cache.append((x_tensor, y_tensor))

        if batched_x:
            x = torch.nn.utils.rnn.pad_sequence(batched_x, batch_first=True, padding_value=TSPreprocessor.PAD_TOKEN)
            y = torch.nn.utils.rnn.pad_sequence(batched_y, batch_first=True, padding_value=TSPreprocessor.PAD_TOKEN)
            
            # nn.Transformers and related modules use attention and padding masks. If these masks are boolean tensors, a True will cause that particular value to be ignored
            attn_mask_auto = torch.triu(torch.ones((x.shape[1], x.shape[1]))).bool()
            attn_mask_auto = ~attn_mask_auto #inverting to ensure the upper triangular matrix is False, so that it is used
            padding_mask = x.clone()
            padding_mask = padding_mask.masked_fill(padding_mask != TSPreprocessor.PAD_TOKEN, 0).bool() #ensures the padded values are True

            #x = x.unsqueeze(-1)

            return (
                x,
                y,
                attn_mask_auto,
                padding_mask,
            )

        raise StopIteration
