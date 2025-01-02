# data.py

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
    # Now these tokens are >= num_classes so they do NOT interfere with data range [0..num_classes-1]
    BOS_TOKEN = 100  # Beginning of Sequence
    EOS_TOKEN = 101  # End of Sequence
    PAD_TOKEN = 102  # Padding Token

    def __init__(self, num_classes, add_bos=False, add_eos=False):
        self.num_classes = num_classes
        self.add_bos = add_bos
        self.add_eos = add_eos

        # We want the data quantized into [0 .. num_classes-1].
        # For example, if num_classes=100, that is [0..99].
        self.range_min = 0
        self.range_max = self.num_classes - 1  # e.g. 99
        # Precision of 1.0 makes each normalized step map to an integer bin.
        self.range_precision = 1.0

    def normalize_vector(self, vector):
        """
        Normalize time series into [0,1].
        """
        min_val = np.min(vector)
        max_val = np.max(vector)
        mean_val = np.mean(vector)
        std_val = np.std(vector)

        # Avoid division-by-zero if the series is constant
        if max_val - min_val == 0:
            # Return a zero array if the entire vector is constant
            normalized_vector = np.zeros_like(vector)
        else:
            normalized_vector = (vector - min_val) / (max_val - min_val)

        return normalized_vector, min_val, max_val, mean_val, std_val

    def quantize_vector(self, normalized_vector):
        """
        Quantize into integer values [0..num_classes-1].
        """
        # Scale [0,1] -> [0..(range_max - range_min)] = [0..(num_classes-1)]
        scaled_vector = normalized_vector * (self.range_max - self.range_min)
        # Round to nearest integer
        quantized_vector = np.round(scaled_vector)
        # Clip to ensure no values go beyond [0..(num_classes-1)]
        quantized_vector = np.clip(quantized_vector, self.range_min, self.range_max)
        return quantized_vector

    def preprocess_series(self, series):
        normalized_segment, min_val, max_val, mean_val, std_val = self.normalize_vector(series)
        quantized_segment = self.quantize_vector(normalized_segment)

        sequence = quantized_segment.tolist()
        # Insert special tokens if requested
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
        return torch.tensor(sequence, dtype=torch.long), metadata
    
    def dequantize_series(self, quantized_series, metadata):
        """
        Reverses the quantization and normalization to reconstruct approximate original values.
        """
        # Identify valid data indices (skip if they are special tokens)
        data_indices = (quantized_series < self.num_classes)
        data_only = quantized_series[data_indices]#.float()

        # Convert from [0..(num_classes-1)] back to [0..1]
        normalized_vector = data_only / (self.num_classes - 1)

        # Scale back to original range
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

                # Build all possible next-step training pairs
                for i in range(1, len(sample)):
                    # (Optional) Skip if sample[0] is BOS token, etc.
                    if i == 1 and sample[0] == TSPreprocessor.BOS_TOKEN:
                        continue


                    x = sample[:i]
                    y = sample[i:i+1]

                    # Make sure these are longs for nn.Embedding + nn.CrossEntropyLoss
                    x_tensor = x.long()
                    y_tensor = y.long()

                    if len(batched_x) < self.batch_size:
                        batched_x.append(x_tensor)
                        batched_y.append(y_tensor)
                    else:
                        self.cache.append((x_tensor, y_tensor))

        if batched_x:
            # Pad sequences
            x = torch.nn.utils.rnn.pad_sequence(
                batched_x, 
                batch_first=True, 
                padding_value=TSPreprocessor.PAD_TOKEN
            )
            y = torch.nn.utils.rnn.pad_sequence(
                batched_y, 
                batch_first=True, 
                padding_value=TSPreprocessor.PAD_TOKEN
            )

            # Create autoregressive masks
            attn_mask_auto = torch.triu(torch.ones((x.shape[1], x.shape[1]))).bool()
            attn_mask_auto = ~attn_mask_auto  # Invert to ensure the upper triangular is False

            # For key padding mask: True means "ignore this position"
            padding_mask = (x == TSPreprocessor.PAD_TOKEN)

            return (
                x,   # [batch_size, seq_len]
                y,   # [batch_size, seq_len]
                attn_mask_auto,  # [seq_len, seq_len]  upper triangular masked
                padding_mask      # [batch_size, seq_len] True where PAD
            )

        raise StopIteration



class MultiStepLoader:
    """
    Produces (X, Y) pairs for the full sequence in a single pass, 
    suitable for "multi-step" teacher-forcing training.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __next__(self):
        """
        Collect a batch of entire sequences (up to max_training_length).
        X = entire sequence minus the last token
        Y = entire sequence minus the first token
        If shorter than 2 tokens, we skip it.
        """
        batched_x, batched_y = [], []

        while len(batched_x) < self.batch_size:
            if self.current_index >= len(self.dataset):
                if batched_x:
                    # Return whatever we have in batched_x
                    break
                else:
                    # No more data
                    self.current_index = 0
                    raise StopIteration

            sequence, _ = self.dataset[self.current_index]
            self.current_index += 1

            if len(sequence) < 2:
                # Not enough tokens to form X->Y
                continue

            # For multi-step approach:
            # X = [x0, x1, ..., x_{n-2}, x_{n-1}]
            # Y = [x1, x2, ..., x_{n-1}, x_{n}] 
            # But if you want to keep it exact, you can do:
            # X = sequence[:-1], Y = sequence[1:]
            # (or keep them same length & rely on shifting in the model; 
            #  depends on preference)

            # We’ll just do full length = len(sequence)
            # X is the entire sequence except the last token
            # Y is the entire sequence except the first token
            X = sequence[:-1]
            Y = sequence[1:]

            batched_x.append(X)
            batched_y.append(Y)

        if not batched_x:
            raise StopIteration

        # Now we have up to batch_size sequences. Pad them
        x_padded = torch.nn.utils.rnn.pad_sequence(
            batched_x,
            batch_first=True,
            padding_value=TSPreprocessor.PAD_TOKEN
        )  # shape [batch_size, max_seq_len]
        y_padded = torch.nn.utils.rnn.pad_sequence(
            batched_y,
            batch_first=True,
            padding_value=TSPreprocessor.PAD_TOKEN
        )  # shape [batch_size, max_seq_len]

        # Create a causal attention mask for the max_seq_len
        seq_len = x_padded.shape[1]
        attn_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool))
        attn_mask = ~attn_mask  # invert => lower-triangular False => allowed

        # Key padding mask (True = ignore)
        padding_mask = (x_padded == TSPreprocessor.PAD_TOKEN)

        return x_padded, y_padded, attn_mask, padding_mask