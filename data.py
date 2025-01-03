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

import numpy as np
import torch
import random
import os
import math

class TSPreprocessor:
    """
    Preprocessor for time series data, with instance-level special tokens.
    BOS_TOKEN, EOS_TOKEN, and PAD_TOKEN are now instance variables.
    """

    def __init__(self, num_classes, add_bos=False, add_eos=False):
        self.num_classes = num_classes
        self.add_bos = add_bos
        self.add_eos = add_eos

        # In your request, you asked that the special tokens be set to
        # num_classes, num_classes+1, and num_classes+2, respectively.
        self.BOS_TOKEN = self.num_classes
        self.EOS_TOKEN = self.num_classes + 1
        self.PAD_TOKEN = self.num_classes + 2
        self.vocab_size = num_classes + 3

        # Range and quantization
        self.range_min = 0
        self.range_max = self.num_classes - 1
        self.range_precision = 1.0

    def normalize_vector(self, vector):
        min_val = np.min(vector)
        max_val = np.max(vector)
        mean_val = np.mean(vector)
        std_val = np.std(vector)

        if max_val - min_val == 0:
            normalized_vector = np.zeros_like(vector)
        else:
            normalized_vector = (vector - min_val) / (max_val - min_val)

        return normalized_vector, min_val, max_val, mean_val, std_val

    def quantize_vector(self, normalized_vector):
        scaled_vector = normalized_vector * (self.range_max - self.range_min)
        quantized_vector = np.round(scaled_vector)
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
        # skip if they are special tokens
        data_indices = (quantized_series < self.num_classes)
        data_only = quantized_series[data_indices]

        normalized_vector = data_only / (self.num_classes - 1)
        original_series = normalized_vector * (metadata["max_val"] - metadata["min_val"]) + metadata["min_val"]
        return original_series

    def save_preprocessed(self, tensors, metadata, file_path):
        """
        Saves the data plus the Preprocessor attributes in one file.
        This includes the instance-level special tokens.
        """
        data = {
            "tensors": tensors,
            "metadata": metadata,
            "preprocessor_attrs": {
                "num_classes": self.num_classes,
                "add_bos": self.add_bos,
                "add_eos": self.add_eos,
                "range_min": self.range_min,
                "range_max": self.range_max,
                "range_precision": self.range_precision,
                # Instance-level special tokens
                "BOS_TOKEN": self.BOS_TOKEN,
                "EOS_TOKEN": self.EOS_TOKEN,
                "PAD_TOKEN": self.PAD_TOKEN,
            }
        }
        torch.save(data, file_path)

    @staticmethod
    def from_preprocessed_file(file_path):
        """
        Factory method that:
        1) Loads the saved file
        2) Creates a TSPreprocessor with the same attributes
        3) Returns (tensors, metadata, preprocessor_instance)
        """
        loaded_data = torch.load(file_path)
        preprocessor_attrs = loaded_data.get("preprocessor_attrs", {})

        # Reconstruct an instance with the same constructor args
        instance = TSPreprocessor(
            num_classes=preprocessor_attrs.get("num_classes", 100),
            add_bos=preprocessor_attrs.get("add_bos", False),
            add_eos=preprocessor_attrs.get("add_eos", False),
        )

        # # Overwrite the internal range values
        # instance.range_min = preprocessor_attrs.get("range_min", 0)
        # instance.range_max = preprocessor_attrs.get("range_max", instance.num_classes - 1)
        # instance.range_precision = preprocessor_attrs.get("range_precision", 1.0)

        # # Overwrite the instance-level special tokens
        # instance.BOS_TOKEN = preprocessor_attrs.get("BOS_TOKEN", instance.num_classes)
        # instance.EOS_TOKEN = preprocessor_attrs.get("EOS_TOKEN", instance.num_classes + 1)
        # instance.PAD_TOKEN = preprocessor_attrs.get("PAD_TOKEN", instance.num_classes + 2)

        # Extract the loaded data
        tensors = loaded_data["tensors"]
        metadata = loaded_data["metadata"]

        return tensors, metadata, instance


class MultiTimeSeriesDataset:
    def __init__(self, tensor_file_path, max_training_length=1024):
        if not os.path.exists(tensor_file_path):
            raise FileNotFoundError(f"File not found: {tensor_file_path}")
        self.tensors, self.metadata, self.preprocessor = TSPreprocessor.from_preprocessed_file(tensor_file_path)
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
                    if i == 1 and sample[0] == self.dataset.preprocessor.BOS_TOKEN:
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
                padding_value=self.dataset.preprocessor.PAD_TOKEN
            )
            y = torch.nn.utils.rnn.pad_sequence(
                batched_y, 
                batch_first=True, 
                padding_value=self.dataset.preprocessor.PAD_TOKEN
            )

            # Create autoregressive masks
            attn_mask_auto = torch.triu(torch.ones((x.shape[1], x.shape[1])), diagonal=1).bool()
            #attn_mask_auto = ~attn_mask_auto  # Invert to ensure the upper triangular is False

            # For key padding mask: True means "ignore this position"
            padding_mask = (x == self.dataset.preprocessor.PAD_TOKEN)

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
            padding_value=self.dataset.preprocessor.PAD_TOKEN
        )  # shape [batch_size, max_seq_len]
        y_padded = torch.nn.utils.rnn.pad_sequence(
            batched_y,
            batch_first=True,
            padding_value=self.dataset.preprocessor.PAD_TOKEN
        )  # shape [batch_size, max_seq_len]

        # Create a causal attention mask for the max_seq_len
        seq_len = x_padded.shape[1]
        attn_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        #attn_mask = ~attn_mask  # invert => lower-triangular False => allowed

        # Key padding mask (True = ignore)
        padding_mask = (x_padded == self.dataset.preprocessor.PAD_TOKEN)

        return x_padded, y_padded, attn_mask, padding_mask