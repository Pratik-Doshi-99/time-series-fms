import unittest
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules from the parent directory
from data import generate_time_series, TSPreprocessor, MultiTimeSeriesDataset, AutoregressiveLoader

class TestData(unittest.TestCase):

    def test_generate_time_series(self):
        series = generate_time_series(length=100, drift=0.1, cycle_amplitude=1.0, noise_std=0.2, trend_slope=0.05, frequency=2.0, bias=10.0)
        self.assertEqual(len(series), 100)
        self.assertTrue(isinstance(series, np.ndarray))

    # -----------------------------------------------------------------------------------
    # NEW TEST: Edge case - length=1
    # -----------------------------------------------------------------------------------
    def test_generate_time_series_length_one(self):
        series = generate_time_series(length=1, drift=0.1, cycle_amplitude=1.0, noise_std=0.2, trend_slope=0.05, frequency=2.0, bias=10.0)
        self.assertEqual(len(series), 1, "Series of length=1 should have exactly 1 data point")
        self.assertTrue(isinstance(series, np.ndarray))

    # -----------------------------------------------------------------------------------
    # NEW TEST: Constant series (no drift, amplitude, noise, slope)
    # -----------------------------------------------------------------------------------
    def test_constant_series_quantization_dequantization(self):
        # Generate a constant series
        series = np.full(50, 5.0)  # all 5.0
        preprocessor = TSPreprocessor(num_classes=100, add_bos=False, add_eos=False)
        tensor, metadata = preprocessor.preprocess_series(series)
        # Dequantize
        dequantized_series = preprocessor.dequantize_series(tensor.numpy(), metadata)

        # Check we didn't produce NaNs or anything weird
        self.assertFalse(np.isnan(dequantized_series).any(), "Dequantized constant series should not contain NaNs.")

        # The difference should be minimal because it's constant
        # Tolerance can be quite low, e.g., 0.001 if truly constant
        self.assertTrue(np.allclose(series, dequantized_series, atol=0.001),
                        "Dequantized constant series should closely match the original.")

    # The rest of your original tests below...
    # -----------------------------------------------------------------------------------
    # def test_quantization_visualization(self):
    #     ...

    def test_quantization_dequantization_visualization(self):
        series = generate_time_series(length=100, drift=0.1, cycle_amplitude=1.0, noise_std=0.2, trend_slope=0.05, frequency=2.0, bias=10.0)
        preprocessor = TSPreprocessor(num_classes=100, add_bos=False, add_eos=False)
        tensor, metadata = preprocessor.preprocess_series(series)

        # Extract the quantized series
        quantized_series = tensor.numpy()

        # Create two parallel plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Plot original time series
        axs[0].plot(series, color="blue")
        axs[0].set_title("Original Time Series")
        axs[0].set_xlabel("Time Steps")
        axs[0].set_ylabel("Value")
        axs[0].grid(True)

        # Plot quantized series
        axs[1].plot(quantized_series, color="orange")
        axs[1].set_title("Quantized Time Series")
        axs[1].set_xlabel("Time Steps")
        axs[1].set_ylabel("Quantized Value")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

        # Assertions
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertTrue("min_val" in metadata and "max_val" in metadata)

    def test_quantization_visualization(self):
        series = generate_time_series(length=100, drift=0.1, cycle_amplitude=1.0, noise_std=0.2, trend_slope=0.05, frequency=2.0, bias=10.0)
        preprocessor = TSPreprocessor(num_classes=100, add_bos=False, add_eos=False)
        tensor, metadata = preprocessor.preprocess_series(series)

        # Dequantize using the new function
        dequantized_series = preprocessor.dequantize_series(tensor, metadata)

        # Plot the raw and dequantized series
        plt.figure(figsize=(12, 6))
        plt.plot(series, label="Raw Time Series", color="blue")
        plt.plot(dequantized_series, label="Dequantized Time Series", color="orange")
        plt.title("Raw vs Dequantized Time Series")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Assertions
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertTrue("min_val" in metadata and "max_val" in metadata)

    def test_quantization_and_dequantization(self):
        series = generate_time_series(length=100, drift=0.1, cycle_amplitude=1.0, noise_std=0.2, trend_slope=0.05, frequency=2.0, bias=10.0)
        preprocessor = TSPreprocessor(num_classes=100, add_bos=False, add_eos=False)
        tensor, metadata = preprocessor.preprocess_series(series)

        # Dequantize and compare
        dequantized_series = preprocessor.dequantize_series(tensor.numpy(), metadata)
        self.assertTrue(np.allclose(series, dequantized_series, atol=0.1))

    def test_file_saving(self):
        # Generate 1000 mock time series
        num_series = 1000
        series_length = 100
        preprocessor = TSPreprocessor(num_classes=100, add_bos=True, add_eos=True)
        preprocessed_tensors = []
        metadata = []

        for _ in range(num_series):
            series = generate_time_series(length=series_length, drift=0.1, cycle_amplitude=1.0, noise_std=0.2, trend_slope=0.05, frequency=2.0, bias=10.0)
            tensor, meta = preprocessor.preprocess_series(series)
            preprocessed_tensors.append(tensor)
            metadata.append(meta)

        # Save the tensors and metadata
        test_file_path = "test_preprocessed_data.pt"
        preprocessor.save_preprocessed(preprocessed_tensors, metadata, test_file_path)

        # Check if file is created
        self.assertTrue(os.path.exists(test_file_path), "The preprocessed file was not created.")

        # Load the file and verify its contents
        loaded_data = torch.load(test_file_path)
        self.assertIn("tensors", loaded_data, "Saved file does not contain 'tensors'.")
        self.assertIn("metadata", loaded_data, "Saved file does not contain 'metadata'.")

        # Validate the saved tensors and metadata
        loaded_tensors = loaded_data["tensors"]
        loaded_metadata = loaded_data["metadata"]
        self.assertEqual(len(loaded_tensors), num_series, "Number of saved tensors is incorrect.")
        self.assertEqual(len(loaded_metadata), num_series, "Number of saved metadata entries is incorrect.")

        for i in range(num_series):
            self.assertTrue(torch.equal(preprocessed_tensors[i], loaded_tensors[i]), f"Tensor {i} does not match the original.")
            self.assertEqual(metadata[i], loaded_metadata[i], f"Metadata {i} does not match the original.")

    def test_autoregressive_loader(self):
        dataset = MultiTimeSeriesDataset(tensor_file_path="test_preprocessed_data.pt", max_training_length=32)
        loader = AutoregressiveLoader(dataset, batch_size=8)

        for x, y, attn_mask, padding_mask in loader:
            self.assertTrue(isinstance(x, torch.Tensor))
            self.assertTrue(isinstance(y, torch.Tensor))
            self.assertTrue(isinstance(attn_mask, torch.Tensor))
            self.assertTrue(isinstance(padding_mask, torch.Tensor))

    # -----------------------------------------------------------------------------------
    # NEW TEST: Non-existent file for MultiTimeSeriesDataset
    # -----------------------------------------------------------------------------------
    def test_non_existent_file(self):
        with self.assertRaises(FileNotFoundError):
            MultiTimeSeriesDataset(tensor_file_path="non_existent_file.pt", max_training_length=32)

    def test_attention_masks(self):
        dataset = MultiTimeSeriesDataset(tensor_file_path="test_preprocessed_data.pt", max_training_length=32)
        loader = AutoregressiveLoader(dataset, batch_size=4)
        i = 0
        for x, y, attn_mask, padding_mask in loader:
            i += 1
            max_len = x.size(1)
            self.assertEqual(attn_mask.size(), (max_len, max_len), "Attention mask dimensions are incorrect.")
            self.assertTrue(
                torch.allclose(
                    attn_mask.float(), 
                    (~torch.triu(torch.ones((max_len, max_len))).bool()).float()
                ),
                "Attention mask does not have correct triangular structure."
            )

            self.assertEqual(padding_mask.size(), (x.size(0), x.size(1)), "Padding mask dimensions are incorrect.")

            # If you are using x of shape [batch_size, seq_len], then you won't have x[i, j, 0].
            # Adjust your check accordingly. For example:
            for b_idx in range(x.size(0)):  # batch size
                for s_idx in range(x.size(1)):  # sequence length
                    if x[b_idx, s_idx].item() == TSPreprocessor.PAD_TOKEN:
                        self.assertTrue(padding_mask[b_idx, s_idx].item(), "Padding mask should be True for PAD_TOKEN.")
                    else:
                        self.assertFalse(padding_mask[b_idx, s_idx].item(), "Padding mask should be False for non-PAD_TOKEN.")


if __name__ == "__main__":
    unittest.main()
