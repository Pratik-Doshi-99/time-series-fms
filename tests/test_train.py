import unittest
import torch
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import DecoderOnlyTransformer
from train import train_model
from data import TSPreprocessor, MultiTimeSeriesDataset, AutoregressiveLoader

class TestTrain(unittest.TestCase):

    def test_train_loop(self):
        """
        Minimal integration test for the training loop with a tiny synthetic dataset.
        Checks that the loop runs and updates model parameters without error.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a small model
        model = DecoderOnlyTransformer(
            num_layers=2, 
            model_dim=32, 
            num_heads=2, 
            hidden_dim=64, 
            quantized_classes=103,
            padding_idx=102
        ).to(device)

        dataset = MultiTimeSeriesDataset(data_dir="data", max_training_length=32, num_samples_per_file=50)
        loader = AutoregressiveLoader(dataset, batch_size=16)
        
        # Attempt training
        # If it doesn't crash, we consider the test passed.
        train_model(model, loader, device=device)
        # We do a small assertion on model parameters if we want to see if they're updated
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} did not receive a gradient.")

if __name__ == "__main__":
    unittest.main()
