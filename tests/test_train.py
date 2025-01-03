import unittest
import torch
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import DecoderOnlyTransformer
from train import train_model
from data import TSPreprocessor

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

        # Create a mini dataset directly (e.g. 8 samples, each of length 5)
        # We'll store it as a list of (x, y, attn_mask, padding_mask) for simplicity
        # or create a dummy loader class inline here.
        def dummy_loader():
            # We'll produce 2 batches total
            for _ in range(2):
                batch_size = 4
                seq_len = 5
                x = torch.randint(0, 103, (batch_size, seq_len)).to(device)
                y = torch.randint(0, 103, (batch_size, seq_len)).to(device)

                # Triangular mask: shape [seq_len, seq_len]
                attn_mask = ~torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
                # Padding mask: shape [batch_size, seq_len]
                padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

                yield x, y, attn_mask, padding_mask

        # We'll define an inline "loader" object
        class DummyLoaderWrapper:
            def __iter__(self):
                return dummy_loader()
            def __len__(self):
                return 2  # we said 2 yields

        loader = DummyLoaderWrapper()

        # Attempt training
        # If it doesn't crash, we consider the test passed.
        train_model(model, loader, epochs=2, device=device)
        # We do a small assertion on model parameters if we want to see if they're updated
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} did not receive a gradient.")

if __name__ == "__main__":
    unittest.main()
