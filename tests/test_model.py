import unittest
import torch
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import DecoderOnlyTransformer

class TestModel(unittest.TestCase):

    def test_model_initialization(self):
        """
        Ensure the model initializes without error and has the correct types of layers.
        """
        model = DecoderOnlyTransformer(num_layers=2, model_dim=64, num_heads=4, hidden_dim=128, quantized_classes=103)
        self.assertIsInstance(model, DecoderOnlyTransformer)
        self.assertTrue(hasattr(model, 'embedding'))
        self.assertTrue(hasattr(model, 'pos_encoding'))
        self.assertTrue(hasattr(model, 'fc_out'))
        self.assertEqual(len(model.layers), 2, "Number of layers is incorrect.")

    def test_forward_pass_shape(self):
        """
        Check that the model forward pass produces the correct output shape.
        """
        batch_size = 4
        seq_len = 10
        quantized_classes = 103

        model = DecoderOnlyTransformer(num_layers=2, model_dim=64, num_heads=4, hidden_dim=128, quantized_classes=quantized_classes)

        # Example input: shape [batch_size, seq_len]
        x = torch.randint(low=0, high=quantized_classes, size=(batch_size, seq_len))
        output = model(x)

        # Expected output shape: [batch_size, seq_len, quantized_classes]
        self.assertEqual(output.shape, (batch_size, seq_len, quantized_classes),
                         f"Output shape {output.shape} does not match expected {(batch_size, seq_len, quantized_classes)}")

    def test_forward_pass_different_positions(self):
        """
        Basic check that positional encoding differs for different positions.
        We'll run the model on two sequences that have the same tokens but shifted positions,
        and verify that the output differs.
        """
        model = DecoderOnlyTransformer(num_layers=2, model_dim=64, num_heads=4, hidden_dim=128, quantized_classes=103)
        model.eval()

        # Sequence A: tokens [10, 20, 30, 40]
        # Sequence B: same tokens but shifted [0, 10, 20, 30], i.e., there's a leading 0
        seq_a = torch.tensor([[10, 20, 30, 40]])
        seq_b = torch.tensor([[0, 10, 20, 30]])

        out_a = model(seq_a)  # shape: [1, 4, 103]
        out_b = model(seq_b)  # shape: [1, 4, 103]

        # Because of positional encoding, out_a should differ from out_b
        # We don't need them to be extremely different, but they should not be identical.
        self.assertFalse(torch.allclose(out_a, out_b), 
                         "Outputs for differently positioned sequences should differ due to positional encoding.")

if __name__ == "__main__":
    unittest.main()
