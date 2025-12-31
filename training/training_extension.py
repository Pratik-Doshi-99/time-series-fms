from typing import Dict, List
import torch


@torch.no_grad()
def calculate_accuracy(predicted_logits, target_classes, pad_mask):
    # predicted_logits: batch, seq, classes
    # target_classes: batch, seq
    # pad_mask: batch, seq
    #print(predicted_logits.shape, target_classes.shape, pad_mask.shape)
    predicted_tokens = predicted_logits.argmax(dim=1) #batch, seq
    total_padded = pad_mask.sum()
    correct_predictions = (predicted_tokens == target_classes) | pad_mask
    accuracy = (correct_predictions.sum() - total_padded) / (target_classes.numel() - total_padded + 1e-6)

    return accuracy.cpu().item()



class MetricsAggregation:
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}

    def accumulate(self, metrics: Dict[str, float]) -> None:
        """
        Accumulate metrics into internal data structure.

        Args:
            metrics: Dictionary mapping metric names to metric values
        """
        for metric_name, metric_value in metrics.items():
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            self._metrics[metric_name].append(metric_value)

    def aggregate(self, reduction: str) -> Dict[str, float]:
        """
        Aggregate accumulated metrics using the specified reduction operation.

        Args:
            reduction: Reduction operation ('min', 'max', or 'mean')

        Returns:
            Dictionary mapping metric names to aggregated values
        """
        if reduction not in ['min', 'max', 'mean']:
            raise ValueError(f"Unsupported reduction operation: {reduction}. Use 'min', 'max', or 'mean'.")

        aggregated = {}
        for metric_name, values in self._metrics.items():
            if not values:
                continue

            if reduction == 'min':
                aggregated[metric_name] = min(values)
            elif reduction == 'max':
                aggregated[metric_name] = max(values)
            elif reduction == 'mean':
                aggregated[metric_name] = sum(values) / len(values)

        return aggregated

    def clear(self) -> None:
        """Clear all accumulated metrics."""
        self._metrics.clear()



if __name__ == '__main__':
    pred = [
        [[0.9,0.1,0.2],[0.8,0.9,0.6],[0.1,0.2,0.3]],
        [[0.8,0.9,0.6],[0.1,0.2,0.3],[0.1,0.2,0.3]]
    ]

    actual = [
        [0,1,2],
        [1,2,2]
    ]

    pad_mask = [
        [0,0,0],
        [0,0,0]
    ]

    pred = torch.tensor(pred, device='cuda:0')
    actual = torch.tensor(actual, device='cuda:0')
    pad_mask = torch.tensor(pad_mask, device='cuda:0')

    accuracy = round(calculate_accuracy(pred, actual, pad_mask), 6)

    print('Accuracy:',accuracy)