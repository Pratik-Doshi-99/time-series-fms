from typing import Dict, List


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