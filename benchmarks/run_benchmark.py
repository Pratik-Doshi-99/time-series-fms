"""
Benchmark Runner for Time Series Foundation Model

Evaluates model performance across different time series patterns and lengths.
"""

import sys
import os
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from models.model import DecoderOnlyTransformer
from data.dataset import TSPreprocessor


def load_model(
    model_path: str,
    num_layers: int = 4,
    model_dim: int = 128,
    num_heads: int = 8,
    hidden_dim: int = 512,
    num_classes: int = 100,
    device: str = "cuda",
) -> DecoderOnlyTransformer:
    """Load a trained model from checkpoint."""
    model = DecoderOnlyTransformer(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        quantized_classes=num_classes + 3,  # +3 for special tokens
        padding_idx=num_classes + 2,  # PAD token
    )
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_benchmark_category(file_path: str) -> dict:
    """Load a benchmark category file."""
    return torch.load(file_path, weights_only=False)


def compute_metrics(
    model: DecoderOnlyTransformer,
    tensors: list,
    metadata: list,
    context_length: int = 128,
    device: str = "cuda",
    num_classes: int = 100,
) -> dict:
    """
    Compute evaluation metrics for a set of time series.

    Args:
        model: Trained model
        tensors: List of preprocessed time series tensors
        metadata: List of metadata dicts
        context_length: Number of tokens to use as context for prediction
        device: Device to run on
        num_classes: Number of quantization classes

    Returns:
        Dictionary containing various metrics
    """
    total_correct = 0
    total_predictions = 0
    total_mae = 0.0
    total_mse = 0.0
    total_cross_entropy = 0.0
    direction_correct = 0
    direction_total = 0

    all_errors = []

    for tensor, meta in zip(tensors, metadata):
        if len(tensor) <= context_length + 1:
            continue

        tensor = tensor.long().to(device)

        # Predict each token after context_length
        for t in range(context_length, len(tensor) - 1):
            context = tensor[:t].unsqueeze(0)

            with torch.no_grad():
                logits = model(context)
                pred_logits = logits[0, -1, :]  # Last position logits

                # Get prediction (argmax)
                pred_token = pred_logits[:num_classes].argmax().item()
                true_token = tensor[t].item()

                # Skip special tokens
                if true_token >= num_classes:
                    continue

                # Accuracy
                if pred_token == true_token:
                    total_correct += 1
                total_predictions += 1

                # MAE and MSE (in quantized space)
                error = abs(pred_token - true_token)
                all_errors.append(error)
                total_mae += error
                total_mse += error ** 2

                # Cross-entropy loss
                ce_loss = F.cross_entropy(
                    pred_logits[:num_classes].unsqueeze(0),
                    torch.tensor([true_token], device=device),
                )
                total_cross_entropy += ce_loss.item()

                # Direction accuracy (did we predict the right trend?)
                if t > context_length:
                    prev_token = tensor[t - 1].item()
                    if prev_token < num_classes:
                        true_direction = true_token - prev_token
                        pred_direction = pred_token - prev_token
                        if (true_direction > 0 and pred_direction > 0) or \
                           (true_direction < 0 and pred_direction < 0) or \
                           (true_direction == 0 and pred_direction == 0):
                            direction_correct += 1
                        direction_total += 1

    if total_predictions == 0:
        return {"error": "No valid predictions made", "total_predictions": 0}

    metrics = {
        "accuracy": total_correct / total_predictions,
        "mae": total_mae / total_predictions,
        "mse": total_mse / total_predictions,
        "rmse": np.sqrt(total_mse / total_predictions),
        "cross_entropy": total_cross_entropy / total_predictions,
        "direction_accuracy": direction_correct / direction_total if direction_total > 0 else 0.0,
        "total_predictions": total_predictions,
        "median_error": np.median(all_errors) if all_errors else 0.0,
        "p90_error": np.percentile(all_errors, 90) if all_errors else 0.0,
        "p95_error": np.percentile(all_errors, 95) if all_errors else 0.0,
    }

    return metrics


def run_benchmark(
    model_path: str,
    benchmark_dir: str,
    output_path: str = None,
    context_length: int = 128,
    device: str = "cuda",
    model_config: dict = None,
) -> dict:
    """
    Run benchmark evaluation across all categories.

    Args:
        model_path: Path to model checkpoint
        benchmark_dir: Directory containing benchmark data files
        output_path: Optional path to save results JSON
        context_length: Context length for predictions
        device: Device to run on
        model_config: Model configuration dict

    Returns:
        Dictionary containing results for all categories
    """
    # Default model config
    if model_config is None:
        model_config = {
            "num_layers": 4,
            "model_dim": 128,
            "num_heads": 8,
            "hidden_dim": 512,
            "num_classes": 100,
        }

    print(f"Loading model from {model_path}...")
    model = load_model(
        model_path=model_path,
        device=device,
        **model_config,
    )

    # Find all benchmark files
    benchmark_files = [
        f for f in os.listdir(benchmark_dir)
        if f.startswith("benchmark_") and f.endswith(".pt") and f != "benchmark_summary.pt"
    ]

    if not benchmark_files:
        raise FileNotFoundError(f"No benchmark files found in {benchmark_dir}")

    results = {
        "model_path": model_path,
        "benchmark_dir": benchmark_dir,
        "context_length": context_length,
        "model_config": model_config,
        "timestamp": datetime.now().isoformat(),
        "categories": {},
    }

    print(f"\nFound {len(benchmark_files)} category files...")
    print("=" * 80)

    for benchmark_file in sorted(benchmark_files):
        category_name = benchmark_file.replace("benchmark_", "").replace(".pt", "")

        file_path = os.path.join(benchmark_dir, benchmark_file)
        data = load_benchmark_category(file_path)

        pattern = data.get("pattern", category_name)
        length_config = data.get("length_config", "unknown")
        series_length = data.get("series_length", "unknown")

        # Skip if series is too short for context
        if isinstance(series_length, int) and series_length <= context_length:
            print(f"\nSkipping {category_name}: series_length ({series_length}) <= context_length ({context_length})")
            continue

        print(f"\n{category_name}:")
        print(f"  Pattern: {pattern}, Length: {series_length}, Samples: {len(data['tensors'])}")

        metrics = compute_metrics(
            model=model,
            tensors=data["tensors"],
            metadata=data["metadata"],
            context_length=context_length,
            device=device,
            num_classes=model_config["num_classes"],
        )

        results["categories"][category_name] = {
            "pattern": pattern,
            "length_config": length_config,
            "series_length": series_length,
            "description": data.get("description", ""),
            "params": data.get("params", {}),
            "metrics": metrics,
        }

        if "error" not in metrics:
            print(f"  Accuracy:           {metrics['accuracy']:.4f}")
            print(f"  MAE:                {metrics['mae']:.4f}")
            print(f"  RMSE:               {metrics['rmse']:.4f}")
            print(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
        else:
            print(f"  Error: {metrics['error']}")

    # Compute aggregate statistics
    valid_results = [
        r for r in results["categories"].values()
        if "error" not in r["metrics"]
    ]

    if valid_results:
        all_accuracies = [r["metrics"]["accuracy"] for r in valid_results]
        all_maes = [r["metrics"]["mae"] for r in valid_results]

        results["aggregate"] = {
            "mean_accuracy": np.mean(all_accuracies),
            "std_accuracy": np.std(all_accuracies),
            "mean_mae": np.mean(all_maes),
            "std_mae": np.std(all_maes),
            "num_categories": len(valid_results),
        }

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS:")
    if "aggregate" in results:
        print(f"  Mean Accuracy: {results['aggregate']['mean_accuracy']:.4f} "
              f"(+/- {results['aggregate']['std_accuracy']:.4f})")
        print(f"  Mean MAE:      {results['aggregate']['mean_mae']:.4f} "
              f"(+/- {results['aggregate']['std_mae']:.4f})")
    else:
        print("  No valid results to aggregate")

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def print_results_table(results: dict):
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print(f"{'Category':<35} {'Length':>8} {'Accuracy':>10} {'MAE':>8} {'RMSE':>8} {'Dir.Acc':>8}")
    print("-" * 100)

    # Sort by pattern then length
    sorted_cats = sorted(
        results["categories"].items(),
        key=lambda x: (x[1].get("pattern", ""), x[1].get("series_length", 0))
    )

    current_pattern = None
    for category, data in sorted_cats:
        pattern = data.get("pattern", category)
        if pattern != current_pattern:
            if current_pattern is not None:
                print("-" * 100)
            current_pattern = pattern

        metrics = data["metrics"]
        series_len = data.get("series_length", "?")

        if "error" in metrics:
            print(f"{category:<35} {series_len:>8} {'SKIP':>10}")
        else:
            print(
                f"{category:<35} "
                f"{series_len:>8} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['mae']:>8.2f} "
                f"{metrics['rmse']:>8.2f} "
                f"{metrics['direction_accuracy']:>8.4f}"
            )

    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on a trained time series model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="benchmark_data",
        help="Directory containing benchmark data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=128,
        help="Context length for predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    # Model configuration
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--model-dim", type=int, default=128, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--hidden-dim", type=int, default=512, help="FFN hidden dimension")
    parser.add_argument("--num-classes", type=int, default=100, help="Number of quantization classes")

    args = parser.parse_args()

    model_config = {
        "num_layers": args.num_layers,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "hidden_dim": args.hidden_dim,
        "num_classes": args.num_classes,
    }

    results = run_benchmark(
        model_path=args.model_path,
        benchmark_dir=args.benchmark_dir,
        output_path=args.output,
        context_length=args.context_length,
        device=args.device,
        model_config=model_config,
    )

    print_results_table(results)