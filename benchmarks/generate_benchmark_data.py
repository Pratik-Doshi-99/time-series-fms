"""
Benchmark Data Generator for Time Series Foundation Model

Generates time series samples across different categories to evaluate
model performance on various patterns and lengths.
"""

import sys
import os
import argparse
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from data.dataset import generate_time_series, TSPreprocessor


# Base pattern configurations (without length)
PATTERN_CONFIGS = {
    # Trend patterns
    "strong_uptrend": {
        "description": "Clear upward linear trend",
        "params": {
            "trend_slope": 0.8,
            "drift": 0.3,
            "cycle_amplitude": 0.0,
            "noise_std": 0.1,
            "frequency": 1.0,
            "bias": 0.0,
        },
    },
    "strong_downtrend": {
        "description": "Clear downward linear trend",
        "params": {
            "trend_slope": -0.8,
            "drift": -0.3,
            "cycle_amplitude": 0.0,
            "noise_std": 0.1,
            "frequency": 1.0,
            "bias": 50.0,
        },
    },
    "flat": {
        "description": "Stationary series with no trend",
        "params": {
            "trend_slope": 0.0,
            "drift": 0.0,
            "cycle_amplitude": 0.0,
            "noise_std": 0.1,
            "frequency": 1.0,
            "bias": 25.0,
        },
    },
    # Cyclical patterns
    "sinusoidal_low_freq": {
        "description": "Slow periodic oscillation (1 cycle)",
        "params": {
            "trend_slope": 0.0,
            "drift": 0.0,
            "cycle_amplitude": 3.0,
            "noise_std": 0.1,
            "frequency": 1.0,
            "bias": 10.0,
        },
    },
    "sinusoidal_high_freq": {
        "description": "Fast periodic oscillation (8 cycles)",
        "params": {
            "trend_slope": 0.0,
            "drift": 0.0,
            "cycle_amplitude": 3.0,
            "noise_std": 0.1,
            "frequency": 8.0,
            "bias": 10.0,
        },
    },
    "weak_seasonality": {
        "description": "Subtle periodic pattern",
        "params": {
            "trend_slope": 0.0,
            "drift": 0.0,
            "cycle_amplitude": 0.5,
            "noise_std": 0.2,
            "frequency": 2.0,
            "bias": 10.0,
        },
    },
    # Noise patterns
    "noise_only": {
        "description": "Pure random noise (no signal)",
        "params": {
            "trend_slope": 0.0,
            "drift": 0.0,
            "cycle_amplitude": 0.0,
            "noise_std": 1.0,
            "frequency": 1.0,
            "bias": 10.0,
        },
    },
    "low_noise": {
        "description": "Very clean signal with minimal noise",
        "params": {
            "trend_slope": 0.2,
            "drift": 0.0,
            "cycle_amplitude": 2.0,
            "noise_std": 0.05,
            "frequency": 2.0,
            "bias": 5.0,
        },
    },
    "high_noise": {
        "description": "Strong signal buried in heavy noise",
        "params": {
            "trend_slope": 0.3,
            "drift": 0.0,
            "cycle_amplitude": 2.0,
            "noise_std": 2.0,
            "frequency": 2.0,
            "bias": 10.0,
        },
    },
    # Combined patterns
    "trend_with_seasonality": {
        "description": "Upward trend with periodic component",
        "params": {
            "trend_slope": 0.5,
            "drift": 0.0,
            "cycle_amplitude": 2.0,
            "noise_std": 0.15,
            "frequency": 3.0,
            "bias": 0.0,
        },
    },
    "noisy_trend": {
        "description": "Trend partially obscured by noise",
        "params": {
            "trend_slope": 0.5,
            "drift": 0.1,
            "cycle_amplitude": 0.0,
            "noise_std": 1.5,
            "frequency": 1.0,
            "bias": 5.0,
        },
    },
    "complex_mixed": {
        "description": "All components present: trend + cycle + noise + drift",
        "params": {
            "trend_slope": 0.3,
            "drift": 0.1,
            "cycle_amplitude": 1.5,
            "noise_std": 0.5,
            "frequency": 4.0,
            "bias": 5.0,
        },
    },
    # Edge cases
    "constant": {
        "description": "Flat line (all values identical)",
        "params": {
            "trend_slope": 0.0,
            "drift": 0.0,
            "cycle_amplitude": 0.0,
            "noise_std": 0.0,
            "frequency": 1.0,
            "bias": 50.0,
        },
    },
    "random_walk_like": {
        "description": "Drift-dominated movement (like random walk)",
        "params": {
            "trend_slope": 0.0,
            "drift": 0.5,
            "cycle_amplitude": 0.0,
            "noise_std": 0.8,
            "frequency": 1.0,
            "bias": 0.0,
        },
    },
}

# Length configurations
LENGTH_CONFIGS = {
    "short": {
        "description": "Short series",
        "length": 64,
    },
    "medium": {
        "description": "Medium series",
        "length": 256,
    }
}


def get_benchmark_category_name(pattern: str, length_config: str) -> str:
    """Generate a combined category name from pattern and length."""
    return f"{pattern}_{length_config}"


def build_benchmark_categories() -> dict:
    """
    Build the full benchmark categories by combining patterns with lengths.

    Returns:
        Dictionary of all benchmark categories with full configuration.
    """
    categories = {}

    for pattern_name, pattern_config in PATTERN_CONFIGS.items():
        for length_name, length_config in LENGTH_CONFIGS.items():
            category_name = get_benchmark_category_name(pattern_name, length_name)
            categories[category_name] = {
                "description": f"{pattern_config['description']} ({length_config['description']})",
                "pattern": pattern_name,
                "length_config": length_name,
                "series_length": length_config["length"],
                "params": pattern_config["params"].copy(),
            }

    return categories


# Build the full category set
BENCHMARK_CATEGORIES = build_benchmark_categories()


def generate_category_samples(
    category_name: str,
    num_samples: int,
    add_variation: bool = True,
) -> tuple:
    """
    Generate samples for a specific category.

    Args:
        category_name: Name of the category from BENCHMARK_CATEGORIES
        num_samples: Number of samples to generate
        add_variation: If True, add small random variations to parameters

    Returns:
        Tuple of (list of numpy arrays, series_length)
    """
    if category_name not in BENCHMARK_CATEGORIES:
        raise ValueError(f"Unknown category: {category_name}")

    config = BENCHMARK_CATEGORIES[category_name]
    base_params = config["params"].copy()
    series_length = config["series_length"]
    pattern_name = config["pattern"]
    samples = []

    for _ in range(num_samples):
        params = base_params.copy()

        # Add small variations to make samples diverse but still in category
        if add_variation and pattern_name != "constant":
            params["trend_slope"] += np.random.uniform(-0.05, 0.05)
            params["cycle_amplitude"] += np.random.uniform(-0.1, 0.1)
            params["noise_std"] = params["noise_std"] + np.random.uniform(-0.05, 0.05)
            params["frequency"] = params["frequency"] + np.random.uniform(-0.2, 0.2)
            params["bias"] += np.random.uniform(-1.0, 1.0)

        series = generate_time_series(length=series_length, **params)
        samples.append(series)

    return samples, series_length


def generate_benchmark_dataset(
    output_dir: str,
    samples_per_category: int = 100,
    num_classes: int = 100,
    patterns: list = None,
    lengths: list = None,
):
    """
    Generate complete benchmark dataset with pattern × length combinations.

    Args:
        output_dir: Directory to save the benchmark data
        samples_per_category: Number of samples per category
        num_classes: Number of quantization classes for preprocessing
        patterns: List of pattern names to generate (None = all)
        lengths: List of length config names to generate (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # preprocessor = TSPreprocessor(num_classes=num_classes)

    # Determine which patterns and lengths to generate
    if patterns is None:
        patterns = list(PATTERN_CONFIGS.keys())
    if lengths is None:
        lengths = list(LENGTH_CONFIGS.keys())

    # Filter categories based on selected patterns and lengths
    selected_categories = []
    for pattern, length in product(patterns, lengths):
        category_name = get_benchmark_category_name(pattern, length)
        if category_name in BENCHMARK_CATEGORIES:
            selected_categories.append(category_name)

    print(f"Generating {len(selected_categories)} categories "
          f"({len(patterns)} patterns × {len(lengths)} lengths)")
    print("=" * 60)

    for category in selected_categories:
        config = BENCHMARK_CATEGORIES[category]
        print(f"\n{category}:")
        print(f"  Pattern: {config['pattern']}, Length: {config['series_length']}")

        samples, series_length = generate_category_samples(
            category_name=category,
            num_samples=samples_per_category,
        )

        # preprocessed_tensors = []
        # metadata_list = []

        # for series in samples:
        #     tensor, meta = preprocessor.preprocess_series(series)
        #     preprocessed_tensors.append(tensor)
        #     metadata_list.append(meta)

        # Save category data
        category_data = {
            "tensors": torch.tensor(samples),
            "category": category,
            "pattern": config["pattern"],
            "length_config": config["length_config"],
            "description": config["description"],
            "params": config["params"],
            "series_length": series_length,
        }

        file_path = os.path.join(output_dir, f"benchmark_{category}.pt")
        torch.save(category_data, file_path)
        print(f"  Saved {samples_per_category} samples to {file_path}")

    # Save category metadata summary
    summary = {
        "categories": {
            cat: BENCHMARK_CATEGORIES[cat]
            for cat in selected_categories
        },
        "pattern_configs": PATTERN_CONFIGS,
        "length_configs": LENGTH_CONFIGS,
        "samples_per_category": samples_per_category,
        "num_classes": num_classes,
        "selected_patterns": patterns,
        "selected_lengths": lengths,
    }
    summary_path = os.path.join(output_dir, "benchmark_summary.pt")
    torch.save(summary, summary_path)
    print(f"\n{'=' * 60}")
    print(f"Saved summary to {summary_path}")
    print(f"Total categories generated: {len(selected_categories)}")


def list_categories():
    """Print all available benchmark categories organized by pattern and length."""
    print("\n" + "=" * 80)
    print("AVAILABLE PATTERNS:")
    print("=" * 80)
    for name, config in PATTERN_CONFIGS.items():
        print(f"\n  {name}:")
        print(f"    Description: {config['description']}")
        print(f"    Parameters:")
        for param, value in config['params'].items():
            print(f"      {param}: {value}")

    print("\n" + "=" * 80)
    print("AVAILABLE LENGTHS:")
    print("=" * 80)
    for name, config in LENGTH_CONFIGS.items():
        print(f"  {name}: {config['length']} timesteps ({config['description']})")

    print("\n" + "=" * 80)
    print("COMBINED CATEGORIES (pattern × length):")
    print("=" * 80)

    # Group by pattern
    for pattern in PATTERN_CONFIGS.keys():
        length_strs = []
        for length in LENGTH_CONFIGS.keys():
            category = get_benchmark_category_name(pattern, length)
            series_len = BENCHMARK_CATEGORIES[category]["series_length"]
            length_strs.append(f"{length}({series_len})")
        print(f"  {pattern}: {', '.join(length_strs)}")

    print(f"\nTotal categories: {len(BENCHMARK_CATEGORIES)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate benchmark data for time series model evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_data",
        help="Output directory for benchmark data",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per category",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="Number of quantization classes",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=None,
        help="Specific patterns to generate (default: all)",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        nargs="+",
        default=None,
        help="Specific length configs to generate: short, medium, long, very_long (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available patterns and lengths, then exit",
    )

    args = parser.parse_args()

    if args.list:
        list_categories()
    else:
        generate_benchmark_dataset(
            output_dir=args.output_dir,
            samples_per_category=args.samples,
            num_classes=args.num_classes,
            patterns=args.patterns,
            lengths=args.lengths,
        )