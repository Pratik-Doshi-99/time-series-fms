# Foundation Models for Time Series Forecasting

This repository provides a **decoder-only Transformer** for modeling **quantized time-series data**. It includes:

- **Data preprocessing** (normalization, quantization, and special tokens)
- **Two training paradigms**: incremental vs. multi-step
- **A Transformer-based model** (decoder-only) with **positional encoding**
- **Single-GPU and distributed training** support (with DeepSpeed)
- **Scripts** to **train** the model and **test** its components

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Data Preprocessing](#data-preprocessing)
   - [Normalization](#normalization)
   - [Quantization](#quantization)
   - [Special Tokens](#special-tokens)
3. [Training Approaches](#training-approaches)
   - [Incremental (Prefix) Approach](#incremental-prefix-approach)
   - [Multi-Step Approach](#multi-step-approach)
4. [Model Architecture](#model-architecture)
   - [Decoder-Only Transformer](#decoder-only-transformer)
   - [Positional Encoding](#positional-encoding)
5. [Running the Training Code](#running-the-training-code)
6. [Running Benchmarks](#running-benchmarks)
7. [Running Unit Tests](#running-unit-tests)


## Repository Structure

The repository is organized into the following subdirectories:

```
time-series-fms/
├── data/               # Data generation and loading
│   └── dataset.py      # TSPreprocessor, Dataset, DataLoaders
├── models/             # Model definitions
│   └── model.py        # DecoderOnlyTransformer, PositionalEncoding
├── training/           # Training loops and entry points
│   ├── main.py         # Single-GPU training entry point
│   ├── main_dist.py    # Distributed training entry point
│   ├── train.py        # Single-GPU training loop
│   └── dist_train.py   # Distributed training with DeepSpeed
├── scripts/            # Shell scripts for running programs
│   ├── generate_data.sh
│   ├── train_size1.sh
│   ├── train_size2.sh
│   ├── train_size3.sh
│   └── dist_train_size3.sh
├── exploration/        # Jupyter notebooks for analysis
│   ├── gradient_analysis.ipynb
│   ├── test_saved_model.ipynb
│   └── tsfm_demo.ipynb
├── tests/              # Unit tests
└── utils.py            # Utility functions
```



## Data Preprocessing

All data preprocessing steps are handled by the **`TSPreprocessor`** in `data/dataset.py`. The major steps are **normalization**, **quantization**, and optionally inserting **special tokens**.

### Normalization

Given a raw time series `x` of length `T`, we first normalize it into the range [0, 1] by computing:

- normalized_value = (original_value - min_x) / (max_x - min_x)

where `min_x` and `max_x` are the minimum and maximum of the entire series.  
If `(max_x == min_x)`, meaning the series is constant, we assign everything to zero.

### Quantization

After normalization, each time step is mapped into integer bins in the range `[0, ..., num_classes - 1]`. For example, if `num_classes = 100`, the valid data bins are `[0..99]`.  
We do the following:

1. Multiply the normalized value by `(range_max - range_min)`.  
2. Round to the nearest integer.  
3. Clip to ensure the result is within `[range_min..range_max]`.

Hence, all raw values end up as integers in `[0..(num_classes - 1)]`.

### Special Tokens

We insert the following **optional** tokens:

- **BOS_TOKEN** (Beginning of Sequence)  
- **EOS_TOKEN** (End of Sequence)  
- **PAD_TOKEN** (Padding Token)  

These tokens are **instance-level** in `TSPreprocessor` and automatically set to `num_classes`, `num_classes + 1`, and `num_classes + 2`, respectively. This ensures they do not overlap with valid data bins in `[0..(num_classes-1)]`.


## Training Approaches

There are two ways to train the model on your time-series:

1. **Incremental (Prefix) Approach**  
2. **Multi-Step Approach**

### Incremental (Prefix) Approach

In this approach, each prefix of a sequence is treated as a separate sample. For a sequence `[x1, x2, ..., xT]`, we generate:

```
X = [x1], Y = [x2]  
X = [x1, x2], Y = [x3]  
...  
X = [x1, x2, ..., x(T-1)], Y = [xT]
```

These pairs get batched in the **`AutoregressiveLoader`**, which also builds an autoregressive (causal) mask for each prefix so the model only attends to past tokens. During training, the loss is computed only on the last token in each prefix.

### Multi-Step Approach

In the multi-step approach, each entire sequence is used in a single pass. For a sequence `[x1, x2, ..., xT]`, we define:
```
X = [x1, x2, ..., x(T-1)]  
Y = [x2, x3, ..., xT]  
```

A causal mask ensures that future tokens cannot be seen at each time step. We then compute the cross-entropy loss over **all** time steps in the sequence. This is sometimes called “teacher forcing,” since the model learns to predict each next token in one forward pass.



## Model Architecture

### Decoder-Only Transformer

We use a **decoder-only** architecture, similar to GPT-style language models. In `models/model.py`:

- **Embedding**: `nn.Embedding(quantized_classes, model_dim)` maps each integer token to a vector.  
- **Positional Encoding**: We add a sine/cosine positional encoding to each embedding.  
- **Stack of Decoder Layers**: Each layer is an `nn.TransformerDecoderLayer` with self-attention and feedforward components.  
- **Output Projection**: A linear layer maps the final hidden states to logits over `quantized_classes`.

### Positional Encoding

We follow the standard sine/cosine approach, commonly used in the original "Attention Is All You Need" paper. For each position `pos` in `[0..seq_len-1]` and embedding dimension index `i`:

- PE(pos, 2i)   = sin( pos / (10000^(2i/d)) )  
- PE(pos, 2i+1) = cos( pos / (10000^(2i/d)) )

These values are added to the embedded tokens before they enter the Transformer layers.



## Running the Training Code

1. **Install Dependencies**  
   - Python 3.8+  
   - PyTorch (1.9+ recommended)  
   - NumPy, Matplotlib (for optional plotting)

2. **Generate Training Data**
   - First, generate synthetic time-series data using the provided script:

     ```bash
     bash scripts/generate_data.sh
     ```

   - This will create preprocessed data files in the `synth-data/` directory.

3. **Train the Model**

   **Single-GPU Training:**
   - The main entry point is **`training/main.py`** which supports command-line arguments for all hyperparameters.
   - You can use the provided training scripts or run directly:

     ```bash
     # Using a training script
     bash scripts/train_size2.sh

     # Or run directly with custom arguments
     python training/main.py \
       --num_layers 4 \
       --model_dim 256 \
       --hidden_dim 1024 \
       --num_heads 16 \
       --epochs 2 \
       --max_training_length 1024 \
       --batch_size 16 \
       --data_dir synth-data \
       --train_mode multi-step
     ```

   **Distributed Training (DeepSpeed):**
   - For multi-GPU training, use `training/main_dist.py`:

     ```bash
     # Using the distributed training script
     bash scripts/dist_train_size3.sh

     # Or run directly
     deepspeed --hostfile=hostfile training/main_dist.py \
       --num_layers 12 \
       --model_dim 768 \
       --hidden_dim 2048 \
       --batch_size 16 \
       --data_dir synth-data
     ```



## Running Benchmarks

After training, evaluate model performance across different time-series patterns:

1. **Generate Benchmark Data**
   ```bash
   python benchmarks/generate_benchmark_data.py --output-dir benchmark_data --samples 100
   ```

2. **Run Benchmark**
   ```bash
   # Using the benchmark script
   bash scripts/benchmark.sh

   # Or run directly
   python benchmarks/run_benchmark.py \
     --model-path path/to/model.pt \
     --benchmark-dir benchmark_data \
     --output results.json
   ```

   Results include accuracy, MAE, RMSE, and direction accuracy across patterns like trends, seasonality, and noise.

### Performance

| Pattern | Accuracy | MAE | RMSE | Cross Entropy | Direction Acc | Median Err | P90 Err | P95 Err |
|---------|----------|-----|------|---------------|---------------|------------|---------|---------|
| Flat line (all values identical) | **100.0%** | **0.000** | **0.000** | **0.092** | **100.0%** | 0.0 | 0.0 | 0.0 |
| Clear upward linear trend | 69.8% | 0.302 | 0.549 | 0.602 | 70.1% | 0.0 | 1.0 | 1.0 |
| Upward trend with periodic component | 69.5% | 0.305 | 0.552 | 0.626 | 69.4% | 0.0 | 1.0 | 1.0 |
| Clear downward linear trend | 68.1% | 0.319 | 0.565 | 0.619 | 68.6% | 0.0 | 1.0 | 1.0 |
| Very clean signal with minimal noise | 67.1% | 0.329 | 0.574 | 0.646 | 67.5% | 0.0 | 1.0 | 1.0 |
| All components: trend + cycle + noise + drift | 50.9% | 0.532 | 0.784 | 1.127 | 55.9% | 0.0 | 1.0 | 1.0 |
| Drift-dominated movement (like random walk) | 44.9% | 0.632 | 0.894 | 1.290 | 52.7% | 1.0 | 1.0 | 2.0 |
| Trend partially obscured by noise | 31.2% | 0.971 | 1.288 | 1.676 | 45.8% | 1.0 | 2.0 | 2.0 |
| Slow periodic oscillation (1 cycle) | 22.7% | 1.552 | 2.120 | 2.097 | 37.9% | 1.0 | 3.0 | 4.0 |
| Stationary series with no trend | 20.9% | 2.885 | 5.734 | 2.316 | 44.0% | 1.0 | 7.0 | 11.0 |
| Strong signal buried in heavy noise | 12.6% | 2.534 | 3.219 | 2.576 | 42.2% | 2.0 | 5.0 | 6.0 |
| Subtle periodic pattern | 11.1% | 4.082 | 6.272 | 2.864 | 41.7% | 3.0 | 10.0 | 14.0 |
| Fast periodic oscillation (8 cycles) | 6.0% | 4.485 | 5.727 | 3.325 | 17.3% | 4.0 | 8.0 | 10.0 |
| Pure random noise (no signal) | 3.3% | 11.327 | 14.917 | 4.042 | 37.5% | 9.0 | 24.0 | 31.0 |

**Aggregate Statistics**: Mean Accuracy: 41.3% ± 29.1% | Mean MAE: 2.16 ± 2.91 | Total Predictions per Pattern: 32,512


## Running Unit Tests

We have unit tests in the `tests/` folder (for example, `test_data.py`, `test_model.py`, `test_train.py`) that validate:

- **Data**: generation, quantization, loaders  
- **Model**: forward pass shape, positional encoding correctness  
- **Training**: end-to-end integration

To run **all** tests:

```
cd tests
python -m unittest discover
```

Or run them individually:

```
python -m unittest test_data.py
python -m unittest test_model.py
python -m unittest test_train.py
```


## Roadmap

The following things are under development.

- [ ] Add validation split, and monitor validation performance when training
- [ ] Add MSE metric, and monitor during training
- [ ] Add tracking of gradient norms, dead neurons and 0 gradients during training
- [ ] Add support for gradient accumulation across mini-batches (to increase the effective batch size)
- [ ] Create an auto-restart system that can restart training from the latest optimizer, lr global step checkpoint. Also append to same WandB run
- [ ] Save all training and gpu logs in the base directory of the experiment


## Citation

If you use this code in your research, please cite:

```bibtex
@software{time_series_fms,
  title = {Foundation Models for Time Series Forecasting},
  author = {Doshi, Pratik},
  year = {2026},
  url = {https://github.com/pratikdoshi/time-series-fms},
  note = {A decoder-only Transformer for quantized time-series forecasting}
}
```