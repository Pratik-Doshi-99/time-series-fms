# Foundation Models for Time Series Forecasting

This repository provides a **decoder-only Transformer** for modeling **quantized time-series data**. It includes:

- **Data preprocessing** (normalization, quantization, and special tokens)  
- **Two training paradigms**: incremental vs. multi-step  
- **A Transformer-based model** (decoder-only) with **positional encoding**  
- **Scripts** to **train** the model and **test** its components

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)  
   - [Normalization](#normalization)  
   - [Quantization](#quantization)  
   - [Special Tokens](#special-tokens)  
2. [Training Approaches](#training-approaches)  
   - [Incremental (Prefix) Approach](#incremental-prefix-approach)  
   - [Multi-Step Approach](#multi-step-approach)  
3. [Model Architecture](#model-architecture)  
   - [Decoder-Only Transformer](#decoder-only-transformer)  
   - [Positional Encoding](#positional-encoding)  
4. [Running the Training Code](#running-the-training-code)  
5. [Running Unit Tests](#running-unit-tests)



## Data Preprocessing

All data preprocessing steps are handled by the **`TSPreprocessor`** in `data.py`. The major steps are **normalization**, **quantization**, and optionally inserting **special tokens**.

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

1. X = [x1], Y = [x2]  
2. X = [x1, x2], Y = [x3]  
3. ...  
4. X = [x1, x2, ..., x(T-1)], Y = [xT]

These pairs get batched in the **`AutoregressiveLoader`**, which also builds an autoregressive (causal) mask for each prefix so the model only attends to past tokens. During training, the loss is computed only on the last token in each prefix.

### Multi-Step Approach

In the multi-step approach, each entire sequence is used in a single pass. For a sequence `[x1, x2, ..., xT]`, we define:

- X = [x1, x2, ..., x(T-1)]  
- Y = [x2, x3, ..., xT]  

A causal mask ensures that future tokens cannot be seen at each time step. We then compute the cross-entropy loss over **all** time steps in the sequence. This is sometimes called “teacher forcing,” since the model learns to predict each next token in one forward pass.



## Model Architecture

### Decoder-Only Transformer

We use a **decoder-only** architecture, similar to GPT-style language models. In `model.py`:

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

2. **Train the Model**  
   - The main entry point is **`main.py`**. By default, it does the following:
     1. Synthesizes time-series data with `generate_time_series`.  
     2. Preprocesses (normalizes + quantizes) them and saves to `preprocessed_data.pt`.  
     3. Creates a `MultiTimeSeriesDataset` and either:
        - Uses the **incremental** `AutoregressiveLoader`, or
        - Uses the **multi-step** loader (e.g., `MultiStepLoader`).  
     4. Builds the decoder-only Transformer.  
     5. Trains for a specified number of epochs, printing the loss each epoch.

   - You can choose **incremental** or **multi-step** by editing the code or a flag inside `main.py`.  
   - Then run:

     ```
     python main.py
     ```

3. **Hyperparameters**  
   - You can adjust hyperparameters such as `model_dim`, `num_layers`, `num_heads`, `epochs` inside `main.py` or in the train functions.



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



**Enjoy experimenting** with this decoder-only time-series Transformer! If you have questions or want to contribute, please open an issue or pull request.