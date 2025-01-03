# Foundation Models for Time Series Forecasting


This repository provides a **decoder-only Transformer** for modeling **quantized time-series data**. It includes:

- **Data preprocessing** (normalization, quantization, and special tokens)
- **Two training paradigms**: *incremental* vs. *multi-step*
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

---

## Data Preprocessing

All data preprocessing steps are handled by the **`TSPreprocessor`** in `data.py`. The major steps are **normalization**, **quantization**, and **special token** insertion (optionally).

### Normalization

Given a raw time-series \(\mathbf{x}\) of length \(T\), we first normalize it into the range \([0, 1]\) by computing:

\[
\text{normalized}_t \;=\; \frac{x_t - \min(\mathbf{x})}{\max(\mathbf{x}) \;-\; \min(\mathbf{x})}
\]

If \(\max(\mathbf{x}) = \min(\mathbf{x})\) (i.e., the series is constant), we simply assign all values to \(0\).

### Quantization

After normalization, each time step is mapped into **integer bins** in the range \([0,\dots,\text{num\_classes}-1]\). Suppose:

- \(\text{range\_min} = 0\)
- \(\text{range\_max} = \text{num\_classes}-1\)

Then each normalized value \(\text{normalized}_t \in [0,1]\) is **scaled** and **rounded**:

\[
\text{quantized}_t \;=\; \text{round}\!\bigl(\text{normalized}_t \cdot (\text{range\_max} - \text{range\_min})\bigr)
\]
\[
\text{quantized}_t \;=\; \max(\text{range\_min},\; \min(\text{quantized}_t,\; \text{range\_max}))
\]

Hence all raw values end up as integers in \([0..(\text{num\_classes}-1)]\).

### Special Tokens

We insert the following **optional** tokens:

- **BOS_TOKEN**: Denotes the beginning of sequence  
- **EOS_TOKEN**: Denotes the end of sequence  
- **PAD_TOKEN**: Used for **padding** sequences to match lengths in a batch  

These are **instance-level** in `TSPreprocessor` and set to:
```
BOS_TOKEN = num_classes
EOS_TOKEN = num_classes + 1
PAD_TOKEN = num_classes + 2
```
This ensures they do not overlap with the valid data range \([0..(\text{num\_classes}-1)]\).

---

## Training Approaches

There are **two** ways to train the model on the time-series:

1. **Incremental (Prefix) Approach**  
2. **Multi-Step Approach**

### Incremental (Prefix) Approach

In the **incremental** approach (often called prefix-based), each prefix of a sequence is treated as a separate sample. For a sequence \(\{x_1,\dots,x_T\}\), we generate:

1. \(\mathbf{X} = [x_1]\), \(\mathbf{Y} = [x_2]\)  
2. \(\mathbf{X} = [x_1,\, x_2]\), \(\mathbf{Y} = [x_3]\)  
3. \(\dots\)  
4. \(\mathbf{X} = [x_1,\dots,x_{T-1}]\), \(\mathbf{Y} = [x_T]\)

These many (X, Y) pairs get batched in the **`AutoregressiveLoader`**, which builds an **autoregressive mask** for each prefix so the model only attends to past tokens. During training, the loss is typically computed **only** on the **last token** in each prefix (i.e., *one-step-ahead* prediction).

### Multi-Step Approach

In the **multi-step** approach, each entire sequence is loaded in a single pass. For a given sequence \(\{x_1,\dots,x_T\}\), we create:

- \(\mathbf{X} = [x_1,\, x_2,\, \dots,\, x_{T-1}]\)  
- \(\mathbf{Y} = [x_2,\, x_3,\, \dots,\, x_T]\)

Then a **causal mask** ensures the model does **not** attend to future tokens. We compute the cross-entropy loss over **all time steps** in the sequence. This is a classic teacher-forcing approach: the model learns to predict **each** next token in a single forward pass.

---

## Model Architecture

### Decoder-Only Transformer

We use a **decoder-only** architecture, akin to GPT-style language models, defined in **`model.py`**:

1. **Embedding**: An `nn.Embedding(quantized_classes, model_dim)` that maps each token (integer ID) to a `model_dim`-sized embedding vector.  
2. **Positional Encoding**: We add a standard sine/cosine positional encoding to each embedding.  
3. **Stack of Decoder Layers** (`nn.TransformerDecoderLayer`):  
   - Each layer has self-attention + feedforward + layer norms (with residual connections *inside*).  
   - We pass `memory=None` for a pure decoder.  
4. **Output Projection**: A `Linear(model_dim, quantized_classes)` that produces logits over all possible classes.

### Positional Encoding

We use the **standard sine/cosine** positional encoding:

\[
\text{PE}(pos,\,2i) = \sin\!\bigl(\frac{pos}{10000^{\,\frac{2i}{d}}}\bigr),
\quad
\text{PE}(pos,\,2i+1) = \cos\!\bigl(\frac{pos}{10000^{\,\frac{2i}{d}}}\bigr)
\]
- \(pos\) is the token position \(\{0,1,\dots\}\)
- \(i\) indexes the embedding dimension
- \(d\) is the embedding size
- We add this vector to the embedding for each position

---

## Running the Training Code

1. **Install Dependencies**  
   - Python 3.8+  
   - PyTorch (1.9+ recommended)  
   - NumPy, Matplotlib (for some optional plots)

2. **Generate / Preprocess Data** + **Run Training**  
   - The main training entry point is **`main.py`**. By default, it:
     1. Synthesizes time-series data (`generate_time_series`)
     2. Preprocesses (normalizes/quantizes) them
     3. Saves them to `preprocessed_data.pt`
     4. Creates a `MultiTimeSeriesDataset`
     5. Either uses the **incremental** `AutoregressiveLoader` or **multi-step** loader to form batches
     6. Builds the **decoder-only transformer** model
     7. Trains for a set number of epochs

   - **Select the training mode** (e.g. `"incremental"` vs. `"multi-step"`) within `main.py`.  
   - Then run:
     ```bash
     python main.py
     ```
   - It will print out the training loss each epoch.

3. **Adjust Hyperparameters**  
   - You can modify `epochs`, `model_dim`, `num_heads`, etc. in `main.py` or `train.py`.  
   - For multi-GPU training, `train_model_multi_gpu` uses `nn.DataParallel`.

---

## Running Unit Tests

We have **unit tests** in the `tests/` folder (e.g. `test_data.py`, `test_model.py`, `test_train.py`) that verify:

- **Data** generation, preprocessing, and loaders  
- **Model** forward pass shape correctness and positional encoding  
- **Training** loop integration

To run **all** tests at once:

```bash
cd tests
python -m unittest discover
```
or individually:

```bash
python -m unittest test_data.py
python -m unittest test_model.py
python -m unittest test_train.py
```

---

**Enjoy experimenting** with this decoder-only Transformer for time series! If you have any issues or questions, please open a discussion or pull request.