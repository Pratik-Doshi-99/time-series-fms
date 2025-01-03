# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from typing import Union
from data import MultiStepLoader, AutoregressiveLoader


def train_model(model, train_loader: Union[MultiStepLoader, AutoregressiveLoader], epochs, device, train_mode="incremental"):
    """
    train_mode can be 'incremental' or 'multi-step'.
    'incremental':  Each batch is from AutoregressiveLoader, 
                    we only look at model output at the last time step.
    'multi-step':   Each batch is from MultiStepLoader, 
                    we compute cross-entropy over entire sequence.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.preprocessor.PAD_TOKEN)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y, attn_mask, padding_mask in train_loader:
            # x,y shapes can differ depending on the loader
            x = x.to(device)
            y = y.to(device)
            attn_mask = attn_mask.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()

            output = model(x, attn_mask=attn_mask, padding_mask=padding_mask)
            # output shape: [batch_size, seq_len, quantized_classes]
            #print('Output Shape:', output.shape)
            if train_mode == "incremental":
                # Here, y is [batch_size, seq_len], but typically seq_len=1
                # or we only want the "last token" from the output
                # Let's assume y is shape [batch_size, 1], so we do:
                # output[:, -1, :] => shape [batch_size, quantized_classes]
                # target => shape [batch_size], so we squeeze if needed
                # But your existing code might pad it. We'll adapt:

                # If y has shape [batch_size, seq_len], we use the last step
                # (the second dimension might be > 1 if padded)
                last_logits = output[:, -1, :]   # [batch_size, num_classes]
                last_target = y[:, -1]          # [batch_size]
                
                loss = criterion(last_logits, last_target)

            elif train_mode == "multi-step":
                # We want to compute cross-entropy across ALL positions
                # output: [batch_size, seq_len, classes]
                # y:      [batch_size, seq_len]
                # CrossEntropyLoss expects [batch_size, classes, seq_len], so we transpose:
                logits = output.transpose(1, 2)   # => [batch_size, classes, seq_len]
                # Also ensure y is shape [batch_size, seq_len]
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            #print('Loss:',loss.item())
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}")

def train_model_multi_gpu(model, train_loader, epochs, train_mode="incremental"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    train_model(model, train_loader, epochs, device, train_mode=train_mode)
