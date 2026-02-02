# model.py

import math
import torch
import torch.nn as nn
from utils.core import log_to_csv
import os

class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding for sequence tokens.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # log of 10000.0 for typical Transformer style
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        # register_buffer ensures it's saved but not treated as a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # Add positional encoding up to seq_len
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, quantized_classes=103,
                 padding_idx=102, verbose_acts=False):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(quantized_classes, model_dim, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model=model_dim)
        self.verbose_acts = verbose_acts
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(model_dim, quantized_classes)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, padding_mask: torch.Tensor = None, act_dir=None):
        """
        x: [batch_size, seq_len]
        attn_mask: [seq_len, seq_len], True where we want to block attention
        padding_mask: [batch_size, seq_len], True where PAD_TOKEN is present
        """
        # Embed tokens
        x = self.embedding(x)  # [batch_size, seq_len, model_dim]
        act_stats = {}
        if self.verbose_acts and act_dir:
            with torch.no_grad():
                act_stats['post_embed_min'] = x.min().item()
                act_stats['post_embed_max'] = x.max().item()
        if self.verbose_acts and torch.isnan(x).any():
            print('NAN detected in embedding')
        
        
        # Add positional encoding
        x = self.pos_encoding(x)
        if self.verbose_acts and act_dir:
            with torch.no_grad():
                act_stats['post_pos_min'] = x.min().item()
                act_stats['post_pos_max'] = x.max().item()
        if self.verbose_acts and torch.isnan(x).any():
            print('NAN detected in pos encoding')
        
        
        #print(f'Pre Transformer Shapes: x={x.shape}, attn_mask={attn_mask.shape}, padding_mask={padding_mask.shape}')
        # Pass through each Transformer Decoder layer
        i = 1
        for layer in self.layers:
            x = layer(
                src=x, 
                src_mask=attn_mask, 
                src_key_padding_mask=padding_mask,
                is_causal=True
            )
            if self.verbose_acts and act_dir:
                with torch.no_grad():
                    act_stats[f'post_layer{i}_min'] = x.min().item()
                    act_stats[f'post_layer{i}_max'] = x.max().item()
            if self.verbose_acts and torch.isnan(x).any():
                print('NAN detected at layer:',i)
            i+=1
        # Map final hidden states to output logits
        x = self.fc_out(x)
        if self.verbose_acts and act_dir:
            with torch.no_grad():
                act_stats[f'post_fcout_min'] = x.min().item()
                act_stats[f'post_fcout_max'] = x.max().item()
            log_to_csv(os.path.join(act_dir,'activations.csv'),act_stats)
        return x


if __name__ == '__main__':
    model = DecoderOnlyTransformer(2, 64, 8, 256)
    t1 = torch.rand((16, 48)) # batch, context_length
    output = model(t1)
    print('Input Shape:',t1.shape)
    print('Output Shape:', output.shape)
    print('Output Logit:', output[0][-1])