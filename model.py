import torch.nn as nn
import torch

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, quantized_classes=102):
        super(DecoderOnlyTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=hidden_dim)
            for _ in range(num_layers)
        ])
        self.embedding = nn.Embedding(quantized_classes, model_dim)
        self.fc_out = nn.Linear(model_dim, quantized_classes)

    def forward(self, x, attn_mask=None, padding_mask=None):
        # x: batch, sequence
        x = self.embedding(x) # batch, sequence, model_dim
        for layer in self.layers:
            x = layer(x, x, tgt_mask=attn_mask, tgt_key_padding_mask=padding_mask)
        return self.fc_out(x)
    


if __name__ == '__main__':
    model = DecoderOnlyTransformer(2, 64, 8, 128)
    t1 = torch.rand((16, 48)) # batch, context_length
    output = model(t1)
    print('Input Shape:',t1.shape)
    print('Output Shape:', output.shape)
    print('Output Logit:', output[0][-1])