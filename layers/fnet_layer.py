import torch
import numpy as np
from torch.fft import fftn
from torch import nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        out = self.dropout2(self.fc2(x))
        return out


def fourier_transform(x):
    return fftn(x, dim=(-1, -2)).real


class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.ff = FeedForward(d_model, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        # x = [batch size, src_len, hid_dim]

        residual = x

        # print(f'tensor before mask: {x.shape}')
        x = fourier_transform(x)
        x = self.norm1(x + residual)

        if mask is not None:
            x = x.masked_fill(mask == 0, -1e10)
            # x = torch.tril(x)
            # x = torch.triu(x * float(-1e10), diagonal=1)

        # print(f'tensor after mask: {x.shape}')

        residual = x
        x = self.ff(x)
        out = self.norm2(x + residual)
        return out


class FNetLayer(nn.TransformerEncoder):
    def __init__(self, d_model=256, expansion_factor=2, dropout=0.1, num_layers=1):
        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
