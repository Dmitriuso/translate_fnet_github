import torch
from torch.fft import fft, fftn
from torch import nn as nn


def fourier_transform(x):
    return fft(torch.fft.fft(x, dim=-1), dim=-2).real


class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        # x = [batch size, src_len, hid_dim]

        residual = x

        x = fourier_transform(x)
        x = self.norm1(x + residual)

        if mask is not None:
            x = x.masked_fill(mask == 0, float(-1))
            # x = torch.tril(x)

        # x = [batch size, src_len, hid_dim]

        return x


class FNetLayer(nn.TransformerEncoder):
    def __init__(self, d_model=256, expansion_factor=2, dropout=0.1, num_layers=1):
        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
