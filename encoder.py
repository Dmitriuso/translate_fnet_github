import torch
import torch.nn as nn
from layers.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model,
                 n_layers,
                 expansion_factor,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = nn.Embedding(input_dim, d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model,
                                                  expansion_factor,
                                                  pf_dim,
                                                  dropout)

                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self, src, src_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, src len, 1]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        #src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        #src = [batch size, src len, hid dim]

        return src
