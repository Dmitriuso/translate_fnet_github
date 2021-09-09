import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.fnet_layer import FNetEncoderLayer, FNetLayer
from layers.pff import PositionwiseFeedforwardLayer

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 expansion_factor,
                 pf_dim,
                 dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_fnet = FNetEncoderLayer(d_model, expansion_factor, dropout)
        self.encoder_fnet = FNetEncoderLayer(d_model, expansion_factor, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(d_model,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len, 1]
        #trg_mask = [batch size, trg len, 1]

        #self attention
        _trg = self.self_fnet(trg, trg_mask)

        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #encoder attention
        _src = self.encoder_fnet(enc_src, src_mask)

        #_src = [batch size, src len, hid dim]

        #dropout, residual connection and layer norm
        max_len = max([trg.shape[1], _src.shape[1]])
        # cannot otherwise add trg and dropout(_src),
        # because src len and trg len dimensions are different
        trg = \
            self.enc_attn_layer_norm(
                F.pad(trg, (0, 0, 0, max_len - trg.shape[1])) +
                self.dropout(F.pad(_src, (0, 0, 0, max_len - _src.shape[1])))
            )

        #trg = [batch size, max len, hid dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, max len, hid dim]

        return trg