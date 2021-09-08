import torch
import torch.nn as nn
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
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg = self.self_fnet(trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
        # print(f'decoder layer trg size: {trg.shape}')
            
        #encoder attention
        _src = self.encoder_fnet(enc_src, src_mask)

        #_src = [batch size, src len, hid dim]
        # print(f'decoder layer _src size: {_src.shape}')

        #dropout, residual connection and layer norm
        max_len = max([trg.shape[1], _src.shape[1]])
        trg = self.enc_attn_layer_norm(
            nn.functional.pad(trg, (0, 0, 0, max_len - trg.shape[1])) + nn.functional.pad(_src, (0, 0, 0, max_len - _src.shape[1])))
                    
        #trg = [batch size, trg len, hid dim]
        # print(f'decoder layer combined trg size: {trg.shape}')

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # print(f'decoder layer trg after feed-forward size: {_trg.shape}')

        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # print(f'decoder layer final trg size: {trg.shape}')

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg
