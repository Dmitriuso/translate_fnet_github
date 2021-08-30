import torch
import torch.nn as nn
from layers.fnet_layer import FNetLayer
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
        self.self_fnet = FNetLayer(d_model, expansion_factor, dropout)
        self.encoder_fnet = FNetLayer(d_model, expansion_factor, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(d_model, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg = self.self_fnet(trg)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg = self.encoder_fnet(trg)
	
        # print('-' * 50)
        # print(f'enc_src : {enc_src.shape}')
        # print(f'trg : {trg.shape}')
        # print(f'_trg : {_trg.shape}')
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        max_len = max([trg.shape[1], enc_src.shape[1]])
        # min_len = torch.min([trg.shape[1], enc_src.shape[1]])
        # pad = (0, 0, 0, max_len - min_len)
        
        trg = self.ff_layer_norm(nn.functional.pad(trg + self.dropout(_trg), (0, 0, 0, max_len - trg.shape[1])) + nn.functional.pad(enc_src, (0, 0, 0, max_len - enc_src.shape[1])) )
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg
