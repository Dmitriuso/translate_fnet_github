import torch
import torch.nn as nn
import torch.optim as optim
# from layers.mha_layer import MultiHeadAttentionLayer
from layers.fnet_layer import FNetLayer, FeedForward
from layers.pff import PositionwiseFeedforwardLayer

class EncoderLayer(nn.Module):
    def __init__(self, 
                 d_model,
                 expansion_factor, 
                 pf_dim,  
                 dropout):
        super().__init__()
        
        self.self_fnet_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_fnet = FNetLayer(d_model, expansion_factor, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(d_model, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src = self.self_fnet(src)
        
        #dropout, residual connection and layer norm
        src = self.self_fnet_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src