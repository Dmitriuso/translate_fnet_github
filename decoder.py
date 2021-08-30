import torch
import torch.nn as nn
from layers.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 d_model, 
                 n_layers, 
                 expansion_factor, 
                 pf_dim, 
                 dropout, 
                 device,
                 ):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = nn.Embedding(output_dim, d_model)
        
        self.layers = nn.ModuleList([DecoderLayer(d_model, 
                                                  expansion_factor, 
                                                  pf_dim, 
                                                  dropout)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(d_model, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
        
    def forward(self, trg, enc_src): # , trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg = layer(trg, enc_src) # , trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output