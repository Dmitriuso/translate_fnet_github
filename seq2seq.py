import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(2)

        #src_mask = [batch size, src len, 1]

        return src_mask

    def make_trg_mask(self, trg):

        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(2)

        #trg_pad_mask = [batch size, trg len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, 1), device=self.device)).bool()

        #trg_sub_mask = [trg len, 1]

        trg_mask = trg_pad_mask & trg_sub_mask

        #trg_mask = [batch size, trg len, 1]

        return trg_mask

    def forward(self, src, trg):

        #src = [batch size, src len]

        #trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        #src_mask = [batch size, src len, 1]
        #trg_mask = [batch size, trg len, 1]

        enc_src = self.encoder(src, src_mask)

        #enc_src = [batch size, src len, hid dim]

        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        #output = [batch size, trg len, output dim]

        return output