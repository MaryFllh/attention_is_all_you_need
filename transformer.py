import torch
import torch.nn as nn

from layers.encoder import Encoder
from layers.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        trg_vocab_size,
        d_model=512,
        max_seq_len=100,
        heads_num=8,
        forward_expansion=4,
        dropout=0.1,
        layers_num=6,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            max_seq_len,
            heads_num,
            forward_expansion,
            dropout,
            layers_num,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            d_model,
            max_seq_len,
            heads_num,
            forward_expansion,
            dropout,
            layers_num,
        )

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.size()
        return torch.tril(torch.ones(trg_len, trg_len)).expand(
            batch_size, 1, trg_len, trg_len
        )

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(encoder_out, trg, trg_mask)
        return decoder_out
