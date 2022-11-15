import copy
import torch
import torch.nn as nn


from sublayers.multihead_attention import MultiheadAttention
from sublayers.point_wise_feed_forward import PointWiseFeedForward
from embeddings.embedder import Embedder
from embeddings.postional_encoder import PositionalEncoder


class DecoderLayer(nn.Module):
    """
    Implements a decoder layer. A stack of these layers
    will be used to build the decoder portion of the transformer
    """

    def __init__(self, d_model, heads_num, forward_expansion, dropout):
        super().__init__()
        self.multihead_attention = MultiheadAttention(d_model, heads_num)
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attention = MultiheadAttention(d_model, heads_num)
        self.enc_dec_att_layer_norm = nn.LayerNorm(d_model)
        self.point_wise_feed_forward = PointWiseFeedForward(d_model, forward_expansion)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out, x, mask):
        # Compute Multi_head attention with masking
        self_attention = self.multihead_attention(q=x, k=x, v=x, mask=mask)

        # Add & Norm
        self_attention_norm = self.attention_layer_norm(
            x + self.dropout(self_attention)
        )

        # Encoder-Decoder attention
        enc_dec_attention = self.encoder_decoder_attention(q=x, k=enc_out, v=enc_out)

        # Add & Norm
        enc_dec_att_norm = self.attention_layer_norm(
            self_attention_norm + self.dropout(enc_dec_attention)
        )

        # Feed forward
        pwff = self.point_wise_feed_forward(enc_dec_att_norm)

        # Add & Norm
        out = self.ff_layer_norm(pwff + self.dropout(pwff))
        return out


class Decoder(nn.Module):
    """
    Consists of a stack of DecoderLayer()s
    """

    def __init__(
        self,
        trg_vocab_size,
        d_model,
        max_seq_len,
        heads_num,
        forward_expansion,
        dropout,
        layers_num,
    ):
        super().__init__()
        self.embedding = Embedder(trg_vocab_size, d_model)
        self.positional_encoding = PositionalEncoder(max_seq_len, d_model)
        self.decoder = nn.ModuleList(
            [
                copy.deepcopy(
                    DecoderLayer(d_model, heads_num, forward_expansion, dropout)
                )
                for _ in range(layers_num)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, trg_vocab_size)

    def forward(self, enc_out, x, mask):
        out = self.dropout(self.embedding(x) + self.positional_encoding(x))
        for decoder_layer in self.decoder:
            out = decoder_layer(enc_out, out, mask)
        dso = self.linear(out)
        out = torch.softmax(dso, dim=-1)
        return out
