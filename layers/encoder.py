import copy
import torch.nn as nn


from sublayers.multihead_attention import MultiheadAttention
from sublayers.point_wise_feed_forward import PointWiseFeedForward
from embeddings.embedder import Embedder
from embeddings.postional_encoder import PositionalEncoder


class EncoderLayer(nn.Module):
    """
    The implementation of a single Encoder layer.
    A stack of these will be used to build
    the encoder portion of the Transformer
    """

    def __init__(self, d_model, heads_num, forward_expansion, dropout):
        super().__init__()
        self.multihead_attention = MultiheadAttention(d_model, heads_num)
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.point_wise_feed_forward = PointWiseFeedForward(d_model, forward_expansion)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        multihead_attention = self.multihead_attention(q=x, k=x, v=x, mask=mask)
        attention_layer_norm = self.attention_layer_norm(
            x + self.dropout(multihead_attention)
        )
        pwff = self.point_wise_feed_forward(attention_layer_norm)
        out = self.ff_layer_norm(pwff + self.dropout(pwff))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        d_model,
        max_seq_len,
        heads_num,
        forward_expansion,
        dropout,
        layers_num,
    ):
        super().__init__()
        self.embedding = Embedder(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoder(max_seq_len, d_model)
        self.encoder_layers = nn.ModuleList(
            [
                copy.deepcopy(
                    EncoderLayer(d_model, heads_num, forward_expansion, dropout)
                )
                for _ in range(layers_num)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.dropout(self.embedding(x) + self.positional_encoding(x))
        for encoder_layer in self.encoder_layers:
            out = encoder_layer(out, mask)
        return out
