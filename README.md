# Implementation of Transformer's building blocks
This repository includes the modules that construct a transformer. It follows the descriptions of Attention is All You Need
Implementation of Transformers as described by the [Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762.pdf).
The structure of the repository is as follows:
```
.
├── embeddings
│   ├── embedder.py
│   └── postional_encoder.py
├── layers
│   ├── decoder.py
│   └── encoder.py
├── sublayers
│   ├── multihead_attention.py
│   ├── point_wise_feed_forward.py
│   └── scaled_dot_product_attention.py
└── transformer.py
```
## embeddings/
### Input Embeddings:
The input text is transformed to a vector representation using the `Embedding` module: 
<img width="204" alt="Screenshot 2022-11-17 at 7 03 55 PM" src="https://user-images.githubusercontent.com/36740868/202585966-3a0db1dd-fc7d-4b13-bf92-b583e5aec6fe.png">

```
# embeddings/embedder.py

import torch.nn as nn


class Embedder(nn.Module):
    """
    Embedding class used to embed the inputs
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        According to section 3.4 of Attention is All You Need,
        the embeddings are multiplied by square root of
        d_model
        """
        input_embeddings = self.embedding(x) * (self.d_model**0.5)
        return input_embeddings
```
### Positional Encodings
The positions are encoded using `sin` and `cos` for each even and odd dimension respectively:
<img width="364" alt="Screenshot 2022-11-17 at 7 40 16 PM" src="https://user-images.githubusercontent.com/36740868/202590062-37f2155f-415c-4fa5-94a6-a92c4c0d4a04.png">

```
# embeddings/postional_encoder.py

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    Implements the positional encodings based
    on section 3.5 in Attention is All you Need
    """

    def __init__(self, max_seq_len, d_model):
        """
        Args:
            max_len(int): maximum length of the input
            d_model: embedding size
        """
        super().__init__()
        self.positional_encodings = torch.zeros(max_seq_len, d_model)
        positions = torch.arange(max_seq_len).unsqueeze(1)
        division_term = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        self.positional_encodings[:, 0::2] = torch.sin(positions / division_term)
        self.positional_encodings[:, 1::2] = torch.cos(positions / division_term)

    def forward(self, x):
        input_len = x.size()[1]
        return self.positional_encodings[:input_len, :]

```
Using the code snippet below, the postional encodings for a sequence of length of 25 and dimesion size of 64 is illustrated:
```
import  matplotlib.pyplot as plt

from embeddings.positional_encoder import PositionalEncoder

x = torch.tensor([[1] * 25, [1] * 25])
max_seq_len = 25
d_model = 64

positional_encoding = PositionalEncoder(max_seq_len=max_seq_len, d_model=d_model)
pos_encoding = positional_encoding(x)

plt.figure(figsize=(12,8))
plt.pcolormesh(pos_encoding, cmap='twilight')
plt.xlabel('Dimensions')
plt.xlim((0, d_model))
plt.ylim((max_seq_len,0))
plt.ylabel('Position')
plt.colorbar()
plt.show()
```
![positional_encodings](https://user-images.githubusercontent.com/36740868/202588564-7a59b574-b081-49e0-ab5f-6aafa0d33d95.png)

## sublayers/
As mentioned in the paper "[There are] two
sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network".
These sub-layers are included in the sublayers directory.
### Scaled Dot-Product Attention
This is the building block of Multi-Head Attention, and is made up of a series of matrix operations as illustraded in the paper:
<img width="168" alt="Screenshot 2022-11-17 at 7 31 44 PM" src="https://user-images.githubusercontent.com/36740868/202589207-2a93e18b-7132-442a-94bf-f179af9c07c9.png">
```
# sublayers/scaled_dot_product_attention.py

from embeddings.postional_encoder import PositionalEncoder
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention as described in
    section 3.2.1 of Attention is All You Need
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, q, k, v, mask=None):
        matmul = torch.einsum("bqhd,bkhd->bhqk", [q, k])
        scaled_matmul = matmul / (self.d_model**0.5)

        if mask is not None:
            scaled_matmul = scaled_matmul.masked_fill(mask == 0, float(1e-20))

        softmax = torch.softmax(scaled_matmul, dim=-1)
        attention = torch.einsum("bhqk, bvhd->bqhd", [softmax, v])

        return attention
```
### Multi-Head Attention
To have multiple attention heads, we simply split the query into multiple heads, calculate the scaled dot-product attention on each head and 
concatenating the results. The concatenation is then passed through a linear layer.

<img width="266" alt="Screenshot 2022-11-17 at 7 36 11 PM" src="https://user-images.githubusercontent.com/36740868/202589663-3ed37bec-6331-46eb-bb93-a840de23bedd.png">

```
# sublayers/scaled_dot_product_attention.py

import torch.nn as nn

from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiheadAttention(nn.Module):
    """
    Implements multi-head attention as described in section 3.2.2 of Attenton is All You Need.
    """

    def __init__(self, d_model, heads_num):
        super().__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        self.d_heads = self.d_model // self.heads_num
        assert (
            self.d_heads * self.heads_num == self.d_model
        ), "Embedding size must be divisible by number of heads"

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.attention = ScaledDotProductAttention(d_model)
        self.w_o = nn.Linear(self.heads_num * self.d_heads, self.d_model)

    def split(self, tensor):
        """
        Splits tensor by number of heads, self.heads_num creating an extra dim

        Args:
            tensor(nn.tensor): tensor of size [batch_size, tensor_len, d_model]

        Returns:
            tensor(nn.tensor): reshaped input tensor of size [batch_size, tensor_len, heads_num, d_tensor]
        """

        batch_size, tensor_len, tensor_dim = tensor.size()
        return tensor.reshape(
            batch_size, tensor_len, self.heads_num, tensor_dim // self.heads_num
        )

    def concat(self, tensor):
        """
        Concatenates the input tensor, opposite of self.split() by reshaping

        Args:
            tensor(nn.tensor): tensor of size [batch_size, tensor_len, heads_num, heads_dim]

        Returns:
            tensort(nn.tensort): reshaped input tensor of size [batch_size, tensor_len, heads_num * heads_dim]
        """

        batch_size, tensor_len, heads_num, heads_dim = tensor.size()
        return tensor.reshape(batch_size, tensor_len, heads_num * heads_dim)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split q, k, v into heads, i.e. from batch_size x q_len x d_model => batch_size x q_len x heads_num x d_heads
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        attention_out = self.attention(q, k, v, mask)
        attention_concat = self.concat(attention_out)
        multihead_attenton_out = self.w_o(attention_concat)
        return multihead_attenton_out
        
```
### Point-wise Feed-forward SubLayer:
"In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
connected feed-forward network, which is applied to each position separately and identically. This
consists of two linear transformations with a ReLU activation in between."
<img width="340" alt="Screenshot 2022-11-17 at 7 46 16 PM" src="https://user-images.githubusercontent.com/36740868/202590664-bc253ff8-f4e8-4362-9525-14b57a99ba89.png">

```
# sublayers/point_wise_feed_forward.py

import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    """
    Implements the point-wise feed-forward sublayer
    used in the Encoder and Decoder as describe in
    section 3.3 of Attention is All You Need:
    It consists of two linear transformations with a
    ReLU activation in between.
    """

    def __init__(self, d_model, forward_expansion):
        """
        Args:
            d_model(int): embedding size
            forward_expansion(int): the multiple that determines
                                    the inner layers' dim, e.g. 4
                                    according to the paper, 2048 = d_model * 4
        """
        super().__init__()
        self.d_model = d_model
        self.point_wise_ff = nn.Sequential(
            nn.Linear(d_model, d_model * forward_expansion),
            nn.ReLU(),
            nn.Linear(d_model * forward_expansion, d_model),
        )

    def forward(self, x):
        return self.point_wise_ff(x)
```
## layers/
The layers directory contains implementation of the Encoder and Decoder layers. Each are a stack of `N = 6` identical layers. The class `EcoderLayer`
is the building block of the Encoder stack:
### Encoder
<img width="216" alt="Screenshot 2022-11-17 at 7 50 27 PM" src="https://user-images.githubusercontent.com/36740868/202591123-bddd6252-6588-4a78-ae8c-8c0fb523cbcf.png">

```
# layers/encoder.py

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
```
The `Encoder` class simply copies `N` of these layers:
```
# layers/encoder.py

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
```
### Decoder
Similarly, the `Decoder` is composed of `N` `DecoderLayers`:
<img width="217" alt="Screenshot 2022-11-17 at 7 52 13 PM" src="https://user-images.githubusercontent.com/36740868/202591279-f627caa9-4cd0-4cb4-a563-61e42ca5d4a2.png">
```
# layers/decoder.py

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
```
## transformer.py

And finally all the pieces come together in the  `Transformer` class:
```
# transformer.py

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
```

