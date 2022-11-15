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
