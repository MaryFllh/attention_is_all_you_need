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
