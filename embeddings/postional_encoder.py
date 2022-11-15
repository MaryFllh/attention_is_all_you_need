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
