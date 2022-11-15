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
