from turtle import forward
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
