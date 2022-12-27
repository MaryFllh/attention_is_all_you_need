import torch
import torch.nn as nn

# Used https://nlp.seas.harvard.edu/2018/04/03/attention.html#label-smoothing as reference


class LabelSmoothing(nn.module):
    def __init__(self, padding_idx, size, smoothing=0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.size = size
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, x, target):
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill(0, mask.squeeze(), 0)
        self.true_dist = true_dist
        return self.true_dist
