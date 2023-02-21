import torch.nn as nn
import torch


class StipformerLoss(nn.Module):
    def __init__(self, lambda1=0.05, lambda2=0.0005):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, output, target):
        pass


class CharLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction="mean"):
        super().__init__()
        if reduction != "mean" or reduction != "sum":
            raise ValueError("Reduction type not supported")
        else:
            self.reduction = reduction
        self.eps = eps

    def forward(self, output, target):
        diff = output - target

        out = torch.sqrt((diff * diff) + (self.eps * self.eps))
        if self.reduction == "mean":
            out = torch.mean(out)
        else:
            out = torch.sum(out)

        return out
