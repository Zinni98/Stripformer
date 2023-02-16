import torch.nn as nn
from strip_modules import FEB, Decoder
from attention import AttentionBlocks


class Stripformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_embedding1 = FEB(3, 64, 128)
        self.feature_embedding2 = FEB(128, 128, 320)
        self.bottleneck = AttentionBlocks(6, 320, 5)

    def forward(self, x):
        pass
