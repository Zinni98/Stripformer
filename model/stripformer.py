import torch.nn as nn
import torch
from .strip_modules import FEB, Decoder
from .attention_paper import AttentionBlocks


class Stripformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_embedding1 = FEB(3, 64, 128)
        self.feature_embedding2 = FEB(128, 128, 320)
        self.bottleneck = AttentionBlocks(6, 320, 5)
        self.dec = Decoder()

    def forward(self, x):
        h = x
        x, res2 = self.feature_embedding1(x)
        x, res1 = self.feature_embedding2(x)

        x = self.bottleneck(x)

        x = self.dec(x, res1, res2)

        x = x + h

        return x


if __name__ == "__main__":
    k = torch.randn([2, 3, 100, 100])
    strip = Stripformer()

    result = strip(k)
    print(result.shape)
