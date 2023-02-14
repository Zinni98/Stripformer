import torch.nn as nn
from einops import rearrange
import torch
import math


class MLPBlock(nn.Module):
    # taken from original repository of the paper (Not much information about this
    # module on the paper)
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.l_norm = nn.LayerNorm(self.in_channels)
        self.fc1 = nn.Linear(self.in_channels,
                             self.in_channels * 4
                             )
        self.fc2 = nn.Linear(self.in_channels*4,
                             self.in_channels
                             )
        self.activation = nn.GELU()
        self.cpe = nn.Conv2d(self.in_channels,
                             self.in_channels,
                             kernel_size=3,
                             padding=1,
                             groups=self.in_channels
                             )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        _, _, height, width = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        in_f = x
        x = self.l_norm(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x) + in_f
        x = rearrange(x, "b (h w) c -> b c h w", h=height, w=width)
        x = self.cpe(x) + x
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        """
        Applies the multi-head attention mechanism on the given input tensors.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape `(batch_size, sequence_length, channels)`.
        k : torch.Tensor
            Key tensor of shape `(batch_size, sequence_length, channels)`.
        v : torch.Tensor
            Value tensor of shape `(batch_size, sequence_length, channels)`.

        Returns
        -------
        torch.Tensor
            Flattened tensor of shape `(batch_size, sequence_length, channels)`.

        Raises
        ------
        ValueError
            If the number of channels in the query tensor does not divide the number
            of heads.

        Examples
        --------
        >>> model = MultiHeadAttention(8)
        >>> q = torch.rand((16, 32, 128))
        >>> k = torch.rand((16, 32, 128))
        >>> v = torch.rand((16, 32, 128))
        >>> output = model(q, k, v)
        """
        _, _, c = q.size()
        if c % self.heads != 0:
            raise ValueError("Number of heads should divide \
                              the number of channels")
        # I am already transposing the tensors to allow matrix multiplication
        # Here I am creating a new dimension having the number of heads (i.e. h)
        query = rearrange(q, "b n (h c) -> b h n c", h=self.heads)
        key = rearrange(k, "b n (h c) -> b h c n", h=self.heads)
        value = rearrange(v, "b n (h c) -> b h n c", h=self.heads)
        _, _, _, d = query.size()
        pre_soft = torch.einsum("bhnc,bhcm->bhnm", query, key)
        att_probs = self.softmax(pre_soft/math.sqrt(d))
        final = torch.einsum("bhmn,bhnc->bhmc", att_probs, value)
        # concatenating all heads and flattening the heads dimension
        flat_final = rearrange(final, "b h n c -> b n (h c)")
        return flat_final
