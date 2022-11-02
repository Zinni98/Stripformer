import torch.nn as nn
from einops import rearrange
import torch


class Attention(nn.Module):
    def __init__(self, heads):
        super().__init__()

    def forward(self, q, k, v):
        pass


class IntraSA(nn.Module):
    def __init__(self, channels: int,
                 heads: int = 5) -> None:
        """

        Parameters
        ----------
        channels : int
            Number of channels of the input image

        heads : int
            Number of heads for the multi-head self attention mechanism
        """
        super().__init__()
        self.channels = channels
        self.heads = heads
        # Number of channels for horizontal and vertical attention
        self.split_channels = channels // 2

        self.l_norm = nn.LayerNorm(self.channels)
        self.conv = nn.Conv2d(self.channels,
                              self.channels,
                              kernel_size=1,
                              )
        # linear projection to obtain queries horizontally
        self.p_q_h = nn.Linear(self.split_channels,
                               self.split_channels)
        # linear projection to obtain keys horizontally
        self.p_k_h = nn.Linear(self.split_channels,
                               self.split_channels)
        # linear projection to obtain values horizontally
        self.p_v_h = nn.Linear(self.split_channels,
                               self.split_channels)

        # linear projection to obtain queries vertically
        self.p_q_v = nn.Linear(self.split_channels,
                               self.split_channels)
        # linear projection to obtain keys vertically
        self.p_k_v = nn.Linear(self.split_channels,
                               self.split_channels)
        # linear projection to obtain values vertically
        self.p_v_v = nn.Linear(self.split_channels,
                               self.split_channels)

        self.attn = Attention(heads)

    def forward(self, x):
        sz = x.size()
        if len(sz) != 4:
            raise ValueError(f"Input has wrong number of dimensions: \
                               expected 4, got {len(sz)}")
        batch_size = sz[0]
        x = rearrange(x,
                      "b c h w -> b h w c")
        x = self.l_norm(x)
        x = rearrange(x,
                      "b h w c -> b c h w")
        x = self.conv(x)

        x_horiz, x_vert = torch.chunk(x, chunks=2, dim=1)

        # Keeping the naming consistent with the paper: d = c/2
        x_horiz = rearrange(x_horiz,
                            "b d h w -> (b h) w d")
        x_vert = rearrange(x_vert,
                           "b d h w -> (b w) h d")

        q_horiz = self.p_q_h(x_horiz)
        k_horiz = self.p_k_h(x_horiz)
        v_horiz = self.p_v_h(x_horiz)

        q_vert = self.p_q_v(x_vert)
        k_vert = self.p_k_v(x_vert)
        v_vert = self.p_v_v(x_vert)

        # (b h) w d
        attn_horiz = self.attn(q_horiz, k_horiz, v_horiz)
        attn_horiz = rearrange(attn_horiz,
                               "(b h) w d -> b d h w",
                               b=batch_size)

        # (b w) h d
        attn_vert = self.attn(q_vert, k_vert, v_vert)
        attn_vert = rearrange(attn_vert,
                              "(b w) h d -> b d h w",
                              b=batch_size)
        attn = torch.cat((attn_horiz, attn_vert))


if __name__ == "__main__":
    x = torch.randn([100, 4, 12, 10])
    x_horiz, x_vert = torch.chunk(x, chunks=2, dim=1)

    x_horiz = rearrange(x_horiz, "b d h w -> (b h) w d")
    x_vert = rearrange(x_vert, "b d h w -> (b w) h d")
    print(x_horiz.size())
    print(x_vert.size())
