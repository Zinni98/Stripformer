import torch.nn as nn
from einops import rearrange
import torch
from .misc import MultiHeadAttention, MLPBlock


class IntraSA(nn.Module):
    def __init__(self, channels: int,
                 heads: int = 5) -> None:
        """

        Parameters
        ----------
        channels : int
            Number of channels of the input image

        heads : int
            Number of heads for the multi-head self attention mechanism (default = 5)
        """
        super().__init__()
        self.channels = channels
        self.heads = heads
        # Number of channels for horizontal and vertical attention
        self.split_channels = channels // 2

        self.l_norm = nn.LayerNorm(self.channels)
        self.conv1 = nn.Conv2d(self.channels,
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

        self.attn = MultiHeadAttention(heads)

        self.conv2 = nn.Conv2d(self.channels,
                               self.channels,
                               kernel_size=1,
                               )

        self.mlp = MLPBlock(self.channels)

    def forward(self, x, batch_dim=0):
        input_f = x
        sz = x.size()
        if len(sz) != 4:
            raise ValueError(f"Input has wrong number of dimensions: \
                               expected 4, got {len(sz)}")
        batch_size = sz[batch_dim]
        x = rearrange(x,
                      "b c h w -> b h w c")
        x = self.l_norm(x)
        x = rearrange(x,
                      "b h w c -> b c h w")
        x = self.conv1(x)

        # Dividing the number of channels
        x_horiz, x_vert = torch.chunk(x, chunks=2, dim=1)

        # Keeping the naming consistent with the paper: d = c/2
        x_horiz = rearrange(x_horiz,
                            "b d h w -> (b h) w d")
        x_vert = rearrange(x_vert,
                           "b d h w -> (b w) h d")

        # Splitting heads inside the attention module, not here
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

        attn_out = self.conv2(torch.cat((attn_horiz, attn_vert), dim=1)) + input_f

        x = self.mlp(attn_out)

        return x


class InterSA(nn.Module):
    def __init__(self, channels, heads=5):
        super().__init__()
        self.channels = channels
        self.heads = heads
        # Number of channels for horizontal and vertical attention
        self.split_channels = channels // 2

        self.l_norm = nn.LayerNorm(self.channels)
        self.conv1 = nn.Conv2d(self.channels,
                               self.channels,
                               kernel_size=1,
                               )
        # In the implementation found in the official repository
        # queries, keys and values are computed using a convolution
        # whereas on the paper, it is said that it is a linear projection
        # I decided to stick with the implementation found on the repo.
        self.conv_h = nn.Conv2d(self.split_channels,
                                3*self.split_channels,
                                kernel_size=1,
                                padding=0
                                )
        self.conv_v = nn.Conv2d(self.split_channels,
                                3*self.split_channels,
                                kernel_size=1,
                                padding=0
                                )

        self.attn = MultiHeadAttention(heads)

        self.conv2 = nn.Conv2d(self.channels,
                               self.channels,
                               kernel_size=1,
                               )

        self.mlp = MLPBlock(self.channels)

    def forward(self, x, batch_dim=0):
        input_f = x
        sz = x.size()
        if len(sz) != 4:
            raise ValueError(f"Input has wrong number of dimensions: \
                               expected 4, got {len(sz)}")

        x = rearrange(x,
                      "b c h w -> b h w c"
                      )
        x = self.l_norm(x)
        x = rearrange(x,
                      "b h w c -> b c h w"
                      )
        x = self.conv1(x)

        x_horiz, x_vert = torch.chunk(x, chunks=2, dim=1)

        q_horiz, k_horiz, v_horiz = torch.chunk(self.conv_h(x_horiz), 3, dim=1)
        q_vert, k_vert, v_vert = torch.chunk(self.conv_h(x_vert), 3, dim=1)

        q_horiz = rearrange(q_horiz, "b c h w -> b h (c w)")
        k_horiz = rearrange(k_horiz, "b c h w -> b h (c w)")
        v_horiz = rearrange(v_horiz, "b c h w -> b h (c w)")

        q_vert = rearrange(q_vert, "b c h w -> b w (c h)")
        k_vert = rearrange(k_vert, "b c h w -> b w (c h)")
        v_vert = rearrange(v_vert, "b c h w -> b w (c h)")

        # (b h) w d
        attn_horiz = self.attn(q_horiz, k_horiz, v_horiz)
        attn_horiz = rearrange(attn_horiz,
                               "b h (d w) -> b d h w",
                               d=self.split_channels)

        # (b w) h d
        attn_vert = self.attn(q_vert, k_vert, v_vert)
        attn_vert = rearrange(attn_vert,
                              "b w (d h) -> b d h w",
                              d=self.split_channels)

        attn_out = self.conv2(torch.cat((attn_horiz, attn_vert), dim=1)) + input_f

        x = self.mlp(attn_out)

        return x


class AttentionBlocks(nn.Module):
    def __init__(self, blocks, channels, heads):
        """
        Creates a module having a number of IntraSA and InterSA blocks,
        passed as parameter

        Parameters
        ----------
        blocks : int
            Number of IntraSA and IterSA blocks
        """
        super().__init__()

        self.layers = nn.ModuleList([sub for i in range(blocks)
                                     for sub in (IntraSA(channels, heads),
                                                 InterSA(channels, heads))])

    def forward(self, x):
        for i, _ in enumerate(self.layers):
            x = self.layers[i](x)

        return x


if __name__ == "__main__":
    x = torch.randn([100, 10, 100, 100])
    intra = AttentionBlocks(2, 10, 5)

    res = intra(x)
    print(res.shape)
