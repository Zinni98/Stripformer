import torch
import torch.nn as nn
from .attention import AttentionBlocks


class FEB(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.activation = nn.LeakyReLU(.2, inplace=True)

        self.layer_1 = nn.Sequential(nn.Conv2d(self.in_channels,
                                               self.mid_channels,
                                               kernel_size=3,
                                               padding=1),
                                     self.activation
                                     )
        self.res_1 = nn.Sequential(nn.Conv2d(self.mid_channels,
                                             self.mid_channels,
                                             kernel_size=3,
                                             padding=1),
                                   self.activation,
                                   nn.Conv2d(self.mid_channels,
                                             self.mid_channels,
                                             kernel_size=3,
                                             padding=1)
                                   )
        self.res_2 = nn.Sequential(nn.Conv2d(self.mid_channels,
                                             self.mid_channels,
                                             kernel_size=3,
                                             padding=1),
                                   self.activation,
                                   nn.Conv2d(self.mid_channels,
                                             self.mid_channels,
                                             kernel_size=3,
                                             padding=1)
                                   )
        self.res_3 = nn.Sequential(nn.Conv2d(self.mid_channels,
                                             self.mid_channels,
                                             kernel_size=3,
                                             padding=1),
                                   self.activation,
                                   nn.Conv2d(self.mid_channels,
                                             self.mid_channels,
                                             kernel_size=3,
                                             padding=1)
                                   )
        self.downsample = nn.Sequential(nn.Conv2d(self.mid_channels,
                                                  self.out_channels,
                                                  stride=2,
                                                  kernel_size=3,
                                                  padding=1),
                                        self.activation
                                        )

    def forward(self, x):
        """

        Returns
        -------
        tuple
            two tensors, the first one is the result after downsampling, and the
            other is the result before downsampling (in order to be used for
            long range residual connection)
        """
        x = self.layer_1(x)
        x = self.activation(self.res_1(x) + x)
        x = self.activation(self.res_2(x) + x)
        res = self.activation(self.res_3(x) + x)

        x = self.downsample(res)

        return x, res


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.activation = nn.LeakyReLU(.2, inplace=True)

        self.upsample_layer1 = nn.Sequential(nn.ConvTranspose2d(in_channels=320,
                                                                out_channels=192,
                                                                kernel_size=4,
                                                                stride=2,
                                                                padding=1),
                                             self.activation)

        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels=192+128,
                                                   out_channels=192,
                                                   kernel_size=1,
                                                   padding=0),
                                         self.activation)

        self.att_layer = AttentionBlocks(blocks=3,
                                         channels=192,
                                         heads=3)

        self.upsample_layer2 = nn.Sequential(nn.ConvTranspose2d(in_channels=192,
                                                                out_channels=64,
                                                                kernel_size=4,
                                                                stride=2,
                                                                padding=1),
                                             self.activation)

        self.res_layer1 = nn.Sequential(nn.Conv2d(in_channels=128,
                                                  out_channels=64,
                                                  kernel_size=1,
                                                  padding=0),
                                        self.activation,
                                        nn.Conv2d(in_channels=64,
                                                  out_channels=64,
                                                  kernel_size=3,
                                                  padding=1))

        self.res_layer2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                  out_channels=64,
                                                  kernel_size=3,
                                                  padding=1),
                                        self.activation,
                                        nn.Conv2d(in_channels=64,
                                                  out_channels=64,
                                                  kernel_size=3,
                                                  padding=1))

        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                   out_channels=3,
                                                   kernel_size=3,
                                                   padding=1),
                                         self.activation)

    def forward(self, x, residual_1, residual_2):
        x = self.upsample_layer1(x)

        x = self.conv_layer1(torch.concat((x, residual_1), dim=1))
        # x = self.att_layer(x)

        x = self.upsample_layer2(x)

        x = self.activation(self.res_layer1(torch.cat((x, residual_2), dim=1)) + x)

        x = self.activation(self.res_layer2(x) + x)

        x = self.conv_layer2(x)

        return x


if __name__ == "__main__":
    x = torch.randn([2, 3, 10, 10])
    f = FEB(3, 64, 128)

    f(x)
