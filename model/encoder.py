import torch.nn as nn


class FEB(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int = None
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not mid_channels:
            self.mid_channels = in_channels
        else:
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
