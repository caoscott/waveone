#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch.nn as nn

from network_parts import double_conv, down, inconv, outconv, up, upconv


class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super(Encoder, self).__init__()
        self.inc = inconv(channels_in, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.down4 = down(256, 256)
        self.tanh = nn.Tanh()

    def forward(self, x: nn.Module) -> nn.Module:
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.tanh(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super(Decoder, self).__init__()
        self.up1 = upconv(256, 256, bilinear=False)
        self.up2 = upconv(256, 128, bilinear=False)
        self.up3 = upconv(128, 64, bilinear=False)
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.flow = upconv(64, 2, bilinear=False)
        self.residual = upconv(64, channels_out, bilinear=False)

    def forward(self, x: nn.Module) -> nn.Module:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        f = self.sigmoid(self.flow(x).permute(0, 2, 3, 1))
        r = self.sigmoid(self.residual(x))
        return f, r
