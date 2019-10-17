#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

from network_parts import double_conv, down, inconv, outconv, up, upconv


class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super(Encoder, self).__init__()
        self.inc = inconv(channels_in, 64)
        self.down1 = down(64, channels_out)
        self.down2 = down(channels_out, channels_out)
        self.down3 = down(channels_out, channels_out)
        self.down4 = down(channels_out, channels_out)

    def forward(self, x: nn.Module) -> nn.Module:
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super(Decoder, self).__init__()
        self.up1 = upconv(channels_in, channels_in, bilinear=False)
        self.up2 = upconv(channels_in, channels_in, bilinear=False)
        self.up3 = upconv(channels_in, 64, bilinear=False)
        self.flow = upconv(64, 2, bilinear=False)
        self.residual = upconv(64, channels_out, bilinear=False)

    def forward(self, x: nn.Module) -> nn.Module:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        f = self.flow(x)
        r = self.residual(x)
        return f, r
