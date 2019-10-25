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
        self.down1 = down(64, 128)
        self.down2 = down(128, 128)
        self.down3 = down(128, 128)
        self.down4 = down(128, 128)
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
    IDENTITY_TRANSFORM = [[[1., 0., 0.], [0., 1., 0.]]]

    def __init__(self, channels_in: int, channels_out: int):
        super(Decoder, self).__init__()
        self.up1 = upconv(128, 128, bilinear=False)
        self.up2 = upconv(128, 128, bilinear=False)
        self.up_flow = upconv(128, 64, bilinear=False)
        self.up_residual = upconv(128, 64, bilinear=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.flow = upconv(64, 2, bilinear=False)
        self.residual = upconv(64, channels_out, bilinear=False)

    def forward(self, x: nn.Module) -> nn.Module:
        x = self.up1(x)
        x = self.up2(x)
        f = self.up_flow(x)
        r = self.up_residual(x)
        r = self.tanh(self.residual(r))
        identity_theta = torch.tensor(
            Decoder.IDENTITY_TRANSFORM * x.shape[0]).cuda()
        f = self.tanh(self.flow(f).permute(0, 2, 3, 1)) + \
            F.affine_grid(identity_theta, r.shape)
        return f, r
