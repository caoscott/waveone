#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

from network_parts import double_conv, down, inconv, outconv, up, upconv


class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            inconv(channels_in, 64),
            down(64, 128),
            down(128, 256),
            down(256, 512),
            down(512, 512),
            down(512, 512),
            nn.Tanh())

    def forward(self, x: nn.Module) -> nn.Module:
        return self.encode(x)


class Decoder(nn.Module):
    IDENTITY_TRANSFORM = [[[1., 0., 0.], [0., 1., 0.]]]

    def __init__(self, channels_in: int, channels_out: int):
        super(Decoder, self).__init__()
        self.ups = nn.Sequential(
            upconv(512, 512, bilinear=False),
            upconv(512, 256, bilinear=False),
            upconv(256, 128, bilinear=False))
        self.flow = nn.Sequential(
            upconv(128, 64, bilinear=False),
            upconv(64, 2, bilinear=False),
            nn.Tanh())
        self.residual = nn.Sequential(
            upconv(128, 64, bilinear=False),
            upconv(64, channels_out, bilinear=False),
            nn.Tanh())

    def forward(self, x: nn.Module) -> nn.Module:
        x = self.ups(x)
        r = self.residual(x)
        identity_theta = torch.tensor(
            Decoder.IDENTITY_TRANSFORM * x.shape[0]).cuda()
        f = self.flow(x).permute(0, 2, 3, 1) + \
            F.affine_grid(identity_theta, r.shape)
        return f, r
