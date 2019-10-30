#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

from network_parts import double_conv, down, inconv, outconv, up, upconv


class Encoder(nn.Module):
    def __init__(self, channels_in: int, use_context: bool):
        super().__init__()
        self.encode_frames = nn.Sequential(
            inconv(channels_in, 64),
            down(64, 128),
        )
        self.encode_context = inconv(128, 128)
        self.encode = nn.Sequential(
            down(128, 128),
            down(128, 128),
            down(128, 128),
            nn.Tanh())
        self.use_context = use_context

    def forward(self, x: nn.Module, context_vec: torch.Tensor) -> nn.Module:
        print(context_vec.shape)
        x = self.encode_frames(x) + (
            self.encode_context(context_vec) if self.use_context else 0.)
        return self.encode(x)


class BitToFlowDecoder(nn.Module):
    IDENTITY_TRANSFORM = [[[1., 0., 0.], [0., 1., 0.]]]

    def __init__(self, channels_out: int):
        super().__init__()
        self.ups = nn.Sequential(
            upconv(128, 128, bilinear=False),
            upconv(128, 128, bilinear=False),
            upconv(128, 128, bilinear=False))
        self.flow = nn.Sequential(
            upconv(128, 64, bilinear=False),
            outconv(64, 2),
            nn.Tanh())
        self.residual = nn.Sequential(
            upconv(128, 64, bilinear=False),
            outconv(64, channels_out),
            nn.Tanh())

    def forward(self, input_tuple) -> nn.Module:
        x, _ = input_tuple
        x = self.ups(x)
        r = self.residual(x)
        identity_theta = torch.tensor(
            BitToFlowDecoder.IDENTITY_TRANSFORM * x.shape[0]).cuda()
        f = self.flow(x).permute(0, 2, 3, 1) + \
            F.affine_grid(identity_theta, r.shape)
        return f, r, 0.


class BitToContextDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ups = nn.Sequential(
            upconv(128, 128, bilinear=False),
            upconv(128, 128, bilinear=False),
            upconv(128, 128, bilinear=False))

    def forward(self, input_tuple) -> nn.Module:
        x, context_vec = input_tuple
        add_to_context = self.ups(x)
        return context_vec + add_to_context


class ContextToFlowDecoder(nn.Module):
    IDENTITY_TRANSFORM = [[[1., 0., 0.], [0., 1., 0.]]]

    def __init__(self, channels_out: int):
        super().__init__()
        self.flow = nn.Sequential(
            upconv(128, 64, bilinear=False),
            outconv(64, 2),
            nn.Tanh())
        self.residual = nn.Sequential(
            upconv(128, 64, bilinear=False),
            outconv(64, channels_out),
            nn.Tanh())

    def forward(self, x) -> nn.Module:
        r = self.residual(x)
        identity_theta = torch.tensor(
            ContextToFlowDecoder.IDENTITY_TRANSFORM * x.shape[0]).cuda()
        f = self.flow(x).permute(0, 2, 3, 1) + \
            F.affine_grid(identity_theta, r.shape)
        return f, r, x
