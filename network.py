#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

from network_parts import Sign, double_conv, down, inconv, outconv, up, upconv


class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, use_context: bool) -> None:
        super().__init__()
        self.encode_frame1 = nn.Sequential(
            inconv(channels_in // 2, 128),
            down(128, 256),
        )
        self.encode_frame2 = nn.Sequential(
            inconv(channels_in // 2, 128),
            down(128, 256),
        )
        self.encode_context = inconv(512, 512)
        self.encode = nn.Sequential(
            down(512, 512),
            down(512, 512),
            down(512, 512),
            nn.Conv2d(512, channels_out, kernel_size=3, padding=1),
        )
        self.use_context = use_context

    def forward(self, frame1, frame2, context_vec: torch.Tensor) -> nn.Module:
        # frames_x = torch.cat(
            # (self.encode_frame1(frame1), self.encode_frame2(frame2)), dim=1)
        frames_x = torch.cat(
            (self.encode_frame1(frame2-frame1), self.encode_frame2(frame2-frame1)), dim=1)
        context_x = self.encode_context(context_vec) if self.use_context else 0.
        return self.encode(frames_x + context_x)


class ResNetEncoder(nn.Module):
    def __init__(self, channels_int: int, channels_out: int, use_context: bool) -> None:
        super().__init__()


class BitToFlowDecoder(nn.Module):
    IDENTITY_TRANSFORM = [[[1., 0., 0.], [0., 1., 0.]]]

    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.ups = nn.Sequential(
            upconv(channels_in, 512, bilinear=False),
            upconv(512, 512, bilinear=False),
            upconv(512, 512, bilinear=False),
        )
        self.flow = nn.Sequential(
            upconv(512, 128, bilinear=False),
            outconv(128, 2),
            # nn.Conv2d(128, 2, kernel_size=3, padding=1),
        )
        self.residual = nn.Sequential(
            upconv(512, 128, bilinear=False),
            outconv(128, channels_out),
            # nn.Conv2d(128, channels_out, kernel_size=3, padding=1),
        )

    def forward(self, input_tuple) -> nn.Module:
        x, context_vec = input_tuple
        x = self.ups(x)
        r = self.residual(x)
        identity_theta = torch.tensor(
            BitToFlowDecoder.IDENTITY_TRANSFORM * x.shape[0]).cuda()
        # f = self.flow(x).permute(0, 2, 3, 1) + \
        # F.affine_grid(identity_theta, r.shape)
        f = F.affine_grid(identity_theta, r.shape)
        return f, r, context_vec


class BitToContextDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ups = nn.Sequential(
            upconv(512, 512, bilinear=False),
            upconv(512, 512, bilinear=False),
            upconv(512, 512, bilinear=False),
            outconv(512, 512),
        )

    def forward(self, input_tuple) -> nn.Module:
        x, context_vec = input_tuple
        add_to_context = self.ups(x)
        return add_to_context, (context_vec + add_to_context).clamp(-1., 1.)
        # TODO: Feed in both x and context_vec


class ContextToFlowDecoder(nn.Module):
    IDENTITY_TRANSFORM = [[[1., 0., 0.], [0., 1., 0.]]]

    def __init__(self, channels_out: int) -> None:
        super().__init__()
        self.flow = nn.Sequential(
            upconv(1024, 128, bilinear=False),
            outconv(128, 2),
            nn.Tanh(),
        )
        self.residual = nn.Sequential(
            upconv(1024, 128, bilinear=False),
            outconv(128, channels_out),
            nn.Tanh(),
        )

    def forward(self, input_tuple) -> nn.Module:
        _, context = input_tuple
        x = torch.cat(input_tuple, dim=1)
        r = self.residual(x)
        identity_theta = torch.tensor(
            ContextToFlowDecoder.IDENTITY_TRANSFORM * x.shape[0]).cuda()
        # f = self.flow(x).permute(0, 2, 3, 1) + \
        # F.affine_grid(identity_theta, r.shape)
        f = F.affine_grid(identity_theta, r.shape)
        return f, r, context


class Binarizer(nn.Module):
    def __init__(self, channels_in, bits, use_binarizer=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels_in, bits, kernel_size=1, bias=False)
        self.sign = Sign()
        self.tanh = nn.Tanh()
        self.use_binarizer = use_binarizer

    def forward(self, x):
        if self.use_binarizer:
            x = self.conv(x)
            x = self.tanh(x)
            return self.sign(x)
        else:
            return x
