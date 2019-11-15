#!/usr/bin/python

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class double_conv(nn.Module):
    '''(conv => BN => ELU) * 2'''

    def __init__(self, in_ch, out_ch, downsample=False, norm="batch"):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(
                out_ch) if norm == "batch" else nn.GroupNorm(32, out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(
                out_ch) if norm == "batch" else nn.GroupNorm(32, out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = double_conv(in_ch, out_ch, downsample=True)

    def forward(self, x):
        return self.mpconv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = double_conv(in_ch * 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class upconv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(out_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SignFunction(Function):
    """
    Variable Rate Image Compression with Recurrent Neural Networks
    https://arxiv.org/abs/1511.06085
    """

    @staticmethod
    def forward(ctx, x, is_training=True):
        # Apply quantization noise while only training
        if is_training:
            prob = x.new(x.size()).uniform_()
            x = x.clone()
            x[(1 - x) / 2 <= prob] = 1
            x[(1 - x) / 2 > prob] = -1
            return x
        else:
            return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # TODO (cywu): consider passing 0 for tanh(x) > 1 or tanh(x) < -1?
        # See https://arxiv.org/pdf/1712.05087.pdf.
        return grad_output, None


class Sign(nn.Module):
    def forward(self, x):
        return SignFunction.apply(x, self.training)


class Pass(nn.Module):
    def forward(self, x):
        return x
