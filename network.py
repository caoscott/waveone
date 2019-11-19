#!/usr/bin/python
# full assembly of the sub-parts to form the complete net
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from waveone.network_parts import (Sign, double_conv, down, inconv, outconv,
                                   revnet_block, up, upconv)


class Encoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_context: bool) -> None:
        super().__init__()
        self.encode_frame1 = nn.Sequential(
            inconv(in_ch // 2, 128),
            down(128, 256),
        )
        self.encode_frame2 = nn.Sequential(
            inconv(in_ch // 2, 128),
            down(128, 256),
        )
        self.encode_context = inconv(512, 512)
        self.encode = nn.Sequential(
            down(512, 512),
            down(512, 512),
            down(512, 512),
            nn.Conv2d(512, out_ch, kernel_size=3, padding=1),
        )
        self.use_context = use_context

    def forward(  # type: ignore
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        context_vec: torch.Tensor
    ) -> torch.Tensor:
        # frames_x = torch.cat(
            # (self.encode_frame1(frame1), self.encode_frame2(frame2)), dim=1)
        frames_x = torch.cat(
            (self.encode_frame1(frame1), self.encode_frame2(frame2)),
            dim=1
        )
        context_x = self.encode_context(context_vec) if self.use_context else 0.
        return self.encode(frames_x + context_x)


# class FeatureEncoder(nn.Module):
#     def __init__(self, out_ch: int, use_context: bool, shrink: int = 1) -> None:
#         super().__init__()
#         self.down2 = down(128 // shrink, 256 // shrink)
#         self.down3 = down(256 // shrink, 512 // shrink)
#         self.down4 = down(512 // shrink, 512 // shrink)
#         self.use_context = use_context

#     def forward(  # type: ignore
#         self,
#         frame1: torch.Tensor,
#         frame2: torch.Tensor,
#         context_vec: torch.Tensor
#     ) -> torch.Tensor:
#         # frames_x = torch.cat(
#             # (self.encode_frame1(frame1), self.encode_frame2(frame2)), dim=1)
#         frames_x = torch.cat(
#             (self.encode_frame1(frame2-frame1), self.encode_frame2(frame2-frame1)),
#             dim=1
#         )
#         context_x = self.encode_context(context_vec) if self.use_context else 0.
#         return self.encode(frames_x + context_x)


class BitToFlowDecoder(nn.Module):
    IDENTITY_TRANSFORM = [[[1., 0., 0.], [0., 1., 0.]]]

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.ups = nn.Sequential(
            upconv(in_ch, 512, bilinear=False),
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
            outconv(128, out_ch),
            nn.Tanh(),
            # nn.Conv2d(128, out_ch, kernel_size=3, padding=1),
        )

    def forward(  # type: ignore
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, context_vec = input_tuple
        x = self.ups(x)
        r = self.residual(x)  # .clamp(-1., 1.)
        identity_theta = torch.tensor(
            BitToFlowDecoder.IDENTITY_TRANSFORM * x.shape[0]).cuda()
        # f = self.flow(x).permute(0, 2, 3, 1) + \
        # F.affine_grid(identity_theta, r.shape)
        f = F.affine_grid(identity_theta, r.shape)  # type: ignore
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

    def forward(  # type: ignore
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, context_vec = input_tuple
        add_to_context = self.ups(x)
        return add_to_context, (context_vec + add_to_context).clamp(-1., 1.)
        # TODO: Feed in both x and context_vec


class ContextToFlowDecoder(nn.Module):
    IDENTITY_TRANSFORM = [[[1., 0., 0.], [0., 1., 0.]]]

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        self.flow = nn.Sequential(
            upconv(1024, 128, bilinear=False),
            outconv(128, 2),
            nn.Tanh(),
        )
        self.residual = nn.Sequential(
            upconv(1024, 128, bilinear=False),
            outconv(128, out_ch),
            nn.Tanh(),
        )

    def forward(  # type: ignore
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, context = input_tuple
        x = torch.cat(input_tuple, dim=1)
        r = self.residual(x)
        identity_theta = torch.tensor(
            ContextToFlowDecoder.IDENTITY_TRANSFORM * x.shape[0]).cuda()
        # f = self.flow(x).permute(0, 2, 3, 1) + \
        # F.affine_grid(identity_theta, r.shape)
        f = F.affine_grid(identity_theta, r.shape)  # type: ignore
        return f, r, context


class Binarizer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        bits: int,
        use_binarizer: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, bits, kernel_size=1, bias=False)
        self.sign = Sign()
        self.tanh = nn.Tanh()
        self.use_binarizer = use_binarizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.conv(x)
        x = self.tanh(x)
        if self.use_binarizer:
            return self.sign(x)
        else:
            return x


class AutoencoderUNet(nn.Module):
    def __init__(self, n_channels: int, shrink: int) -> None:
        super().__init__()
        self.inc = inconv(n_channels, 64 // shrink)
        self.down1 = down(64 // shrink, 128 // shrink)
        self.down2 = down(128 // shrink, 256 // shrink)
        self.down3 = down(256 // shrink, 512 // shrink)
        self.down4 = down(512 // shrink, 512 // shrink)
        self.up1 = up(512 // shrink, 256 // shrink, bilinear=False)
        self.up2 = up(256 // shrink, 128 // shrink, bilinear=False)
        self.up3 = up(128 // shrink, 64 // shrink, bilinear=False)
        self.up4 = upconv(64 // shrink, 64 // shrink, bilinear=False)
        self.outconv = outconv(64 // shrink, n_channels // 2)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y1 = self.up1(x5, x4)
        y2 = self.up2(y1, x3)
        y3 = self.up3(y2, x2)
        y4 = self.up4(y3)
        y5 = self.outconv(y4)
        return self.tanh(y5)


class UNet(nn.Module):
    def __init__(self, in_ch: int, shrink: int) -> None:
        super().__init__()
        self.inc = inconv(in_ch, 64 // shrink)
        self.down1 = down(64 // shrink, 128 // shrink)
        self.down2 = down(128 // shrink, 256 // shrink)
        self.down3 = down(256 // shrink, 512 // shrink)
        self.down4 = down(512 // shrink, 512 // shrink)
        self.up1 = up(512 // shrink, 256 // shrink, bilinear=False)
        self.up2 = up(256 // shrink, 128 // shrink, bilinear=False)
        self.up3 = up(128 // shrink, 64 // shrink, bilinear=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out4 = self.up1(x5, x4)
        out3 = self.up2(out4, x3)
        out2 = self.up3(out3, x2)
        return [out4, out3, out2]


class SimpleRevNet(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 6) -> None:
        super().__init__()
        self.model = nn.Sequential(  # type: ignore
            revnet_block(channels) for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)
