#!/usr/bin/python
# full assembly of the sub-parts to form the complete net
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from waveone.network_parts import (ConvLSTMCell, SatLU, Sign, double_conv,
                                   down, inconv, outconv, revnet_block, up,
                                   upconv)


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
            down(512, 1024),
            down(1024, 1024),
            down(1024, 1024),
            nn.Conv2d(1024, out_ch, kernel_size=3, padding=1),
        )
        self.use_context = use_context

    def forward(  # type: ignore
            self,
            frame1: torch.Tensor,
            frame2: torch.Tensor,
            context_vec: torch.Tensor
    ) -> torch.Tensor:
        # frames_x = torch.cat(
        #     (self.encode_frame1(frame1), self.encode_frame2(frame2)),
        #     dim=1
        # )
        x = frame2 - frame1
        frames_x = torch.cat(
            (self.encode_frame1(x), self.encode_frame2(x)),
            dim=1
        )
        context_x = self.encode_context(context_vec) if self.use_context else 0.
        return self.encode(frames_x + context_x)


class SmallEncoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_ch // 2, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, out_ch, 3, stride=2, padding=1),
        )

    def forward(  # type: ignore
            self,
            frame1: torch.Tensor,
            frame2: torch.Tensor,
            context_vec: torch.Tensor
    ) -> torch.Tensor:
        # frames_x = torch.cat(
        #     (self.encode_frame1(frame1), self.encode_frame2(frame2)),
        #     dim=1
        # )
        x = frame2 - frame1
        return self.encode(x)


class SmallBinarizer(nn.Module):
    def __init__(
        self,
        use_binarizer: bool = True
    ) -> None:
        super().__init__()
        self.sign = Sign()
        self.tanh = nn.Tanh()
        self.use_binarizer = use_binarizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.tanh(x)
        if self.use_binarizer:
            return self.sign(x)
        else:
            return x


class SmallDecoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 128, 2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, out_ch, 2, stride=2),
            nn.Tanh(),
        )

    def forward(  # type: ignore
            self,
            input_tuple: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _ = input_tuple
        return 0., self.residual(x) * 2, 0.  # type: ignore


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
            upconv(in_ch, 1024, bilinear=False),
            upconv(1024, 512, bilinear=False),
            upconv(512, 256, bilinear=False),
        )
        self.flow = nn.Sequential(
            upconv(256, 128, bilinear=False),
            nn.Conv2d(128, 2, kernel_size=1, bias=False),
            # nn.Conv2d(128, 2, kernel_size=3, padding=1),
        )
        self.residual = nn.Sequential(
            upconv(256, 128, bilinear=False),
            nn.Conv2d(128, out_ch, kernel_size=1, bias=False),
            # nn.Conv2d(128, out_ch, kernel_size=3, padding=1),
        )

    def forward(  # type: ignore
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, context_vec = input_tuple
        x = self.ups(x)
        r = self.residual(x) * 2
        identity_theta = torch.tensor(
            BitToFlowDecoder.IDENTITY_TRANSFORM * x.shape[0]).to(x.device)
        f = self.flow(x).permute(0, 2, 3, 1)
        assert f.shape[-1] == 2
        grid_normalize = torch.tensor(
            f.shape[1: 3]).reshape(1, 1, 1, 2).to(x.device)
        f_grid = f / grid_normalize + F.affine_grid(  # type: ignore
            identity_theta, r.shape, align_corners=False)
        # f = F.affine_grid(identity_theta, r.shape,  # type: ignore
        #                   align_corners=False)
        return f_grid, r, context_vec


class WaveoneModel(nn.Module):
    NAMES = ("encoder", "binarizer", "decoder")

    def __init__(self,
                 encoder: nn.Module,
                 binarizer: nn.Module,
                 decoder: nn.Module,
                 flow_off: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.binarizer = binarizer
        self.decoder = decoder
        self.flow_off = flow_off
        self.nets = (encoder, binarizer, decoder)

    def forward(  # type: ignore
            self,
            frame1: torch.Tensor,
            frame2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        codes = self.binarizer(self.encoder(frame1, frame2, 0.))
        flows, residuals, _ = self.decoder((codes, 0.))
        flow_frame = frame1 if self.flow_off else F.grid_sample(  # type: ignore
            frame1, flows, align_corners=False)
        reconstructed_frame2 = flow_frame + residuals
        if self.training is False:  # type: ignore
            reconstructed_frame2 = torch.clamp(
                reconstructed_frame2, min=-1., max=1.)
        return codes, flows, residuals, flow_frame, reconstructed_frame2


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
            nn.Conv2d(128, 2, kernel_size=1, bias=False),
            nn.Tanh(),
        )
        self.residual = nn.Sequential(
            upconv(1024, 128, bilinear=False),
            nn.Conv2d(128, out_ch, kernel_size=1, bias=False),
            nn.Tanh(),
        )

    def forward(  # type: ignore
        self,
        input_tuple: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, context = input_tuple
        x = torch.cat(input_tuple, dim=1)
        r = self.residual(x) * 2
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
        return self.tanh(y5) * 2


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
    def __init__(self, channels: int, num_blocks: int = 10) -> None:
        super().__init__()
        self.model = nn.Sequential(  # type: ignore
            revnet_block(channels // 2) for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        f1, f2 = x[:, :3], x[:, 3:]
        return self.model(f2 - f1)


class CAE(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)
    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAE, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(
                5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )

        # DECODER

        # 128x64x64
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128,
                               kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256,
                               kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(inplace=True),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(  # type: ignore
            self,
            frame1: torch.Tensor,
            frame2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = frame2 - frame1
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)

        r = self.decode(self.encoded)
        return self.encoded, 0., r, frame1, frame1 + r  # type: ignore

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec * 2


class PredNet(nn.Module):
    def __init__(self, R_channels, A_channels, output_mode='error'):
        super(PredNet, self).__init__()
        self.r_channels = R_channels + (0, )  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        self.set_output_mode(output_mode)

        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i+1],                                                                             self.r_channels[i],
                                (3, 3))
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(
                nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1),
                # nn.ReLU(),
            )
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)

        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(
                2 * self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def set_output_mode(self, output_mode: str) -> None:
        self.output_mode = output_mode
        default_output_modes = ['prediction', 'error']
        assert output_mode in default_output_modes, f'Invalid output_mode: {output_mode}'

    def forward(self, input):  # type: ignore
        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = torch.zeros(
                batch_size, 2*self.a_channels[l], w, h).to(input.device)
            R_seq[l] = torch.zeros(
                batch_size, self.r_channels[l], w, h).to(input.device)
            w = w//2
            h = h//2
        time_steps = input.size(1)
        total_error = []

        for t in range(time_steps):
            A = input[:, t]
            A = A.type(torch.cuda.FloatTensor)

            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                if t == 0:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = (R, R)
                else:
                    E = E_seq[l]
                    R = R_seq[l]
                    hx = H_seq[l]
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    tmp = torch.cat((E, self.upsample(R_seq[l+1])), 1)
                    R, hx = cell(tmp, hx)
                R_seq[l] = R
                H_seq[l] = hx

            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    frame_prediction = A_hat
                pos = F.relu(A_hat - A)
                neg = F.relu(A - A_hat)
                E = torch.cat([pos, neg], 1)
                E_seq[l] = E
                if l < self.n_layers - 1:
                    update_A = getattr(self, 'update_A{}'.format(l))
                    A = update_A(E)
            if self.output_mode == 'error':
                mean_error = torch.cat(
                    [torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in E_seq], 1)
                # batch x n_layers
                total_error.append(mean_error)

        if self.output_mode == 'error':
            return torch.stack(total_error, 2)  # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            return frame_prediction.clamp(-1., 1.)
