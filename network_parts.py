#!/usr/bin/python
from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import nn as nn
from torch import optim
from torch.autograd import Function
from torchvision import datasets, transforms
from torchvision.utils import save_image


class double_conv(nn.Module):
    '''(conv => BN => LeakyReLU) * 2'''

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        downsample: bool = False,
        norm: str = "batch",
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            self.get_norm(norm, out_ch),
            self.get_activation(activation, out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            self.get_norm(norm, out_ch),
            self.get_activation(activation, out_ch),
        )

    def get_norm(self, norm: str, ch: int) -> nn.Module:
        if norm == "off":
            return nn.Identity()
        if norm == "batch":
            return nn.BatchNorm2d(ch)
        if norm == "group":
            return nn.GroupNorm(32, ch)
        raise ValueError("f{norm} normalization not found.")

    def get_activation(self, activation: str, ch: int) -> nn.Module:
        if activation == "leaky_relu":
            return nn.LeakyReLU(inplace=True)
        if activation == "gdn":
            return GDN(ch, inverse=False)
        if activation == "igdn":
            return GDN(ch, inverse=True)
        raise ValueError(f"{activation} activation not found.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = double_conv(in_ch, out_ch, norm="off", activation="gdn")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mpconv = double_conv(
            in_ch, out_ch, downsample=True, norm="off", activation="gdn")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.mpconv(x)


class up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up: nn.Module = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up: nn.Module = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = double_conv(
            in_ch * 2, out_ch, norm="off", activation="igdn")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # type: ignore
        x1 = self.up(x1)
        diff_x = x1.size()[2] - x2.size()[2]
        diff_y = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, [diff_x // 2, int(diff_x / 2),
                        diff_y // 2, int(diff_y / 2)])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class upconv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up: nn.Module = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up: nn.Module = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(out_ch, out_ch, norm="off", activation="igdn")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.up(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.conv(x)
        return x


class SignFunction(Function):
    """
    Variable Rate Image Compression with Recurrent Neural Networks
    https://arxiv.org/abs/1511.06085
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: torch.Tensor,
        x: torch.Tensor,
        is_training: bool = True
    ) -> torch.Tensor:
        # Apply quantization noise while only training
        if is_training:
            prob = x.new(x.size()).uniform_()  # type: ignore
            x = x.clone()
            x[(1 - x) / 2 <= prob] = 1  # type: ignore
            x[(1 - x) / 2 > prob] = -1  # type: ignore
            return x
        else:
            return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # TODO (cywu): consider passing 0 for tanh(x) > 1 or tanh(x) < -1?
        # See https://arxiv.org/pdf/1712.05087.pdf.
        return grad_output, None


class Sign(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return SignFunction.apply(x, self.training)  # type: ignore


class LambdaModule(nn.Module):
    def __init__(self, lambd: Callable[..., Any]) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, *argv: Tuple[Any]):  # type: ignore
        return self.lambd(*argv)


class revnet_block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.f_ch = channels // 2
        self.g_ch = channels - (channels // 2)
        self.f = double_conv(self.f_ch, self.f_ch)
        self.g = double_conv(self.g_ch, self.g_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x1, x2 = x[:, :self.f_ch], x[:, self.f_ch:]
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        y = torch.cat((y2, y1), dim=1)
        return y


# class permute_block(nn.Module):
#     def __init__(self):
#         self.t1000 = torch.tensor([[1., 0.], [0., 0.]])
#         self.t0100 = torch.tensor([[0., 1.], [0., 0.]])
#         self.t0010 = torch.tensor([[0., 0.], [1., 0.]])
#         self.t0001 = torch.tensor([[0., 0.], [0., 1.]])

#     def forward(x):
#         F.conv2d(x, self.t1000, stride=2)


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size())*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch: int,
                 inverse: bool = False,
                 beta_min: float = 1e-6,
                 gamma_init: float = 0.1,
                 reparam_offset: float = 2**-18) -> None:
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.tensor(reparam_offset)

        self.build(ch)

    def build(self, ch: int) -> None:
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self,  # type: ignore
                inputs: torch.Tensor) -> torch.Tensor:
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs
