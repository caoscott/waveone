#!/usr/bin/python
from typing import Any, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import nn as nn
from torch import optim
from torch.autograd import Function, Variable
from torch.nn.modules.utils import _pair
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
            return nn.Identity()  # type: ignore
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
        self.conv = double_conv(in_ch, out_ch, norm="batch",
                                activation="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mpconv = double_conv(
            in_ch, out_ch, downsample=True, norm="batch", activation="leaky_relu")

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
            in_ch * 2, out_ch, norm="batch", activation="leaky_relu")

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

        self.conv = double_conv(
            out_ch, out_ch, norm="batch", activation="leaky_relu")

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
        beta = np.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)  # type: ignore

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)  # type: ignore
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
        beta = LowerBound.apply(self.beta, self.beta_bound)  # type: ignore
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)  # type: ignore
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


class ConvRNNCellBase(nn.Module):
    def __repr__(self):
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvLSTMCell(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias: bool = True) -> None:
        super(ConvLSTMCell, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_h = tuple(
            k // 2 for k, s, p, d in zip(kernel_size, stride, padding, dilation))
        self.dilation = dilation
        self.groups = groups
        self.weight_ih = nn.Parameter(torch.tensor(  # type: ignore
            4 * out_channels, in_channels // groups, *kernel_size))
        self.weight_hh = nn.Parameter(torch.tensor(  # type: ignore
            4 * out_channels, out_channels // groups, *kernel_size))
        self.weight_ch = nn.Parameter(torch.tensor(  # type: ignore
            3 * out_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias_ih = nn.Parameter(  # type: ignore
                torch.tensor(4 * out_channels))
            self.bias_hh = nn.Parameter(  # type: ignore
                torch.tensor(4 * out_channels))
            self.bias_ch = nn.Parameter(  # type: ignore
                torch.tensor(3 * out_channels))
        else:
            self.register_parameter('bias_ih', None)  # type: ignore
            self.register_parameter('bias_hh', None)  # type: ignore
            self.register_parameter('bias_ch', None)  # type: ignore
        self.register_buffer('wc_blank', torch.zeros(1, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        n = 4 * self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / np.sqrt(n)
        self.weight_ih.data.uniform_(-stdv, stdv)
        self.weight_hh.data.uniform_(-stdv, stdv)
        self.weight_ch.data.uniform_(-stdv, stdv)
        if self.bias_ih is not None:
            self.bias_ih.data.uniform_(-stdv, stdv)
            self.bias_hh.data.uniform_(-stdv, stdv)
            self.bias_ch.data.uniform_(-stdv, stdv)

    def forward(self, x, hx):  # type: ignore
        h_0, c_0 = hx
        wx = F.conv2d(x, self.weight_ih, self.bias_ih,
                      self.stride, self.padding, self.dilation, self.groups)

        wh = F.conv2d(h_0, self.weight_hh, self.bias_hh, self.stride,
                      self.padding_h, self.dilation, self.groups)

        # Cell uses a Hadamard product instead of a convolution?
        wc = F.conv2d(c_0, self.weight_ch, self.bias_ch, self.stride,
                      self.padding_h, self.dilation, self.groups)

        wxhc = wx + wh + torch.cat((wc[:, :2 * self.out_channels], Variable(self.wc_blank).expand(
            wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), wc[:, 2 * self.out_channels:]), 1)

        i = F.sigmoid(wxhc[:, :self.out_channels])
        f = F.sigmoid(wxhc[:, self.out_channels:2 * self.out_channels])
        g = F.tanh(wxhc[:, 2 * self.out_channels:3 * self.out_channels])
        o = F.sigmoid(wxhc[:, 3 * self.out_channels:])

        c_1 = f * c_0 + i * g
        h_1 = o * F.tanh(c_1)
        return h_1, (h_1, c_1)


class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
            + ', max_val=' + str(self.upper) \
            + inplace_str + ')'


class ResBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, in_ch, out_ch,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x
