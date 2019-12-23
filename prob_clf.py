"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch
from torch import nn as nn

from logistic_mixture import non_shared_get_Kp


def conv(in_channels, out_channels, kernel_size, bias=True, rate=1, stride=1):
    padding = kernel_size // 2 if rate == 1 else rate
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, dilation=rate,
        padding=padding, bias=bias)


class AtrousProbabilityClassifier(nn.Module):
    def __init__(self, K=5, C=3, Cf=64, kernel_size=3, atrous_rates_str='1,2,4'):
        super(AtrousProbabilityClassifier, self).__init__()
        Kp = non_shared_get_Kp(K, C)

        self.atrous = StackedAtrousConvs(atrous_rates_str, Cf, Kp,
                                         kernel_size=kernel_size)
        self._repr = f'C={C}; K={K}; Kp={Kp}; rates={atrous_rates_str}'

    def __repr__(self):
        return f'AtrousProbabilityClassifier({self._repr})'

    def forward(self, x):
        """
        :param x: NCfHW
        :return: NKpHW
        """
        return self.atrous(x)


class StackedAtrousConvs(nn.Module):
    def __init__(self, atrous_rates_str, Cin, Cout, bias=True, kernel_size=3):
        super(StackedAtrousConvs, self).__init__()
        atrous_rates = self._parse_atrous_rates_str(atrous_rates_str)
        self.atrous = nn.ModuleList(
            [conv(Cin, Cin, kernel_size, rate=rate) for rate in atrous_rates])
        self.lin = conv(len(atrous_rates) * Cin, Cout, 1, bias=bias)
        self._extra_repr = 'rates={}'.format(atrous_rates)

    @staticmethod
    def _parse_atrous_rates_str(atrous_rates_str):
        # expected to either be an int or a comma-separated string 1,2,4
        if isinstance(atrous_rates_str, int):
            return [atrous_rates_str]
        else:
            return list(map(int, atrous_rates_str.split(',')))

    def extra_repr(self):
        return self._extra_repr

    def forward(self, x):
        x = torch.cat([atrous(x) for atrous in self.atrous], dim=1)
        x = self.lin(x)
        return x
