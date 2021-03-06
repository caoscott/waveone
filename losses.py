from math import exp
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    offset = 0.5 if window_size % 2 == 0 else 0
    gauss = torch.tensor(
        [exp(-(x + offset - window_size//2)**2/(2.0*sigma**2))
         for x in range(window_size)]
    )
    return gauss/gauss.sum()


def create_window(window_size: int, channel: int = 1) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(
        channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1: torch.Tensor,
         img2: torch.Tensor,
         window_size: int = 11,
         window: torch.Tensor = None,
         size_average: bool = True,
         full: bool = False,
         val_range: Union[int, float] = -1,
         ) -> Tuple[torch.Tensor, torch.Tensor]:
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    L: Union[int, float] = 0
    if val_range < 0:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd,
                         groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd,
                       groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret, cs


def msssim(img1: torch.Tensor,
           img2: torch.Tensor,
           window_size: int = 11,
           size_average: bool = True,
           val_range: Union[int, float] = -1,
           normalize: bool = False,
           ) -> torch.Tensor:
    device = img1.device
    weights = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim_list = []
    mcs_list = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size,
                       size_average=size_average, full=True, val_range=val_range)
        mssim_list.append(sim)
        mcs_list.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim_list)
    mcs = torch.stack(mcs_list)

    # Normalize (to avoid NaNs during training unstable models,
    # not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2  # type: ignore
        mcs = (mcs + 1) / 2  # type: ignore

    pow1 = mcs ** weights  # type: ignore
    pow2 = mssim ** weights  # type: ignore
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self,
                 window_size: int = 11,
                 size_average: bool = True,
                 val_range: Union[int, float] = -1,
                 ) -> None:
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self,   # type: ignore
                img1: torch.Tensor,
                img2: torch.Tensor
                ) -> torch.Tensor:
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(  # type: ignore
                img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        score, _ = ssim(img1, img2, window=window,
                        window_size=self.window_size,
                        size_average=self.size_average)
        return score


class MSSSIM(torch.nn.Module):
    def __init__(self,
                 window_size: int = 11,
                 size_average: bool = True,
                 channel: int = 3,
                 val_range: Union[int, float] = -1,
                 normalize: bool = False,
                 negative: bool = False,
                 ) -> None:
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.val_range = val_range
        self.normalize = normalize
        self.negative = negative

    def forward(self,   # type: ignore
                img1: torch.Tensor,
                img2: torch.Tensor,
                ) -> torch.Tensor:
        assert img1.shape == img2.shape
        if len(img1.shape) > 4:
            new_shape = (-1, img1.shape[-3], img1.shape[-2], img1.shape[-1])
            img1 = img1.reshape(*new_shape)
            img2 = img2.reshape(*new_shape)
        # TODO: store window between calls if possible
        score = msssim(
            img1, img2, window_size=self.window_size,
            size_average=self.size_average,
            val_range=self.val_range, normalize=self.normalize
        )
        return score * (-1 if self.negative else 1)


class CharbonnierLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.eps_squared = 1e-8

    def forward(self,  # type: ignore
                img1: torch.Tensor,
                img2: torch.Tensor
                ) -> torch.Tensor:
        return ((self.mse(img1, img2) + self.eps_squared) ** 0.5) / img1.shape[0]


# class PSNR(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse_loss = nn.MSELoss(reduction='mean')

#     def forward(self, img1, img2):
#         mse = self.mse_loss(img1, img2)
#         return torch.clamp(
#             torch.mul(torch.log10(255. * 2), 10.), 0., 99.99)[0]

class TotalVariation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return (torch.sum(torch.abs(x[:, :-1, :, :] - x[:, 1:, :, :])) +
                torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
