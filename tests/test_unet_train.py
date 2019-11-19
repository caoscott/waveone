import pytest
import torch
from torch import nn

from waveone.losses import MSSSIM
from waveone.network_parts import LambdaModule
from waveone.unet_train import forward_model


def test_forward_model_exact_residual():
    shape = (32, 3, 64, 64)
    frame1 = torch.rand(shape) - 0.5
    frame2 = torch.rand(shape) - 0.5
    network = LambdaModule(lambda x: x[:, 3:] - x[:, :3])
    _, reconstructed_frame2 = forward_model(network, frame1, frame2)
    msssim_score = MSSSIM(val_range=1)(frame2, reconstructed_frame2).item()
    assert msssim_score == pytest.approx(1.0)
    l2_score = nn.MSELoss()(frame2, reconstructed_frame2).item()
    assert l2_score == pytest.approx(0.)


def test_forward_model_zero_residual():
    shape = (24, 3, 255, 255)
    frame = torch.rand(shape) - 0.5
    network = LambdaModule(lambda x: x[:, 3:] - x[:, :3])
    residuals, reconstructed = forward_model(network, frame, frame)
    assert residuals.norm().item() == pytest.approx(0.)
    l2_score = nn.MSELoss()(reconstructed, frame).item()
    assert l2_score == pytest.approx(0.)
