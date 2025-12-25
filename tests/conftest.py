# tests/conftest.py
import torch
import torch.nn as nn
import pytest

from diffusion.ddpm import DDPM
from diffusion.scheduler import BetaScheduler


class DummyModel(nn.Module):
    """
    ε をそのまま返すだけのダミーモデル
    """
    def forward(self, x, t):
        return torch.zeros_like(x)


@pytest.fixture
def scheduler():
    return BetaScheduler(
        time_step=10,
        beta_start=0.0001,
        beta_end=0.02,
    )


@pytest.fixture
def ddpm(scheduler):
    model = DummyModel()
    return DDPM(model=model, scheduler=scheduler)
