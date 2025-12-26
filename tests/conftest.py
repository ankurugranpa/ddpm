import torch
import torch.nn as nn
import pytest

from diffusion.ddpm import DDPM
from diffusion.scheduler import BetaScheduler


class DummyModel(nn.Module):
    """
    ε を常に 0 と予測するダミーモデル
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
    # model を渡さない
    return DDPM(scheduler=scheduler)


@pytest.fixture
def dummy_model():
    return DummyModel()
