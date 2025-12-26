import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x, t=None):
        # x: [B, C, H, W] → flatten
        b = x.size(0)
        return self.linear(x.view(b, -1)).view_as(x)

# tests/mocks.py
class DummyDDPM:
    def training_loss(self, model, x0):
        # 常に微分可能な loss を返す
        pred = model(x0)
        return (pred ** 2).mean()

