# tests/test_ddpm_training_loss.py
import torch


# loss がスカラーで返るかつ非負であることのテスト
def test_training_loss_scalar(ddpm, dummy_model):
    x0 = torch.randn(4, 3, 16, 16)

    loss = ddpm.training_loss(dummy_model, x0)

    assert loss.dim() == 0
    assert loss.item() >= 0


# 同じ入力に対して同じ損失が返ることのテスト
def test_training_loss_deterministic(ddpm, dummy_model):
    torch.manual_seed(0)

    x0 = torch.randn(2, 3, 8, 8)
    t = torch.tensor([3, 7])
    noise = torch.randn_like(x0)

    loss1 = ddpm.training_loss(dummy_model, x0, t=t, noise=noise)
    loss2 = ddpm.training_loss(dummy_model, x0, t=t, noise=noise)

    assert torch.allclose(loss1, loss2)

