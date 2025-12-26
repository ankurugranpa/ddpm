import torch


# t=0 でノイズが加わらないか のテスト
import torch

def test_reverse_t_zero_no_noise(ddpm, dummy_model):
    B, C, H, W = 2, 3, 8, 8
    xt = torch.randn(B, C, H, W)

    t = torch.zeros(B, dtype=torch.long)

    noise = torch.randn_like(xt)

    out = ddpm.reverse(
        dummy_model,
        xt,
        t,
        noise=noise,
    )

    mean_only = ddpm.reverse(
        dummy_model,
        xt,
        t,
        noise=torch.zeros_like(xt),
    )

    # t=0 なので noise 項は無視される
    assert torch.allclose(out, mean_only)

def test_reverse_shape(ddpm, dummy_model):
    B, C, H, W = 4, 1, 16, 16
    xt = torch.randn(B, C, H, W)
    t = torch.randint(0, ddpm.scheduler.time_step, (B,))

    xt_prev = ddpm.reverse(dummy_model, xt, t)

    assert xt_prev.shape == xt.shape

