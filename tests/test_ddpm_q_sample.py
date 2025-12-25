import torch

# noise=0 → √ᾱ x₀ になるかのテスト

def test_q_sample_zero_noise(ddpm):
    B, C, H, W = 2, 3, 8, 8
    x0 = torch.randn(B, C, H, W)
    noise = torch.zeros_like(x0)
    t = torch.tensor([0, 5])

    xt = ddpm.q_sample(x0, t, noise)

    sqrt_alpha_bar = ddpm.scheduler.sqrt_alphas_cumprod[t]
    while sqrt_alpha_bar.dim() < x0.dim():
        sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)

    expected = sqrt_alpha_bar * x0
    assert torch.allclose(xt, expected)

