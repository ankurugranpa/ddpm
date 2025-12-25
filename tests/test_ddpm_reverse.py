import torch


# t=0 でノイズが加わらないか のテスト
def test_reverse_t_zero_no_noise(ddpm):
    B, C, H, W = 2, 3, 8, 8
    xt = torch.randn(B, C, H, W)
    eps_pred = torch.randn_like(xt)

    t = torch.zeros(B, dtype=torch.long)

    noise = torch.randn_like(xt)
    out = ddpm.reverse(xt, t, eps_pred, noise=noise)

    # t=0 なので noise が加算されていない
    mean_only = ddpm.reverse(xt, t, eps_pred, noise=torch.zeros_like(xt))

    assert torch.allclose(out, mean_only)

# 出力の shape が入力と同じであることのテスト
def test_reverse_shape(ddpm):
    B, C, H, W = 4, 1, 16, 16
    xt = torch.randn(B, C, H, W)
    eps_pred = torch.randn_like(xt)
    t = torch.randint(0, ddpm.scheduler.time_step, (B,))

    xt_prev = ddpm.reverse(xt, t, eps_pred)

    assert xt_prev.shape == xt.shape

