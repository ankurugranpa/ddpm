import pytest
import torch

from diffusion.scheduler import BetaScheduler



def test_beta_scheduler_creates_correct_betas():
    """
    【テスト目的】
    BetaScheduler が指定した time_step, beta_start, beta_end に基づいて
    正しく betas を線形生成できているかを確認するテスト。

    【検証内容】
    - betas の shape が (time_step,) である
    - 最初と最後の値が beta_start, beta_end と一致する
    """
    scheduler = BetaScheduler(
        time_step=10,
        beta_start=0.1,
        beta_end=0.2,
    )

    betas = scheduler.betas

    # betas の長さが time_step と一致すること
    assert betas.shape == (10,)

    # 最初と最後の値が指定値と一致すること
    assert torch.isclose(betas[0], torch.tensor(0.1))
    assert torch.isclose(betas[-1], torch.tensor(0.2))


def test_beta_scheduler_dtype_is_float32():
    """
    【テスト目的】
    betas の dtype が torch.float32 であることを確認するテスト。

    【理由】
    - diffusion の計算は float32 を前提にすることが多い
    - dtype の暗黙変換によるバグを防ぐため
    """
    scheduler = BetaScheduler(time_step=5)

    assert scheduler.betas.dtype == torch.float32


def test_alpha_and_cumprod_are_computed_correctly():
    """
    【テスト目的】
    alphas, alphas_cumprod が betas から正しく計算されているかを確認するテスト。

    【数式】
    alphas_t = 1 - betas_t
    alphas_cumprod_t = ∏_{i=1}^t alphas_i
    """
    scheduler = BetaScheduler(
        time_step=3,
        beta_start=0.1,
        beta_end=0.1,  # 固定値にして検証しやすくする
    )

    betas = scheduler.betas
    alphas = scheduler.alphas
    alphas_cumprod = scheduler.alphas_cumprod

    # alphas = 1 - betas になっているか
    assert torch.allclose(alphas, 1.0 - betas)

    # cumprod が手計算と一致するか
    expected = torch.tensor([
        alphas[0],
        alphas[0] * alphas[1],
        alphas[0] * alphas[1] * alphas[2],
    ])

    assert torch.allclose(alphas_cumprod, expected)


def test_buffers_are_registered():
    """
    【テスト目的】
    betas, alphas, alphas_cumprod などが
    nn.Module の buffer として登録されているかを確認するテスト。

    【理由】
    - buffer であれば state_dict に含まれる
    - .to(device) に追従する
    """
    scheduler = BetaScheduler(time_step=5)

    buffer_names = dict(scheduler.named_buffers()).keys()

    assert "betas" in buffer_names
    assert "alphas" in buffer_names
    assert "alphas_cumprod" in buffer_names
    assert "sqrt_alphas_cumprod" in buffer_names


def test_invalid_schedule_type_raises_error():
    """
    【テスト目的】
    未対応の schedule_type を指定した場合に
    ValueError が発生することを確認する異常系テスト。
    """
    with pytest.raises(ValueError):
        BetaScheduler(
            time_step=10,
            schedule_type="unknown",
        )

