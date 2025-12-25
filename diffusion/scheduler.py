import torch
import torch.nn as nn


class BetaScheduler(nn.Module):
    """
    diffusionのパラメーターのスケジューラー

    Attributes:
        time_step (int): タイムスケジュール数(何ステップでノイズを付与するか)
        schedule_type (str): betaスケジューラーのタイプ
        beta_start (float): betaの開始値
        beta_end (float): betaの終了値
    """
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor

    def __init__(self,
                 time_step: int,
                 schedule_type: str = "normal",
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02) -> None:
        """
        BetaSchedulerの初期化

        Args:
            time_step (int): タイムスケジュール数
            schedule_type (str): スケジューラ―タイプ(normalのみ)

        """

        # 初期化処理
        super().__init__()
        self.time_step = time_step # タイムスケジュール数
        self.schedule_type = schedule_type.lower()
        self._beta_start = beta_start # betaの開始値
        self._beta_end = beta_end # betaの終了値


        if self.schedule_type == "normal":
            betas = torch.linspace(self._beta_start,
                                   self._beta_end,
                                   self.time_step,
                                   dtype=torch.float32)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


        # alphas, alphas_cumprodの計算
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)


        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",
                             torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))

if __name__ == "__main__":
    scheduler = BetaScheduler(time_step=10)
    print("betas:", scheduler.betas)
