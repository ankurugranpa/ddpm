import torch 
import torch.nn as nn

from diffusion.scheduler import BetaScheduler
from utils.extract import extract_to_shape


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model
    Attribut:
        - model: ノイズ予測モデル
        - scheduler: ベータスケジューラ
    """
    def __init__(self,
                 scheduler: BetaScheduler):

        super().__init__()
        self.scheduler = scheduler


    """
    ノイズをデータに追加する関数
    Args:
        x0 (torch.Tensor): 元のデータ(B, C, H, W)
        t (torch.Tensor): タイムステップ(B,)
        noise (torch.Tensor): 追加するノイズ(B, C, H, W)
    
    Returns:
        torch.Tensor: ノイズが追加されたデータ
    """

    def q_sample(self,
                  x0: torch.Tensor,
                  t: torch.Tensor,
                  noise: torch.Tensor) -> torch.Tensor:

        # 時間の1次元行列から対応するtime_stepの one_minus_alphas_cumprod と sqrt_alpha_bar を取得
        sqrt_alpha_bar = self.scheduler.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod[t]

        # tとx0の次元を合わせる
        sqrt_alpha_bar = extract_to_shape(self.scheduler.sqrt_alphas_cumprod, t, x0) 
        sqrt_one_minus_alphas_cumprod = extract_to_shape(self.scheduler.sqrt_one_minus_alphas_cumprod, t, x0) 



        return sqrt_alpha_bar * x0 + sqrt_one_minus_alphas_cumprod * noise


    def training_loss(self,
                      model: nn.Module,
                      x0: torch.Tensor,
                      t: torch.Tensor | None = None,
                      noise: torch.Tensor | None = None,) -> torch.Tensor:
        """
        トレーニング時の損失計算を行う関数
        Args:
            x0 (torch.Tensor): 元のデータ(B, C, H, W)
            t (torch.Tensor): タイムステップ(B,)
            noise (torch.Tensor): 追加するノイズ(B, C, H, W)
        
        Returns:
            torch.Tensor: MSE損失
        """
        if t is None:
            batch_size = x0.size(0)
            t = torch.randint(0,
                              self.scheduler.time_step,
                              (batch_size,),
                              device=x0.device)

        if noise is None:
            noise = torch.randn_like(x0)



        # ノイズが付与されたデータを生成
        xt = self.q_sample(x0, t, noise)

        # ノイズ予測モデルでノイズを予測
        eps_pred = model(xt, t)

        # MSE損失を計算
        loss = nn.functional.mse_loss(noise, eps_pred)

        return loss


    def reverse(self,
                model: nn.Module,
                xt: torch.Tensor,
                t: torch.Tensor,
                noise: torch.Tensor |  None = None,) -> torch.Tensor:
        """
        逆拡散過程を実行する関数
        Args:
            xt (torch.Tensor): ノイズが付与されたデータ(B, C, H, W)
            t (torch.Tensor): タイムステップ(B,)
            model (nn.Module): ノイズ予測モデル
            noise (torch.Tensor): 追加するノイズ(B, C, H, W)

        Returns:
            torch.Tensor: 1ステップ逆拡散したデータ(B, C, H, W)
        """

        eps_pred = model(xt, t) 
        
        # 次元を合わせる
        sqrt_alphas = extract_to_shape(torch.sqrt(self.scheduler.alphas),
                                       t,
                                       xt)

        sqrt_one_minus_alphas_cumprod = extract_to_shape(self.scheduler.sqrt_one_minus_alphas_cumprod,
                                                         t,
                                                         xt) 

        one_minus_alphas = extract_to_shape(1.0 - self.scheduler.alphas,
                                            t,
                                            xt)

        sqrt_beta = extract_to_shape(torch.sqrt(self.scheduler.betas),
                                     t, 
                                     xt)

        # 逆拡散の計算
        coef1 = 1 / sqrt_alphas 
        coef2 = one_minus_alphas / sqrt_one_minus_alphas_cumprod 

        if noise is None:
            noise = torch.randn_like(xt)

    
        mean = coef1 * (xt - coef2 * eps_pred)
        # tが0のときはノイズを加えないようにするマスク
        nonzero_mask = (t != 0).float().view(-1, *([1] * (xt.dim() - 1)))
        

        return mean + sqrt_beta * noise * nonzero_mask


if __name__ == "__main__": 
    beta_scheduler = BetaScheduler(time_step=1)
    print(beta_scheduler.sqrt_one_minus_alphas_cumprod[0])
    beta_scheduler.sqrt_one_minus_alphas_cumprod[0] = 0.1

    print(beta_scheduler.sqrt_one_minus_alphas_cumprod[0])
