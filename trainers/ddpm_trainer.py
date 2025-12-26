import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainers.base_trainer import BaseTrainer
from diffusion.ddpm import DDPM



class DDPMTrainer(BaseTrainer):
    """
    DDPM 用 Trainer
        Args:
            model: ノイズ予測モデル (UNet など)
            diffusion: DDPM インスタンス
            dataloader: 学習用 DataLoader
            optimizer: Optimizer
            device: 使用デバイス
            num_epochs: 学習エポック数
            log_interval: ログ出力間隔
    """

    def __init__(
            self,
            model: torch.nn.Module,
            ddpm : DDPM,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            num_epochs: int,
            log_interval: int = 100):

        self.model = model.to(device)
        self.ddpm = ddpm
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.log_interval = log_interval

        self.global_step = 0
        self.current_epoch = 0

    # def train_step(self, batch: torch.Tensor) -> torch.Tensor:
    #     """
    #     1 step の学習処理

    #     Args:
    #         batch (Tensor): 学習時の元画像 [B, C, H, W]

    #     Returns:
    #         loss (Tensor)
    #     """
    #     x0 = batch.to(self.device)

    #     self.optimizer.zero_grad() # 勾配初期化
    #     loss = self.ddpm.training_loss(self.model, x0)
    #     loss.backward() # 勾配計算
    #     self.optimizer.step() # パラメータ更新

    #     return loss.detach()
    """
    1 step の学習処理
    """
    def train_step(self, batch: torch.Tensor) -> torch.Tensor:
        # batch がリストやタプルの場合、最初の要素を使用
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        x0 = batch.to(self.device) # 元画像をデバイスに転送

        self.optimizer.zero_grad() # 勾配初期化
        loss = self.ddpm.training_loss(self.model, x0)
        loss.backward()
        self.optimizer.step()

        return loss.detach()


    def train_epoch(self):
        self.model.train()
        pbar = tqdm(self.dataloader)

        for batch in pbar:
            loss = self.train_step(batch)

            if self.global_step % self.log_interval == 0:
                pbar.set_postfix(
                    epoch=self.current_epoch,
                    step=self.global_step,
                    loss=f"{loss.item():.6f}",
                )

            self.global_step += 1



    def train(self):
        """
        学習全体のループ
        """
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()

