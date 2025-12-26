from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Trainer 抽象基底クラス

    学習ループの「契約」だけを定義する。
    実装は各アルゴリズム固有の Trainer に任せる。
    """

    @abstractmethod
    def train_step(self, batch):
        """
        1 step 分の学習処理

        Args:
            batch: DataLoader から取得した 1 バッチ
        """
        raise NotImplementedError

    @abstractmethod
    def train_epoch(self):
        """
        1 epoch 分の学習処理
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """
        学習全体のループ
        """
        raise NotImplementedError

