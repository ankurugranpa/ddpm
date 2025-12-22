from typing import TypeAlias
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


# 基本型
Tensor: TypeAlias = torch.Tensor # データ型
Device: TypeAlias = torch.device # デバイス型


# 1バッチ当たりのデータ TensorType["B", "C", "H", "W"]
BatchData: TypeAlias  = TensorType["B", "C", "H", "W"]


# ノイズのデータ TensorType["B", "C", "H", "W"]
Noise: TypeAlias = TensorType["B", "C", "H", "W"]
