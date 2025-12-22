from typing import TypeAlias
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


# 基本型
Tensor: TypeAlias = torch.Tensor # データ型
Device: TypeAlias = torch.device # デバイス型




"""
ノイズが付与されたデータ TensorType["B", "C", "H", "W"]
B: バッチサイズ
C: チャンネル数
H: 画像の高さ
W: 画像の幅

バッチサイズは一度に扱うでーたの数を表す
ex) B=1の場合
x[0]: 1枚目の画像データ[C, H, W]
x[1]: 2枚目の画像データ[C, H, W]
"""
Noises: TypeAlias = TensorType["B", "C", "H", "W"]
