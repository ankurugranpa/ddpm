import torch 


def extract_to_shape(arr: torch.Tensor,
                     t: torch.Tensor,
                     x: torch.Tensor) -> torch.Tensor:

    """
    1次元のスケジューラ配列から t に対応する値を取り出し、
    x とブロードキャスト可能な shape に変形する。

    Args:
        arr (torch.Tensor): (T,)
        t (torch.Tensor): (B,)
        x (torch.Tensor): (B, C, H, W)

    Returns:
        torch.Tensor: (B, 1, 1, 1)
    """

    out = arr[t]
    while out.dim() < x.dim():
        out = out.unsqueeze(-1)
    return out
