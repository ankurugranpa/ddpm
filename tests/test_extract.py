import torch
import pytest

from utils.extract import extract_to_shape


# シェイプの抽出が正しく行われるかのテスト
def test_extract_shape():
    T = 10
    B, C, H, W = 4, 3, 8, 8

    arr = torch.linspace(0, 1, T)        # (T,)
    t = torch.tensor([0, 3, 5, 9])        # (B,)
    x = torch.randn(B, C, H, W)

    out = extract_to_shape(arr, t, x)

    assert out.shape == (B, 1, 1, 1)

# 抽出された値が正しいかのテスト
def test_extract_values():
    arr = torch.tensor([10., 20., 30., 40.])
    t = torch.tensor([3, 0, 1])
    x = torch.randn(3, 2, 4, 4)

    out = extract_to_shape(arr, t, x)

    expected = torch.tensor([40., 10., 20.]).view(3, 1, 1, 1)

    assert torch.allclose(out, expected)


# 抽出された値が broadcast 可能であるかのテスト
def test_extract_broadcastable():
    arr = torch.tensor([1., 2., 3., 4.])
    t = torch.tensor([1, 2])
    x = torch.ones(2, 3, 5, 5)

    out = extract_to_shape(arr, t, x)

    y = out * x  # broadcast 可能であることが重要

    expected = torch.tensor([2., 3.]).view(2, 1, 1, 1)

    assert torch.allclose(y[:, 0, 0, 0], expected.squeeze())

# 抽出されたテンソルが元のテンソルと同じデバイス上にあるかのテスト
def test_extract_device():
    arr = torch.arange(5, dtype=torch.float32)
    t = torch.tensor([1, 3])
    x = torch.randn(2, 3, 4, 4)

    out = extract_to_shape(arr, t, x)

    assert out.device == x.device

# CUDA デバイス上でのテスト
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_extract_cuda():
    device = torch.device("cuda")

    arr = torch.linspace(0, 1, 10, device=device)
    t = torch.tensor([2, 5, 7], device=device)
    x = torch.randn(3, 1, 8, 8, device=device)

    out = extract_to_shape(arr, t, x)

    assert out.device.type == "cuda"
    assert out.shape == (3, 1, 1, 1)
