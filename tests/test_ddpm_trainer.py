import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from trainers.ddpm_trainer import DDPMTrainer
from tests.mocks import DummyModel, DummyDDPM 


# テスト: DDPMTrainer の train_step がパラメータを更新することを確認
def test_train_step_updates_params():
    device = torch.device("cpu")

    model = DummyModel().to(device)
    ddpm = DummyDDPM()

    dataset = TensorDataset(torch.randn(2, 1, 2, 2))
    dataloader = DataLoader(dataset, batch_size=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = DDPMTrainer(
        model=model,
        ddpm=ddpm,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
    )

    batch = next(iter(dataloader))[0]

    # パラメータをコピー
    before = model.linear.weight.clone()

    loss = trainer.train_step(batch)

    after = model.linear.weight

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert not torch.equal(before, after), "パラメータが更新されていない"

# テスト: DDPMTrainer の train_epoch が global_step を正しく更新することを確認
def test_train_epoch_increments_global_step():
    device = torch.device("cpu")

    model = DummyModel().to(device)
    ddpm = DummyDDPM()

    dataset = TensorDataset(torch.randn(5, 1, 2, 2))
    dataloader = DataLoader(dataset, batch_size=1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = DDPMTrainer(
        model=model,
        ddpm=ddpm,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
    )

    trainer.train_epoch()

    assert trainer.global_step == len(dataloader)

# テスト: DDPMTrainer の train が複数エポックを正しく実行することを確認
def test_train_runs_multiple_epochs():
    device = torch.device("cpu")

    model = DummyModel().to(device)
    ddpm = DummyDDPM()

    dataset = TensorDataset(torch.randn(4, 1, 2, 2))
    dataloader = DataLoader(dataset, batch_size=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = DDPMTrainer(
        model=model,
        ddpm=ddpm,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=3,
    )

    trainer.train()

    assert trainer.current_epoch == 2
    assert trainer.global_step == 3 * len(dataloader)

# テスト: DDPMTrainer の train_step が CUDA デバイスで動作することを確認
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_step_cuda():
    device = torch.device("cuda")

    model = DummyModel().to(device)
    ddpm = DummyDDPM()

    dataset = TensorDataset(torch.randn(2, 1, 2, 2))
    dataloader = DataLoader(dataset, batch_size=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = DDPMTrainer(
        model=model,
        ddpm=ddpm,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
    )

    batch = next(iter(dataloader))[0].to(device)
    loss = trainer.train_step(batch)

    assert loss.device.type == "cuda"

