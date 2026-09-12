import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gpuforecast.train import (
    collect_environment,
    count_parameters,
    seed_everything,
    train_epoch,
)


def test_seed_everything_reproducible():
    seed_everything(42)
    first = torch.randn(5)

    seed_everything(42)
    second = torch.randn(5)

    torch.testing.assert_close(
        first,
        second,
    )


def test_count_parameters():
    model = nn.Linear(3, 2)

    assert count_parameters(model) == 8


def test_environment_metadata_contains_required_fields():
    metadata = collect_environment(
        torch.device("cpu")
    )

    required = {
        "python_version",
        "platform",
        "machine",
        "torch_version",
        "cuda_version",
        "device",
        "gpu_name",
        "git_sha",
    }

    assert required.issubset(
        metadata.keys()
    )


def test_train_epoch_uses_configured_grad_clip(
    monkeypatch,
):
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=4,
    )

    model = nn.Linear(2, 1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )

    observed = []

    original = (
        torch.nn.utils.clip_grad_norm_
    )

    def capture_clip(
        parameters,
        max_norm,
        *args,
        **kwargs,
    ):
        observed.append(max_norm)

        return original(
            parameters,
            max_norm,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        capture_clip,
    )

    train_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scaler=scaler,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        amp=False,
        prefetch_stream=False,
        grad_clip=0.25,
    )

    assert observed
    assert all(
        value == 0.25
        for value in observed
    )
