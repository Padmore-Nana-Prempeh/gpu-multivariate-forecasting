import torch

from gpuforecast.models import SinusoidalPosition, TransformerForecaster


def test_sinusoidal_positions_differ():
    pos = SinusoidalPosition(dim=8, max_len=10)

    x = torch.zeros(1, 5, 8)
    z = pos(x)

    assert z.shape == x.shape
    assert not torch.equal(z[:, 0], z[:, 1])

    assert sum(
        p.numel()
        for p in pos.parameters()
    ) == 0


def test_parameter_matched_transformer():
    model = TransformerForecaster(
        n_features=7,
        horizon=24,
        hidden=92,
        layers=2,
        nhead=4,
        dropout=0.10,
    )

    x = torch.randn(4, 96, 7)
    y = model(x)

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    assert y.shape == (4, 24, 7)
    assert trainable == 222072
