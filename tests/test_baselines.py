import torch

from gpuforecast.baselines import persistence_forecast
from gpuforecast.metrics import regression_metrics


def test_persistence_shape():
    x = torch.randn(4, 96, 7)

    pred = persistence_forecast(
        x,
        horizon=24,
    )

    assert pred.shape == (4, 24, 7)


def test_persistence_repeats_last_observation():
    x = torch.tensor(
        [
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        ]
    )

    pred = persistence_forecast(
        x,
        horizon=2,
    )

    expected = torch.tensor(
        [
            [
                [3.0, 30.0],
                [3.0, 30.0],
            ]
        ]
    )

    torch.testing.assert_close(
        pred,
        expected,
    )


def test_regression_metrics_hand_calculation():
    pred = torch.tensor([1.0, 3.0, 5.0])
    target = torch.tensor([2.0, 3.0, 1.0])

    metrics = regression_metrics(
        pred,
        target,
    )

    # Errors: [-1, 0, 4]
    # Squared errors: [1, 0, 16]
    # MSE = 17 / 3
    # MAE = 5 / 3

    expected_mse = 17.0 / 3.0
    expected_mae = 5.0 / 3.0

    assert abs(metrics["mse"] - expected_mse) < 1e-6
    assert abs(metrics["rmse"] - expected_mse**0.5) < 1e-6
    assert abs(metrics["mae"] - expected_mae) < 1e-6
