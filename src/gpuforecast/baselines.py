from __future__ import annotations

import torch


def persistence_forecast(
    x: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Repeat the last observed feature vector across the forecast horizon."""

    if x.ndim != 3:
        raise ValueError(
            "Expected input with shape [batch, time, features]"
        )

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    last = x[:, -1:, :]

    return last.expand(
        -1,
        horizon,
        -1,
    )
