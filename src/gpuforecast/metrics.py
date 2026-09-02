from __future__ import annotations

import torch


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    err = pred - target
    mse = torch.mean(err.square()).item()
    mae = torch.mean(err.abs()).item()
    return {"mse": mse, "rmse": mse**0.5, "mae": mae}
