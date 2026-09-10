from __future__ import annotations

import json
from pathlib import Path

import torch

from gpuforecast.baselines import persistence_forecast
from gpuforecast.config import load_config
from gpuforecast.data import build_dataloaders
from gpuforecast.metrics import regression_metrics


def main() -> None:
    cfg = load_config("configs/base.yaml")

    cfg["data"]["num_workers"] = 0
    cfg["data"]["pin_memory"] = False

    bundle = build_dataloaders(
        cfg,
        pin_memory=False,
    )

    preds = []
    targets = []

    for x, y in bundle.test:
        pred = persistence_forecast(
            x,
            horizon=int(cfg["data"]["horizon"]),
        )

        preds.append(
            bundle.scaler.inverse_torch(pred)
        )

        targets.append(
            bundle.scaler.inverse_torch(y)
        )

    pred = torch.cat(preds)
    target = torch.cat(targets)

    metrics = regression_metrics(
        pred,
        target,
    )

    result = {
        "run": "persistence_baseline",
        "model": "persistence",
        "parameters": 0,
        "test_windows": len(bundle.test.dataset),
        "prediction_shape": list(pred.shape),
        "test": metrics,
    }

    out_dir = Path("results/persistence_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "persistence.json"

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

