from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class StandardScaler:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_torch(self, x: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        mean = torch.as_tensor(self.mean.squeeze(0), device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std.squeeze(0), device=x.device, dtype=x.dtype)
        return x * std + mean


class WindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        seq_len: int,
        horizon: int,
        label_start: int,
        label_end: int,
    ) -> None:
        self.values = values
        self.seq_len = seq_len
        self.horizon = horizon
        first_input = label_start - seq_len
        last_input = label_end - seq_len - horizon
        if first_input < 0 or last_input < first_input:
            raise ValueError("Split is too short for seq_len + horizon")
        self.starts = np.arange(first_input, last_input + 1, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        x = self.values[start : start + self.seq_len]
        y_start = start + self.seq_len
        y = self.values[y_start : y_start + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    scaler: StandardScaler
    n_features: int


def load_numeric_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError(f"No numeric columns found in {path}")
    return numeric.to_numpy(dtype=np.float32), numeric.columns.tolist()


def build_dataloaders(cfg: dict, *, pin_memory: bool | None = None) -> DataBundle:
    dcfg = cfg["data"]
    raw, _ = load_numeric_csv(dcfg["path"])
    n = len(raw)
    train_size = int(n * dcfg["train_ratio"])
    val_size = int(n * dcfg["val_ratio"])

    train_end = train_size
    val_end = train_end + val_size

    scaler = StandardScaler().fit(raw[:train_end])
    values = scaler.transform(raw)

    seq_len = int(dcfg["seq_len"])
    horizon = int(dcfg["horizon"])
    train_ds = WindowDataset(values, seq_len, horizon, seq_len, train_end)
    val_ds = WindowDataset(values, seq_len, horizon, train_end, val_end)
    test_ds = WindowDataset(values, seq_len, horizon, val_end, n)

    use_pin = bool(dcfg.get("pin_memory", True) if pin_memory is None else pin_memory)
    common = {
        "batch_size": int(dcfg["batch_size"]),
        "num_workers": int(dcfg.get("num_workers", 0)),
        "pin_memory": use_pin,
        "persistent_workers": int(dcfg.get("num_workers", 0)) > 0,
    }
    if int(dcfg.get("num_workers", 0)) > 0:
        common["prefetch_factor"] = 2

    return DataBundle(
        train=DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        val=DataLoader(val_ds, shuffle=False, drop_last=False, **common),
        test=DataLoader(test_ds, shuffle=False, drop_last=False, **common),
        scaler=scaler,
        n_features=raw.shape[1],
    )
