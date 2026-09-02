from __future__ import annotations

import math

import torch
from torch import nn


class RNNForecaster(nn.Module):
    def __init__(self, kind: str, n_features: int, horizon: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.horizon = horizon
        self.n_features = n_features
        self.head = nn.Linear(hidden, horizon * n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        y = self.head(out[:, -1])
        return y.view(x.shape[0], self.horizon, self.n_features)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :-self.pad] if self.pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.drop(self.act(self._crop(self.conv1(x))))
        y = self.drop(self.act(self._crop(self.conv2(y))))
        return self.act(y + residual)


class TCNForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, channels: list[int], dropout: float):
        super().__init__()
        blocks = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            blocks.append(TCNBlock(in_ch, out_ch, kernel=3, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels[-1], horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2))[:, :, -1]
        y = self.head(y)
        return y.view(x.shape[0], self.horizon, self.n_features)


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        hidden: int,
        layers: int,
        nhead: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.pos = SinusoidalPosition(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=4 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers, norm=nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, horizon * n_features)
        self.horizon = horizon
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.pos(self.input_proj(x)))
        y = self.head(z[:, -1])
        return y.view(x.shape[0], self.horizon, self.n_features)


def build_model(cfg: dict, n_features: int) -> nn.Module:
    mcfg = cfg["model"]
    name = mcfg["name"].lower()
    horizon = int(cfg["data"]["horizon"])
    if name in {"lstm", "gru"}:
        return RNNForecaster(
            name,
            n_features,
            horizon,
            int(mcfg["hidden_size"]),
            int(mcfg["num_layers"]),
            float(mcfg["dropout"]),
        )
    if name == "tcn":
        return TCNForecaster(
            n_features,
            horizon,
            list(mcfg["tcn_channels"]),
            float(mcfg["dropout"]),
        )
    if name == "transformer":
        return TransformerForecaster(
            n_features,
            horizon,
            int(mcfg["hidden_size"]),
            int(mcfg["num_layers"]),
            int(mcfg["nhead"]),
            float(mcfg["dropout"]),
        )
    raise ValueError(f"Unknown model: {name}")
