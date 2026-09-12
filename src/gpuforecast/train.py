from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gpuforecast.config import load_config
from gpuforecast.data import build_dataloaders
from gpuforecast.gpu import CUDAPrefetcher
from gpuforecast.metrics import regression_metrics
from gpuforecast.models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_environment(device: torch.device) -> dict[str, Any]:
    gpu_name = None

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
        "git_sha": get_git_sha(),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def move_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        x.to(device),
        y.to(device),
    )


def train_epoch(model, loader, optimizer, scaler, loss_fn, device, amp, prefetch_stream, grad_clip,):
    model.train()
    total_loss = 0.0
    total_items = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    iterable = CUDAPrefetcher(loader, device) if (device.type == "cuda" and prefetch_stream) else loader
    for x, y in iterable:
        if not (device.type == "cuda" and prefetch_stream):
            x, y = move_batch(x, y, device)
        optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip,)
            optimizer.step()

        total_loss += float(loss.detach()) * x.size(0)
        total_items += x.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if device.type == "cuda":
        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024**2)
        )

        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(device)
            / (1024**2)
        )
    else:
        peak_allocated_mb = float("nan")
        peak_reserved_mb = float("nan")
    return {
        "loss": total_loss / max(total_items, 1),
        "seconds": elapsed,
        "samples_per_sec": total_items / elapsed,
        "peak_allocated_memory_mb": peak_allocated_mb,
        "peak_reserved_memory_mb": peak_reserved_mb,
    }


@torch.inference_mode()
def evaluate(model, loader, scaler, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x, y = move_batch(x, y, device)
        pred = model(x)
        preds.append(scaler.inverse_torch(pred).float().cpu())
        targets.append(scaler.inverse_torch(y).float().cpu())
    return regression_metrics(torch.cat(preds), torch.cat(targets))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--model", choices=["lstm", "gru", "tcn", "transformer"])
    p.add_argument("--seed", type=int)
    p.add_argument("--amp", choices=["on", "off"])
    p.add_argument("--pin-memory", choices=["on", "off"])
    p.add_argument("--prefetch-stream", choices=["on", "off"])
    p.add_argument("--compile", choices=["on", "off"])
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.amp:
        cfg["train"]["amp"] = args.amp == "on"
    if args.pin_memory:
        cfg["data"]["pin_memory"] = args.pin_memory == "on"
    if args.prefetch_stream:
        cfg["train"]["prefetch_stream"] = args.prefetch_stream == "on"
    if args.compile:
        cfg["train"]["compile"] = args.compile == "on"
    

    seed_everything(int(cfg["seed"]))
    
    device = choose_device(
        cfg["train"].get("device", "auto")
    )

    bundle = build_dataloaders(
        cfg,
        pin_memory=bool(cfg["data"]["pin_memory"]),
    )

    model = build_model(
        cfg,
        bundle.n_features,
    ).to(device)

    parameter_count = count_parameters(model)

    if (
        cfg["train"].get("compile", False)
        and hasattr(torch, "compile")
    ):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(
            cfg["train"]["weight_decay"]
        ),
    )

    loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
            and cfg["train"]["amp"]
        ),
    )

    history = []
    best_rmse = float("inf")
    best_epoch = -1

    run_name = (
        f"{cfg['model']['name']}"
        f"_seed{cfg['seed']}"
        f"_amp{int(cfg['train']['amp'])}"
        f"_pin{int(cfg['data']['pin_memory'])}"
        f"_prefetch{int(cfg['train']['prefetch_stream'])}"
        f"_compile{int(cfg['train'].get('compile', False))}"
    )

    root_dir = Path(cfg["output"]["dir"])
    run_dir = root_dir / run_name

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = run_dir / "best.pt"
    config_path = run_dir / "config.json"
    environment_path = run_dir / "environment.json"
    status_path = run_dir / "status.json"
    result_path = run_dir / "result.json"

    environment = collect_environment(device)

    write_json(
        config_path,
        cfg,
    )

    write_json(
        environment_path,
        environment,
    )

    write_json(
        status_path,
        {
            "run": run_name,
            "status": "running",
        },
    )


    try:
        for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
            tr = train_epoch(
                model,
                bundle.train,
                optimizer,
                scaler,
                loss_fn,
                device,
                bool(cfg["train"]["amp"]),
                bool(cfg["train"]["prefetch_stream"]),
                float(cfg["train"]["grad_clip"]),
            )

            val = evaluate(
                model,
                bundle.val,
                bundle.scaler,
                device,
            )

            row = {
                "epoch": epoch,
                **{
                    f"train_{k}": v
                    for k, v in tr.items()
                },
                **{
                    f"val_{k}": v
                    for k, v in val.items()
                },
            }

            history.append(row)
            print(json.dumps(row))

            if val["rmse"] < best_rmse:
                best_rmse = val["rmse"]
                best_epoch = epoch

                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "epoch": epoch,
                        "val_rmse": val["rmse"],
                        "run": run_name,
                        "environment": environment,
                    },
                    best_path,
                )

        checkpoint = torch.load(
            best_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        test = evaluate(
            model,
            bundle.test,
            bundle.scaler,
            device,
        )

        summary = {
            "run": run_name,
            "status": "completed",
            "config_source": args.config,
            "config": cfg,
            "environment": environment,
            "parameters": parameter_count,
            "checkpoint": str(best_path),
            "best_val_rmse": best_rmse,
            "best_epoch": best_epoch,
            "test": test,
            "history": history,
        }

        write_json(
            result_path,
            summary,
        )

        write_json(
            status_path,
            {
                "run": run_name,
                "status": "completed",
                "best_epoch": best_epoch,
                "best_val_rmse": best_rmse,
            },
        )

        print(
            json.dumps(
                {
                    "final": summary["run"],
                    "test": test,
                },
                indent=2,
            )
        )

    except BaseException as exc:
        write_json(
            status_path,
            {
                "run": run_name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_epochs": len(history),
            },
        )

        raise


if __name__ == "__main__":
    main()
