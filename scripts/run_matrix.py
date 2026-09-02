from __future__ import annotations

import itertools
import subprocess
import sys

MODELS = ["lstm", "gru", "tcn", "transformer"]
SEEDS = [17, 42, 101]
SYSTEMS = [
    # name, AMP, pin_memory, prefetch_stream, compile
    ("fp32", "off", "off", "off", "off"),
    ("amp", "on", "off", "off", "off"),
    ("amp_pin", "on", "on", "off", "off"),
    ("amp_pin_prefetch", "on", "on", "on", "off"),
    ("amp_pin_prefetch_compile", "on", "on", "on", "on"),
]


def main() -> None:
    runs = list(itertools.product(MODELS, SEEDS, SYSTEMS))
    print(f"Planned controlled runs: {len(runs)}")
    for i, (model, seed, system) in enumerate(runs, 1):
        name, amp, pin, prefetch, compile_ = system
        print(f"[{i:02d}/{len(runs)}] model={model} seed={seed} system={name}")
        cmd = [
            sys.executable,
            "-m",
            "gpuforecast.train",
            "--model",
            model,
            "--seed",
            str(seed),
            "--amp",
            amp,
            "--pin-memory",
            pin,
            "--prefetch-stream",
            prefetch,
            "--compile",
            compile_,
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
