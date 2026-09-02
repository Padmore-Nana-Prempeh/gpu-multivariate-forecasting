#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
curl -L "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv" -o data/ETTm1.csv
echo "Saved data/ETTm1.csv"
