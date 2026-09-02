.PHONY: setup data test train matrix cuda-build cuda-bench

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e '.[dev]'

data:
	./scripts/download_ett.sh

test:
	pytest -q

train:
	python -m gpuforecast.train --model transformer

matrix:
	python scripts/run_matrix.py

cuda-build:
	cd cuda/fused_ln_gelu && python setup.py build_ext --inplace

cuda-bench:
	cd cuda/fused_ln_gelu && python benchmark.py --dtype fp32
