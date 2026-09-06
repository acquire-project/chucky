#!/usr/bin/env bash
set -euo pipefail
run_root=/mnt/main0/home/nclack/tmp/2026-09-05-pr264-l40-full
repo="$run_root/repo"
deps=/mnt/main0/home/nclack/opt/chucky-deps/.pixi/envs/default
: "${SLURM_JOB_ID:?Run this check through Slurm}"
export PATH="$HOME/.pixi/bin:$HOME/.pixi/envs/cuda-toolkit/bin:$PATH"
export LD_LIBRARY_PATH="$deps/lib:$HOME/opt/lib"
export CHUCKY_MACHINE=reef-l40
export TMPDIR="$run_root/gpu-tmp"
mkdir -p "$TMPDIR"
exec > >(tee "$run_root/gpu.log") 2>&1
python3 "$run_root/sweep.py" check-build
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
python3 "$run_root/cli_check.py" "$repo/build/bench/bench_stream_smallepoch_single"
ctest --test-dir "$repo/build" --output-on-failure --timeout 240 \
  -R '^test-test_(compress_blosc_gpu|multiarray_gpu|zarr_readback_gpu)$'
python3 "$run_root/sweep.py" run
