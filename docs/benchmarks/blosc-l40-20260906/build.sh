#!/usr/bin/env bash
set -euo pipefail
run_root=/mnt/main0/home/nclack/tmp/2026-09-05-pr264-l40-full
repo="$run_root/repo"
cuda=/mnt/main0/home/nclack/.pixi/envs/cuda-toolkit
deps=/mnt/main0/home/nclack/opt/chucky-deps/.pixi/envs/default
nvcomp=/mnt/main0/home/nclack/opt/nvcomp-5.3.0.16
: "${SLURM_JOB_ID:?Run this build through Slurm}"
export PATH="$HOME/.pixi/bin:$cuda/bin:$PATH"
export CUDAToolkit_ROOT="$cuda"
export PKG_CONFIG_PATH="$HOME/opt/lib/pkgconfig"
export LD_LIBRARY_PATH="$deps/lib:$HOME/opt/lib"
export TMPDIR="$run_root/build-tmp"
mkdir -p "$TMPDIR"
exec > >(tee "$run_root/build.log") 2>&1
python3 "$run_root/sweep.py" check-source
cd "$repo"
cmake --preset default -DCHUCKY_ENABLE_GPU=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$deps;$nvcomp;$cuda" \
  -DNVCOMP_INCLUDE_DIR="$nvcomp/include" \
  -DNVCOMP_LIBRARY="$nvcomp/lib/libnvcomp_static.a"
cmake --build build -j "${SLURM_CPUS_PER_TASK:-16}" --target \
  bench_stream_orca2_single bench_stream_smallepoch_single \
  test_compress_cpu test_compress_blosc_gpu test_multiarray_gpu \
  test_zarr_readback_gpu test_bench_memory
ctest --test-dir build --output-on-failure -R '^test-test_(compress_cpu|bench_memory)$'
python3 "$run_root/sweep.py" record-build
