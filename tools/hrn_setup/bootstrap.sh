#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
TASK_RUNTIME="$PWD/.runtime/hrn"
mkdir -p "$TASK_RUNTIME/bin"
export MAMBA_ROOT_PREFIX="$TASK_RUNTIME/mamba"
if [ ! -x "$TASK_RUNTIME/bin/micromamba" ]; then
  curl --fail --location --retry 2 https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$TASK_RUNTIME/micromamba.tar.bz2"
  tar -xjf "$TASK_RUNTIME/micromamba.tar.bz2" -C "$TASK_RUNTIME" bin/micromamba
fi
if [ ! -x "$TASK_RUNTIME/env/bin/python" ]; then
  "$TASK_RUNTIME/bin/micromamba" create -y -p "$TASK_RUNTIME/env" -c nvidia/label/cuda-11.8.0 -c conda-forge python=3.10 pip cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-nvrtc-dev=11.8
fi
"$TASK_RUNTIME/bin/micromamba" install -y -p "$TASK_RUNTIME/env" -c nvidia/label/cuda-11.8.0 -c conda-forge libcusparse-dev libcublas-dev libcusolver-dev cuda-driver-dev
if [ ! -e "$TASK_RUNTIME/env/lib64" ]; then
  ln -s lib "$TASK_RUNTIME/env/lib64"
fi
bash tools/hrn_setup/install_dependencies.sh
bash tools/hrn_setup/check_cuda.sh
