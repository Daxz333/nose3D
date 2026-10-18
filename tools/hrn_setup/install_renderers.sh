#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
TASK_RUNTIME="$PWD/.runtime/hrn"
export PATH="$TASK_RUNTIME/env/bin:$PATH"
export CUDA_HOME="$TASK_RUNTIME/env"
export PIP_CACHE_DIR="$TASK_RUNTIME/pip-cache"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=2
python -m pip install setuptools==69.5.1 wheel ninja
if [ -d third_party/nvdiffrast ]; then
  python -m pip install --no-build-isolation ./third_party/nvdiffrast
else
  python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3'
fi
python -c 'import torch, nvdiffrast.torch; print(torch.__version__, torch.cuda.is_available())'
