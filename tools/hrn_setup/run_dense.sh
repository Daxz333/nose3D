#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
TASK_RUNTIME="$PWD/.runtime/hrn"
export PATH="$TASK_RUNTIME/env/bin:$PATH"
export CUDA_HOME="$TASK_RUNTIME/env"
export TORCH_HOME="$TASK_RUNTIME/torch"
export TORCH_EXTENSIONS_DIR="$TASK_RUNTIME/torch_extensions"
export MPLCONFIGDIR="$TASK_RUNTIME/matplotlib"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=2
export LD_LIBRARY_PATH="$TASK_RUNTIME/env/lib:${LD_LIBRARY_PATH:-}"
python -u tools/hrn_setup/run_dense.py "$@"
