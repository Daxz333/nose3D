#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
TASK_RUNTIME="$PWD/.runtime/hrn"
export PATH="$TASK_RUNTIME/env/bin:$PATH"
export CUDA_HOME="$TASK_RUNTIME/env"
export PIP_CACHE_DIR="$TASK_RUNTIME/pip-cache"
python -m pip install numpy==1.24.3 torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install scipy==1.10.1 tensorflow-cpu==2.13.1 opencv-python-headless==4.8.1.78 scikit-image==0.21.0 pillow==10.2.0 ninja trimesh==4.6.8 face-alignment==1.3.5 kornia==0.6.12 einops==0.8.0 torchsummary==1.5.1 tensorboard==2.13.0 gdown==5.2.0 modelscope==1.18.0 addict==2.4.0 pyyaml==6.0.2 tqdm matplotlib==3.7.2 imageio==2.34.2
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip install albumentations==1.3.1 scikit-learn==1.3.2 opencv-python-headless==4.8.1.78
bash tools/hrn_setup/install_renderers.sh
