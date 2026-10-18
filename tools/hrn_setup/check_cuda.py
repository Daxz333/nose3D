"""Compile CUDA and check UV orientation/interpolation on known data."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/HRN'))
os.chdir(ROOT / 'third_party/HRN')

import numpy as np
import torch
from head_reconstruction.hrn_uv import SRenderY

renderer = SRenderY(224).to('cuda')
uv = np.load('assets/3dmm_assets/template_mesh/bfm_uvs2.npy').astype(np.float32)
attributes = torch.tensor(np.column_stack([uv, np.ones(len(uv))]), dtype=torch.float32, device='cuda')[None]
result = renderer.world2uv(attributes)[0].detach().cpu().numpy()
y, x = np.mgrid[:256, :256]
valid = result[2] > .5
error = max(np.max(np.abs(result[0][valid] - ((x + .5) / 256)[valid])),
            np.max(np.abs(result[1][valid] - (1 - (y + .5) / 256)[valid])))
if error > .2 / 256 or np.count_nonzero(valid) < 20_000:
    raise RuntimeError(f'UV interpolation/orientation check failed: {error}')
report = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
              uv_pixels=int(np.count_nonzero(valid)), max_uv_error=float(error),
              max_error_pixels=float(error * 256))
(ROOT / '.runtime/hrn/cuda_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
