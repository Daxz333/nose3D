"""Only enable the dense API after one successful end-to-end local inference."""
import json
from pathlib import Path
import subprocess

from head_reconstruction.dense_geometry import export_head, RUNTIME, ROOT
from head_reconstruction.dense_worker import reconstruct
from head_reconstruction.api import dense_signature

sample = ROOT / "frontend/public/examples/regions/nose-result.jpg"
output = ROOT / "outputs/head_dense/setup_validation.glb"
output.parent.mkdir(parents=True, exist_ok=True)
metadata = reconstruct(sample, output, 200)
if not metadata["topology"]["positive_volume"]:
    raise RuntimeError("End-to-end mesh validation failed")
export_head(RUNTIME / "templates/neutral_head.glb", height_mm=200)
commit = subprocess.check_output(["git", "-C", str(ROOT / "third_party/HRN"), "rev-parse", "HEAD"], text=True).strip()
(RUNTIME / "ready.json").write_text(json.dumps(dict(engine="hrn-dense-face", upstream_commit=commit, pipeline_signature=dense_signature(),
                                                  validation=metadata), indent=2), encoding="utf-8")
print("Dense runtime validated and enabled")
