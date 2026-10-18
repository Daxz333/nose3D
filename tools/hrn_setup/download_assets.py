"""Fetch published research assets from the authors' ModelScope repositories."""
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
destination = ROOT / ".runtime" / "hrn" / "models"
snapshot_download("damo/cv_resnet50_face-reconstruction", revision="v2.0.0-HRN",
                  local_dir=str(destination / "face"), ignore_patterns=["results/*"])
snapshot_download("damo/cv_HRN_head-reconstruction", local_dir=str(destination / "head"),
                  allow_patterns=["assets/3dmm/*", "assets/texture/template_bald_tex_2.jpg", "README.md", "configuration.json"])
print("Assets ready:", destination)
