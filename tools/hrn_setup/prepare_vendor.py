"""Map author-published assets to the upstream HRN layout and patch runtime compatibility."""
from pathlib import Path
import os
import shutil
import re

ROOT = Path(__file__).resolve().parents[2]
vendor = ROOT / "third_party" / "HRN"
assets = ROOT / ".runtime" / "hrn" / "models" / "face"
mapping = {
    "assets/BFM_model_front.mat": "assets/3dmm_assets/BFM/BFM_model_front.mat",
    "assets/similarity_Lm3D_all.mat": "assets/3dmm_assets/BFM/similarity_Lm3D_all.mat",
    "assets/bfm_tex_mean2.npy": "assets/3dmm_assets/bfm_albedo_map_basis/bfm_tex_mean2.npy",
    "assets/bfm_texmap_base2.npy": "assets/3dmm_assets/bfm_albedo_map_basis/bfm_texmap_base2.npy",
    "assets/bfm_uvs2.npy": "assets/3dmm_assets/template_mesh/bfm_uvs2.npy",
    "assets/template_bfm.obj": "assets/3dmm_assets/template_mesh/template_bfm.obj",
    "de_retouching_model.pth": "assets/pretrained_models/de-retouching.pth",
    "large_base_net.pth": "assets/pretrained_models/large_base_net.pth",
    "segment_face.pb": "assets/pretrained_models/segment_face.pb",
    "retinaface_resnet50_2020-07-20_old_torch.pth": "assets/pretrained_models/retinaface_resnet50_2020-07-20_old_torch.pth",
    "pytorch_model.pt": "assets/pretrained_models/hrn_v1.1/epoch_10.pth",
}
for source, target in mapping.items():
    source, target = assets / source, vendor / target
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

cache = ROOT / ".runtime" / "hrn" / "torch" / "hub" / "checkpoints"
cache.mkdir(parents=True, exist_ok=True)
for source in (assets / "face_alignment").iterdir():
    target = cache / source.name
    if not target.exists():
        shutil.copy2(source, target)

path = vendor / "util" / "nv_diffrast.py"
text = path.read_text()
path.write_text(text.replace("dr.RasterizeGLContext(device=device)", "dr.RasterizeCudaContext(device=device)"))

# TensorFlow segmentation runs on CPU so PyTorch owns the 8 GB GPU budget.
path = vendor / "models" / "hrn.py"
text = path.read_text().replace("tf.ConfigProto(allow_soft_placement=True)",
                                "tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})")
text = text.replace("face_alignment.LandmarksType._3D", "face_alignment.LandmarksType.THREE_D")
path.write_text(text)

# Inference only needs UV interpolation; keep one CUDA rasterizer instead of
# compiling the entire PyTorch3D training stack.
path = vendor / "models" / "bfm.py"
text = path.read_text()
text = text.replace("from util.renderer import SRenderY, set_rasterizer",
                    "from head_reconstruction.hrn_uv import SRenderY, set_rasterizer")
text = text.replace("set_rasterizer('pytorch3d')", "set_rasterizer('nvdiffrast')")
text = text.replace("rasterizer_type='pytorch3d'", "rasterizer_type='nvdiffrast'")
path.write_text(text)
path = vendor / "models" / "losses.py"
text = path.read_text()
text = text.replace("from pytorch3d.ops import (\n    corresponding_points_alignment,\n    knn_points,\n    knn_gather\n)", "")
if "    from pytorch3d.ops import knn_points, knn_gather" not in text:
    text = text.replace("    knn = knn_points(template_verts", "    from pytorch3d.ops import knn_points, knn_gather\n    knn = knn_points(template_verts")
path.write_text(text)
for name in ("util/util_.py", "util/image_liquify.py", "util/deca_util.py"):
    path = vendor / name
    text = re.sub(r"np\.float\b", "np.float64", path.read_text())
    text = re.sub(r"np\.int\b", "np.int64", text)
    path.write_text(text)
path = vendor / "util/preprocess.py"
text = path.read_text().replace("np.array([w0, h0, s, t[0], t[1]])",
                                "np.array([w0, h0, np.asarray(s).item(), np.asarray(t[0]).item(), np.asarray(t[1]).item()])")
path.write_text(text)
print("Upstream HRN assets and local compatibility patches ready")
