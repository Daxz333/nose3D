"""Diagnostic HRN inference: export true geometry before integrating UI/printing."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
vendor = ROOT / "third_party" / "HRN"
sys.path.insert(0, str(vendor))
sys.path.insert(0, str(ROOT))
os.chdir(vendor)

import cv2
import numpy as np
import torch
import trimesh
from models.hrn import Reconstructor

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
image = cv2.imread(args.input)
if image is None:
    raise ValueError("Cannot read input image")
torch.set_num_threads(4)
reconstructor = Reconstructor(["--checkpoints_dir", "assets/pretrained_models", "--name", "hrn_v1.1", "--epoch", "10"])
reconstructor.predict(image, visualize=False)
model = reconstructor.model
with torch.no_grad():
    coeff = model.facemodel_front.split_coeff(model.coeffs)
    shape = model.facemodel_front.compute_shape(coeff['id'], coeff['exp'])
    shape, _ = model.facemodel_front.add_nonlinear_offset(shape, model.deformation_map.permute(0, 2, 3, 1), model.bfm_UVs)
    normals = model.facemodel_front.compute_norm(shape)
    dense = model.facemodel_front.get_dense_mesh(model.displacement_map, shape, normals)
    mid_vertices = shape[0].cpu().numpy()
    vertices = dense['vertices'][0].cpu().numpy()
    faces = dense['faces'][0].cpu().numpy()
    # Upstream grid triangles are zero-based (check before exporting below).
    print('DENSE', vertices.shape, faces.shape, int(faces.min()), int(faces.max()))
    np.savez_compressed(args.output, mid_vertices=mid_vertices, vertices=vertices, faces=faces,
                        mid_faces=model.facemodel_front.face_buf.cpu().numpy(),
                        uv=model.bfm_UVs.cpu().numpy(),
                        displacement=model.displacement_map[0].cpu().numpy(),
                        mean_shape=model.facemodel_front.mean_shape.cpu().numpy())
    print('Saved geometry', args.output)
