"""UV-only rasterization for HRN inference using its existing nvdiffrast dependency.

Upstream SRenderY is used only for world2uv and its dense grid in this pipeline.
The UV projection remains barycentric and differentiable; model weights and
predicted geometry are unchanged. This avoids a second compiled rasterizer.
"""
import numpy as np
import torch
from torch import nn
import nvdiffrast.torch as dr


def set_rasterizer(name):
    if name != "nvdiffrast":
        raise ValueError("The HRN inference adapter uses nvdiffrast")


class SRenderY(nn.Module):
    def __init__(self, image_size, uv_size=256, rasterizer_type="nvdiffrast"):
        super().__init__()
        from scipy.io import loadmat
        from util.deca_util import generate_triangles
        self.uv_size = uv_size
        faces = loadmat("assets/3dmm_assets/BFM/BFM_model_front.mat")["tri"].astype(np.int32) - 1
        uv = np.load("assets/3dmm_assets/template_mesh/bfm_uvs2.npy").astype(np.float32)
        clip = np.column_stack([uv * 2 - 1, np.zeros(len(uv)), np.ones(len(uv))]).astype(np.float32)
        self.register_buffer("faces", torch.from_numpy(faces))
        self.register_buffer("clip", torch.from_numpy(clip)[None])
        self.register_buffer("dense_faces", torch.from_numpy(generate_triangles(uv_size, uv_size)).long()[None])
        self._context = None
        self._raster = None

    def world2uv(self, vertices):
        if self._context is None:
            self._context = dr.RasterizeCudaContext(device=vertices.device)
        if self._raster is None:
            self._raster, _ = dr.rasterize(self._context, self.clip.contiguous(),
                                            self.faces.contiguous(), [self.uv_size, self.uv_size])
        raster = self._raster.expand(vertices.shape[0], -1, -1, -1).contiguous()
        values, _ = dr.interpolate(vertices.contiguous(), raster, self.faces.contiguous())
        # nvdiffrast rows run from bottom to top; HRN expects top to bottom.
        return values.flip(1).permute(0, 3, 1, 2).contiguous()
