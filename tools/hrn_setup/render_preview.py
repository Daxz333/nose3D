"""Simple CPU mesh diagnostic renderer; no photograph or texture is used."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()
scene = trimesh.load(args.input, force='scene')
mesh = scene.to_geometry()
vertices = np.asarray(mesh.vertices)
normals = mesh.vertex_normals
width, height = 720, 900
scale = 3700.
pixels = np.column_stack([vertices[:, 0] * scale + width / 2,
                          height / 2 - vertices[:, 1] * scale]).astype(np.int32)
face_normals = normals[mesh.faces].mean(axis=1)
front = mesh.face_normals[:, 2] > 0
light = np.array([-.5, .6, 1.])
light /= np.linalg.norm(light)
value = np.clip(.48 + .48 * face_normals @ light, .1, 1)
order = np.flatnonzero(front)
order = order[np.argsort(vertices[mesh.faces[order], 2].mean(axis=1))]
image = np.empty((height, width, 3), np.uint8)
image[:] = [36, 29, 20]
for index in order:
    shade = int(235 * value[index])
    cv2.fillConvexPoly(image, pixels[mesh.faces[index]], [shade] * 3, lineType=cv2.LINE_AA)
cv2.imwrite(args.output, image)
print(Path(args.output).resolve())
