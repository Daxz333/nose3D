"""A deterministic head; only the interior of its face can deform.

Coordinates are centimetres internally, +Y up and +Z towards the viewer.
MediaPipe z is a learned weak-perspective estimate, NOT measured depth.
"""
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import trimesh

ASSETS = Path(__file__).parent / "assets"
SKIN = np.array([184, 154, 137], dtype=np.uint8)
TEMPLATE_VERSION = "neutral-head-v1"


def canonical_face():
    vertices, texcoords, faces, texture_faces = [], [], [], []
    for line in (ASSETS / "canonical_face_model.obj").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append(list(map(float, parts[1:4])))
        elif parts[0] == "vt":
            texcoords.append(list(map(float, parts[1:3])))
        elif parts[0] == "f":
            indices = [list(map(int, p.split("/")[:2])) for p in parts[1:]]
            faces.append([p[0] - 1 for p in indices])
            texture_faces.append([p[1] - 1 for p in indices])
    vertices, faces = np.array(vertices), np.array(faces)
    uv = np.zeros((len(vertices), 2))
    for face, tf in zip(faces, texture_faces):
        uv[face] = np.asarray(texcoords)[tf]
    return vertices, faces, uv


def boundary_loops(faces):
    counts = Counter(tuple(sorted((int(a), int(b)))) for f in faces
                     for a, b in zip(f, np.roll(f, -1)))
    adjacency = {}
    for (a, b), count in counts.items():
        if count == 1:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adjacency.values()):
        raise ValueError("Invalid template boundary")
    remaining = set(adjacency)
    loops = []
    while remaining:
        first = min(remaining)
        loop, previous, current = [first], None, first
        while True:
            following = next(n for n in adjacency[current] if n != previous)
            if following == first:
                break
            loop.append(following)
            previous, current = current, following
        remaining.difference_update(loop)
        loops.append(np.array(loop))
    return sorted(loops, key=len, reverse=True)


def face_weights(vertices, faces, outer):
    adjacency = [set() for _ in vertices]
    for face in faces:
        for i in face:
            adjacency[i].update(face)
    distance = np.full(len(vertices), 100, dtype=int)
    distance[outer] = 0
    queue = deque(outer)
    while queue:
        i = queue.popleft()
        for j in adjacency[i]:
            if distance[j] > distance[i] + 1:
                distance[j] = distance[i] + 1
                queue.append(j)
    # First two rings remain exactly unchanged; the next rings blend smoothly.
    t = np.clip((distance - 1) / 3, 0, 1)
    return t * t * (3 - 2 * t)


def fit_face(landmarks, image_size, canonical, weights):
    width, height = image_size
    points = landmarks[:468] * np.array([width, -height, -width])
    # Similarity alignment removes camera pose and scale before local editing.
    anchors = np.array([4, 6, 33, 133, 263, 362, 168, 10, 152, 234, 454, 93, 323])
    source, target = points[anchors], canonical[anchors]
    source_center, target_center = source.mean(0), target.mean(0)
    x, y = source - source_center, target - target_center
    u, singular, vt = np.linalg.svd(x.T @ y)
    correction = np.ones(3)
    correction[-1] = np.linalg.det(u @ vt)
    rotation = (u * correction) @ vt
    scale = np.sum(singular * correction) / np.sum(x * x)
    fitted = (points - source_center) @ rotation * scale + target_center
    delta = fitted - canonical
    # Keep learned depth conservative and reject extreme template deformations.
    delta[:, 2] *= .65
    delta = np.clip(delta, -1.8, 1.8)
    return canonical + delta * weights[:, None]


def close_face_holes(vertices, faces, uv, pixels, weights, loops):
    """Close eyes/mouth with textured patches, without generating hidden anatomy."""
    vertices, faces, uv = list(vertices), list(faces), list(uv)
    pixels, weights = list(pixels), list(weights)
    for loop in loops[1:]:
        center = len(vertices)
        vertices.append(np.mean([vertices[i] for i in loop], axis=0))
        uv.append(np.mean([uv[i] for i in loop], axis=0))
        pixels.append(np.mean([pixels[i] for i in loop], axis=0))
        weights.append(float(np.mean([weights[i] for i in loop])))
        for a, b in zip(loop, np.roll(loop, -1)):
            faces.append([a, b, center])
    return tuple(np.asarray(v) for v in (vertices, faces, uv, pixels, weights))


def bake_texture(image, faces, uv, pixels, weights, size=1024):
    """Bake photo triangles into a stable atlas, feathering to fixed template skin."""
    atlas = np.empty((size, size, 3), np.uint8)
    atlas[:] = SKIN
    dst = uv * np.array([size - 1, -(size - 1)]) + [0, size - 1]
    for face in faces:
        triangle = dst[face].astype(np.float32)
        x, y, w, h = cv2.boundingRect(triangle)
        x, y = max(0, x), max(0, y)
        w, h = min(w, size - x), min(h, size - y)
        if w <= 0 or h <= 0:
            continue
        local = triangle - [x, y]
        matrix = cv2.getAffineTransform(pixels[face].astype(np.float32), local.astype(np.float32))
        warped = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        yy, xx = np.mgrid[:h, :w]
        system = np.vstack([local.T, np.ones(3)])
        if abs(np.linalg.det(system)) < 1e-5:
            continue
        bary = np.linalg.solve(system, np.stack([xx.ravel(), yy.ravel(), np.ones(w * h)]))
        inside = (bary.min(axis=0) >= -.03).reshape(h, w)
        alpha = np.clip(weights[face] @ bary, 0, 1).reshape(h, w, 1)
        blended = np.clip(warped * alpha + SKIN * (1 - alpha), 0, 255).astype(np.uint8)
        atlas[y:y+h, x:x+w][inside] = blended[inside]
    return Image.fromarray(atlas)


def solid_material():
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=[*SKIN, 255], metallicFactor=0., roughnessFactor=.85,
        doubleSided=True)


def fixed_head(canonical, outer):
    """Fixed scalp/back joined exactly to the canonical face boundary."""
    border = canonical[outer]
    # Preserve the face edge; sweep backwards into a rounded cranium.
    vertices = [p.copy() for p in border]
    n, ring_count = len(border), 28
    for k in range(1, ring_count):
        t = k / ring_count
        angle = np.pi * t / 2
        ring = border.copy()
        ring[:, 0] *= np.cos(angle) * (1 + .10 * np.sin(np.pi * t))
        ring[:, 1] = (border[:, 1] - .7) * np.cos(angle) + .7 + 4.5 * np.sin(np.pi * t)
        ring[:, 2] = border[:, 2] * (1 - t) - 8.8 * np.sin(angle)
        vertices.extend(ring)
    vertices.append([0, .7, -8.8])
    faces = []
    for ring in range(ring_count - 1):
        for i in range(n):
            a, b = ring * n + i, ring * n + (i + 1) % n
            faces.extend([[a, b, b + n], [a, b + n, a + n]])
    for i in range(n):
        faces.append([(ring_count - 1) * n + i,
                      (ring_count - 1) * n + (i + 1) % n, len(vertices) - 1])
    mesh = trimesh.Trimesh(np.array(vertices) / 100, faces, process=False)
    mesh.fix_normals()
    mesh.visual = trimesh.visual.TextureVisuals(material=solid_material())
    return mesh


def ellipsoid(center, scale):
    mesh = trimesh.creation.uv_sphere(count=[32, 32])
    mesh.vertices = (mesh.vertices * scale + center) / 100
    mesh.visual = trimesh.visual.TextureVisuals(material=solid_material())
    return mesh


def fixed_neck():
    mesh = trimesh.creation.cylinder(radius=1, height=7.4, sections=64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.vertices = (mesh.vertices * [3.3, 1, 3.1] + [0, -10.8, -3.5]) / 100
    mesh.visual = trimesh.visual.TextureVisuals(material=solid_material())
    return mesh


def build_scene(image=None, landmarks=None):
    canonical, faces, uv = canonical_face()
    loops = boundary_loops(faces)
    weights = face_weights(canonical, faces, loops[0])
    vertices = canonical.copy()
    if landmarks is not None:
        vertices = fit_face(landmarks, (image.shape[1], image.shape[0]), canonical, weights)
        pixels = landmarks[:468, :2] * [image.shape[1], image.shape[0]]
    else:
        pixels = uv.copy()
    vertices, faces, uv, pixels, full_weights = close_face_holes(
        vertices, faces, uv, pixels, weights, loops)
    face_mesh = trimesh.Trimesh(vertices / 100, faces, process=False)
    face_mesh.fix_normals()
    material = solid_material()
    if image is not None:
        texture = bake_texture(image, faces, uv, pixels, full_weights)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture, baseColorFactor=[255]*4,
            metallicFactor=0., roughnessFactor=.85, doubleSided=True)
    face_mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    scene = trimesh.Scene()
    scene.add_geometry(face_mesh, geom_name="face", node_name="face")
    scene.add_geometry(fixed_head(canonical, loops[0]), geom_name="fixed_scalp", node_name="fixed_scalp")
    for side in (-1, 1):
        scene.add_geometry(ellipsoid([side * 7.4, -.5, -.6], [.85, 2.0, 1.0]),
                           geom_name=f"fixed_ear_{side}", node_name=f"fixed_ear_{side}")
    scene.add_geometry(fixed_neck(),
                       geom_name="fixed_neck", node_name="fixed_neck")
    metadata = {
        "template": TEMPLATE_VERSION, "method": "mediapipe-coarse-face",
        "units": "metres", "face_vertices": len(vertices),
        "editable_vertices": int(np.count_nonzero(weights)),
        "fixed_parts": ["scalp", "ears", "neck", "face_boundary"],
        "approximate_depth": True,
    }
    scene.metadata.update(metadata)
    return scene, metadata
