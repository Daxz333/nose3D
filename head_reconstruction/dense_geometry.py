"""Transfer learned HRN face geometry onto one closed, fixed research head.

The finer template is only a sampling surface. New facial detail comes from
HRN's learned deformation/displacement maps, not from subdivision itself.
"""
from pathlib import Path
import json

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import trimesh

from .geometry import boundary_loops

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime" / "hrn"
CACHE = RUNTIME / "templates"
TEMPLATE_ID = "hrn-fixed-head-v2"


def smoothstep(values):
    t = np.clip(values, 0, 1)
    return t * t * (3 - 2 * t)


def template_data():
    cache_file = CACHE / "fixed_head_v7.npz"
    if cache_file.is_file():
        return dict(np.load(cache_file))
    import pymeshfix
    model = loadmat(RUNTIME / "models/head/assets/3dmm/BFM/ourRefineFull_model.mat")
    vertices, faces = pymeshfix.clean_from_arrays(
        np.asarray(model["meanshape"].reshape(-1, 3), dtype=np.float64),
        np.asarray(model["tri"] - 1, dtype=np.int32), verbose=False)
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    # A flat, closed neck base gives the slicer a consistent placement plane.
    base_y = float(mesh.bounds[0, 1] + .10)
    mesh = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0, 1, 0], plane_origin=[0, base_y, 0], cap=True)
    mesh.merge_vertices()
    mesh.fix_normals()
    if not mesh.is_volume or mesh.body_count != 1:
        raise ValueError("The fixed template is not one closed volume")
    vertices, faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
    raw = loadmat(RUNTIME / "models/face/assets/BFM_model_front.mat")
    mean_face = raw["meanshape"].reshape(-1, 3).astype(np.float64)
    # Fair the common scalp/neck geometrically, while protecting the detailed
    # face, ear folds and flat print base from the smoothing operation.
    shared = trimesh.Trimesh(vertices.copy(), faces, process=False)
    trimesh.smoothing.filter_taubin(shared, lamb=.5, nu=.53, iterations=20)
    face_distance = cKDTree(mean_face).query(vertices)[0]
    smooth_weight = smoothstep((face_distance - .12) / .15)
    ear = smoothstep((np.abs(vertices[:, 0]) - .70) / .08)
    ear *= 1 - smoothstep((np.abs(vertices[:, 1]) - .6) / .15)
    ear *= smoothstep((vertices[:, 2] + .45) / .15)
    smooth_weight *= (1 - ear) * smoothstep((vertices[:, 1] - base_y) / .12)
    vertices += (shared.vertices - vertices) * smooth_weight[:, None]
    face_tri = raw["tri"].astype(np.int64) - 1
    outer = boundary_loops(face_tri)[0]
    edges = trimesh.Trimesh(mean_face, face_tri, process=False).edges_unique
    lengths = np.linalg.norm(mean_face[edges[:, 0]] - mean_face[edges[:, 1]], axis=1)
    graph = coo_matrix((np.tile(lengths, 2),
                       (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(len(mean_face), len(mean_face))).tocsr()
    # min_only avoids a boundary_count x vertex_count distance matrix.
    distance_from_border = dijkstra(graph, directed=False, indices=outer, min_only=True)
    nearest_distance, nearest = cKDTree(mean_face).query(vertices, k=4)
    inverse = 1 / np.maximum(nearest_distance, 1e-8) ** 2
    interpolation = inverse / inverse.sum(axis=1, keepdims=True)
    inner_distance = (distance_from_border[nearest] * interpolation).sum(axis=1)
    # Repaired nostril and lip patches sit slightly away from the original BFM
    # surface. They must follow the surrounding face instead of being frozen.
    face_region = 1 - smoothstep((nearest_distance[:, 0] - .03) / .09)
    local_y = (mean_face[nearest, 1] * interpolation).sum(axis=1)
    fade_width = .25 + .20 * smoothstep((-local_y - .45) / .20)
    weights = smoothstep(inner_distance / fade_width) * face_region
    weights[nearest_distance[:, 0] >= .12] = 0
    # Exterior cranium, ears and neck have zero edit weight, shared by every run.
    data = dict(vertices=vertices, faces=faces, nearest=nearest,
                interpolation=interpolation, weights=weights, mean_face=mean_face,
                face_tri=face_tri)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **data)
    return data


def apply_dense_prediction(prediction, template):
    mid = prediction["mid_vertices"]
    mean = prediction["mean_shape"].reshape(-1, 3)
    if mid.shape != mean.shape or mid.shape != template["mean_face"].shape:
        raise ValueError("Unexpected HRN face topology")
    normals = trimesh.Trimesh(mid, template["face_tri"], process=False).vertex_normals
    displacement = gaussian_filter(np.asarray(prediction["displacement"]).squeeze(), .6)
    uv = prediction["uv"]
    sampled = map_coordinates(displacement,
                              [(1 - uv[:, 1]) * (displacement.shape[0] - 1),
                              uv[:, 0] * (displacement.shape[1] - 1)], order=1, mode="nearest")
    # Hair/skin changes near the photo's hairline must not become a sculpted
    # ridge. Keep detail on the features, fading it above the upper forehead.
    sampled *= smoothstep((.80 - template["mean_face"][:, 1]) / .18)
    sampled *= smoothstep((template["mean_face"][:, 1] + .72) / .16)
    delta = mid - mean + normals * sampled[:, None]
    if not np.isfinite(delta).all() or np.max(np.abs(delta)) > .8:
        raise ValueError("Face deformation exceeds the fixed head's supported range")
    local_delta = np.sum(delta[template["nearest"]] * template["interpolation"][:, :, None], axis=1)
    return template["vertices"] + local_delta * template["weights"][:, None]


def topology_checks(mesh):
    edge_use = np.bincount(mesh.edges_unique_inverse)
    return dict(watertight=bool(mesh.is_watertight),
                winding_consistent=bool(mesh.is_winding_consistent),
                positive_volume=bool(mesh.is_volume), bodies=int(mesh.body_count),
                boundary_edges=int(np.count_nonzero(edge_use == 1)),
                nonmanifold_edges=int(np.count_nonzero(edge_use > 2)))


def export_head(output, prediction=None, height_mm=200):
    if not 150 <= height_mm <= 250:
        raise ValueError("打印高度须在 150–250 mm 之间")
    template = template_data()
    vertices = template["vertices"] if prediction is None else apply_dense_prediction(prediction, template)
    mesh = trimesh.Trimesh(vertices.copy(), template["faces"], process=False)
    # Scaling uses the fixed template, keeping all exterior points identical
    # across people at the same requested print height.
    scale = height_mm / np.ptp(template["vertices"][:, 1])
    mesh.vertices *= scale
    mesh.vertices[:, 1] -= template["vertices"][:, 1].min() * scale
    # Exact binary millimetre coordinates survive STL float32 without cracks
    # or near-coplanar rounding noise. The maximum change is below 1 micrometre.
    mesh.vertices = np.rint(mesh.vertices * 1024) / 1024
    checks = topology_checks(mesh)
    if not (checks["positive_volume"] and checks["bodies"] == 1 and
            checks["boundary_edges"] == 0 and checks["nonmanifold_edges"] == 0):
        raise ValueError("头模拓扑检查未通过，已停止打印文件导出")
    import pymeshfix
    intersection_check = pymeshfix.PyTMesh()
    intersection_check.set_quiet(True)
    intersection_check.load_array(np.asarray(mesh.vertices, dtype=np.float64),
                                 np.asarray(mesh.faces, dtype=np.int32))
    intersections = intersection_check.select_intersecting_triangles(justproper=True)
    repair_count = int(len(np.unique(intersections)))
    exterior_drift = 0.
    if repair_count:
        before_repair = mesh.vertices.copy()
        if not intersection_check.clean(max_iters=10, inner_loops=3):
            raise ValueError("局部自交修复未完成，已停止打印文件导出")
        repaired_vertices, repaired_faces = intersection_check.return_arrays()
        mesh = trimesh.Trimesh(repaired_vertices, repaired_faces, process=False)
        checks = topology_checks(mesh)
        if not mesh.is_volume or mesh.body_count != 1:
            raise ValueError("局部修复后网格未封闭，已停止打印文件导出")
        # The repaired interior of the face may change topology. Protect the
        # cranium/ears/neck against repair that would alter the common head.
        face_distance = np.linalg.norm(template["vertices"] - template["mean_face"][template["nearest"][:, 0]], axis=1)
        exterior = (template["weights"] == 0) & (face_distance > .15)
        # Repair can remove a redundant vertex without moving the surface.
        # Measure both surface directions rather than nearest-vertex spacing.
        forward_distance = trimesh.proximity.closest_point(mesh, before_repair[exterior])[1]
        repaired_raw = mesh.vertices.copy() / scale
        repaired_raw[:, 1] += template["vertices"][:, 1].min()
        repaired_exterior = cKDTree(template["mean_face"]).query(repaired_raw)[0] > .15
        before_mesh = trimesh.Trimesh(before_repair, template["faces"], process=False)
        reverse_distance = trimesh.proximity.closest_point(before_mesh, mesh.vertices[repaired_exterior])[1]
        exterior_drift = float(max(forward_distance.max(), reverse_distance.max()))
        if exterior_drift > .005:
            raise ValueError("修复影响了固定头部区域，已停止打印文件导出")
        intersections = intersection_check.select_intersecting_triangles(justproper=True)
    checks["self_intersection_triangles"] = int(len(np.unique(intersections)))
    if checks["self_intersection_triangles"]:
        raise ValueError("头模仍存在自交三角形，已停止打印文件导出")
    checks["repaired_intersection_triangles"] = repair_count
    checks["fixed_exterior_drift_mm"] = exterior_drift
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # GLB is metres, Y up, centered for the browser viewer.
    preview = mesh.copy()
    preview.vertices[:, 1] -= height_mm / 2
    preview.vertices /= 1000
    preview.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[183, 188, 198, 255], metallicFactor=0., roughnessFactor=.85))
    preview.export(output.with_suffix(".glb"), include_normals=True)
    # STL has numeric millimetres, Z up, neck base at Z=0. STL itself has no unit field.
    printable = mesh.copy()
    printable.vertices = printable.vertices[:, [0, 2, 1]] * [1, -1, 1]
    printable.export(output.with_suffix(".stl"))
    metadata = dict(template=TEMPLATE_ID, method="hrn-dense-face", units="mm",
                    height_mm=float(height_mm), dimensions_mm=printable.extents.tolist(),
                    vertices=len(mesh.vertices), faces=len(mesh.faces),
                    face_vertices=int(np.count_nonzero(template["weights"])),
                    fixed_parts=["scalp", "ears", "neck", "face_boundary"],
                    approximate_depth=True, topology=checks,
                    precheck_coordinate_resolution_mm=1 / 1024,
                    quality_status="topology_checked_requires_slicer_review")
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=float, default=200)
    args = parser.parse_args()
    prediction = None if not args.prediction else dict(np.load(args.prediction))
    print(json.dumps(export_head(args.output, prediction, args.height), ensure_ascii=False))
