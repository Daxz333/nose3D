"""Run only with .venv-head Python; keep ML dependencies out of the Flask host."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .geometry import ASSETS, build_scene

# Keep optional matplotlib's cache inside the project, including on locked-down hosts.
_cache = ASSETS.parent.parent / "outputs" / "head_cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))


class InputError(ValueError):
    pass


def detect_image(input_path):
    import mediapipe as mp

    try:
        with Image.open(input_path) as opened:
            if opened.width * opened.height > 24_000_000:
                raise InputError("图片分辨率过大，请缩小到 2400 万像素以内")
            photo = ImageOps.exif_transpose(opened).convert("RGB")
            photo.thumbnail((1600, 1600))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InputError("无法读取图片，请上传有效的 JPG、PNG 或 WebP 图片") from exc
    image = np.asarray(photo).copy()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ASSETS / "face_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
        output_facial_transformation_matrixes=True)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
    if len(result.face_landmarks) != 1:
        raise InputError("请上传仅包含一张清晰人脸的照片；当前未检测到人脸或检测到多个人脸")
    points = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]])
    if not np.isfinite(points).all():
        raise InputError("人脸关键点无效，请更换清晰照片")
    if np.any(points[:468, :2] < 0) or np.any(points[:468, :2] > 1):
        raise InputError("人脸被裁切，请上传额头、下巴和两侧脸颊完整可见的照片")
    face_size = np.ptp(points[:468, :2], axis=0) * photo.size
    if min(face_size) < 100:
        raise InputError("人脸太小，请上传脸部至少 100 像素宽的清晰照片")
    matrix = np.asarray(result.facial_transformation_matrixes[0])
    normal = matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    if abs(normal[2]) < np.cos(np.deg2rad(30)):
        raise InputError("请使用正面或轻微侧转的人脸照片（转角不超过约 30°）")
    return image, points


def reconstruct(input_path, output_path):
    image, points = detect_image(input_path)
    scene, metadata = build_scene(image, points)
    Path(output_path).write_bytes(scene.export(file_type="glb", include_normals=True))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()
    try:
        if args.template:
            scene, metadata = build_scene()
            Path(args.output).write_bytes(scene.export(file_type="glb", include_normals=True))
        else:
            metadata = reconstruct(args.input, args.output)
        print(json.dumps({"success": True, "metadata": metadata}, ensure_ascii=True))
    except InputError as exc:
        print(json.dumps({"success": False, "error": str(exc), "error_code": "invalid_face"}, ensure_ascii=True))
        sys.exit(2)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"success": False, "error": "三维重建失败，请检查模型与独立环境", "error_code": "reconstruction_failed"}, ensure_ascii=True))
        sys.exit(1)


if __name__ == "__main__":
    main()
