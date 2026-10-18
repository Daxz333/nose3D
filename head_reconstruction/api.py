"""Local-only reconstruction API, shared by the main and standalone servers."""
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request, send_from_directory, url_for

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "heads"
ASSETS = ROOT / "head_reconstruction" / "assets"
DENSE_RUNTIME = ROOT / ".runtime" / "hrn"
MAX_UPLOAD = 12 * 1024 * 1024
_slot = threading.BoundedSemaphore(1)
head_api = Blueprint("heads", __name__, url_prefix="/api/head")


def python_path():
    fallback = ROOT / ".venv-head" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return Path(os.environ.get("AIHB_HEAD_PYTHON", str(fallback)))


def readiness():
    required = [python_path(), ASSETS / "face_landmarker.task",
                ASSETS / "canonical_face_model.obj", ASSETS / "neutral_head.glb"]
    return all(p.is_file() for p in required)


def dense_signature():
    checksum = hashlib.sha256()
    for name in ("dense_geometry.py", "dense_worker.py", "hrn_uv.py"):
        checksum.update((ROOT / "head_reconstruction" / name).read_bytes())
    return checksum.hexdigest()


def dense_readiness():
    if not (python_path().is_file() and (DENSE_RUNTIME / "ready.json").is_file()
            and (DENSE_RUNTIME / "templates" / "neutral_head.glb").is_file()):
        return False
    try:
        validated = json.loads((DENSE_RUNTIME / "ready.json").read_text(encoding="utf-8"))
        return validated.get("pipeline_signature") == dense_signature()
    except (ValueError, OSError):
        return False


@head_api.get("/dense/status")
def dense_status():
    return jsonify(success=True, ready=dense_readiness(), template="hrn-fixed-head-v2",
                   engine="hrn-dense-face", template_url=url_for("heads.dense_template"),
                   template_available=(DENSE_RUNTIME / "templates" / "neutral_head.glb").is_file(),
                   message="稠密重建就绪" if dense_readiness() else "稠密重建环境尚未就绪，请完成 HRN 安装与样例验证")


@head_api.get("/dense/template.glb")
def dense_template():
    return send_from_directory(DENSE_RUNTIME / "templates", "neutral_head.glb", mimetype="model/gltf-binary")


@head_api.get("/status")
def status():
    return jsonify(success=True, ready=readiness(), template="neutral-head-v1",
                   template_url=url_for("heads.template"),
                   message="就绪" if readiness() else "请先运行 setup_head.bat 安装三维重建环境")


@head_api.get("/template.glb")
def template():
    return send_from_directory(ASSETS, "neutral_head.glb", mimetype="model/gltf-binary")


@head_api.get("/models/<name>")
def model(name):
    if not re.fullmatch(r"[0-9a-f]{32}\.(glb|stl|json)", name):
        return jsonify(success=False, error="模型不存在"), 404
    mimetype = {"glb": "model/gltf-binary", "stl": "model/stl", "json": "application/json"}[name.rsplit('.', 1)[1]]
    return send_from_directory(OUTPUT, name, mimetype=mimetype,
                               as_attachment=request.args.get("download") == "1")


@head_api.post("/reconstruct")
def reconstruct():
    return reconstruct_request(dense=False)


@head_api.post("/dense/reconstruct")
def reconstruct_dense():
    return reconstruct_request(dense=True)


def reconstruct_request(dense=False):
    if request.content_length and request.content_length > MAX_UPLOAD + 65536:
        return jsonify(success=False, error="图片过大，请控制在 12 MB 内"), 413
    if dense and not dense_readiness():
        return jsonify(success=False, error="稠密重建环境尚未就绪"), 503
    if not dense and not readiness():
        return jsonify(success=False, error="三维环境尚未就绪，请先运行 setup_head.bat"), 503
    try:
        height = float(request.form.get("height_mm", "200"))
        if not 150 <= height <= 250:
            raise ValueError()
    except ValueError:
        return jsonify(success=False, error="打印高度须在 150–250 mm 之间"), 400
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify(success=False, error="请上传 image 图片文件"), 400
    content = uploaded.stream.read(MAX_UPLOAD + 1)
    if not content or len(content) > MAX_UPLOAD:
        return jsonify(success=False, error="图片为空或超过 12 MB"), 400
    if not _slot.acquire(blocking=False):
        return jsonify(success=False, error="正在处理另一张人脸，请稍后重试"), 429
    name = f"{uuid4().hex}.glb"
    destination = OUTPUT / name
    try:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="head-", dir=OUTPUT) as folder:
            source = Path(folder) / "input.img"
            source.write_bytes(content)
            command = [str(python_path()), "-m",
                       "head_reconstruction.dense_worker" if dense else "head_reconstruction.worker",
                       "--input", str(source), "--output", str(destination)]
            if dense:
                command += ["--height", str(height)]
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=480 if dense else 120,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        data = json.loads(result.stdout.strip().splitlines()[-1])
        if result.returncode or not data.get("success"):
            current_app.logger.warning("Head worker failed: %s", result.stderr[-2000:])
            destination.unlink(missing_ok=True)
            return jsonify(data), 422 if result.returncode == 2 else 500
        if not destination.is_file():
            raise RuntimeError("Worker did not create a model")
        response = dict(success=True, model_url=url_for("heads.model", name=name),
                        download_url=url_for("heads.model", name=name, download=1), metadata=data["metadata"])
        if dense:
            response["stl_url"] = url_for("heads.model", name=destination.with_suffix(".stl").name, download=1)
            response["report_url"] = url_for("heads.model", name=destination.with_suffix(".json").name, download=1)
        return jsonify(response)
    except subprocess.TimeoutExpired:
        destination.unlink(missing_ok=True)
        return jsonify(success=False, error="重建超时，请缩小图片后重试"), 504
    except Exception:
        destination.unlink(missing_ok=True)
        current_app.logger.exception("Head reconstruction failed")
        return jsonify(success=False, error="三维重建失败，请查看服务日志"), 500
    finally:
        # Do not leave partially written sidecars if a worker failed.
        if not destination.is_file():
            destination.with_suffix(".stl").unlink(missing_ok=True)
            destination.with_suffix(".json").unlink(missing_ok=True)
        _slot.release()
