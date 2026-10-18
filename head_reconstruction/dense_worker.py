"""Windows-side HRN bridge and print mesh export. No coarse fallback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

from .dense_geometry import ROOT, RUNTIME, export_head
from .worker import detect_image


def wsl_path(path):
    path = Path(path).resolve()
    if os.name != "nt":
        return str(path)
    return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]


def reconstruct(source, destination, height_mm=200):
    with tempfile.TemporaryDirectory(prefix="dense-", dir=ROOT / "outputs") as folder:
        folder = Path(folder)
        image, _ = detect_image(source)
        Image.fromarray(image).save(folder / "photo.png")
        script = ROOT / "tools/hrn_setup/run_dense.sh"
        command = ["bash", str(script), "--input", str(folder / "photo.png"),
                   "--output", str(folder / "prediction.npz")]
        if os.name == "nt":
            command = ["wsl.exe", "-d", os.environ.get("AIHB_HRN_DISTRO", "Ubuntu-22.04"),
                       "--", "timeout", "420s", "bash", wsl_path(script),
                       "--input", wsl_path(folder / "photo.png"),
                       "--output", wsl_path(folder / "prediction.npz")]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=450,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if process.returncode:
            print(process.stdout[-3000:], file=sys.stderr)
            print(process.stderr[-6000:], file=sys.stderr)
            raise RuntimeError("稠密人脸重建失败，请检查服务日志")
        prediction = dict(np.load(folder / "prediction.npz"))
        return export_head(destination, prediction, height_mm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=float, default=200)
    args = parser.parse_args()
    try:
        print(json.dumps(dict(success=True, metadata=reconstruct(args.input, args.output, args.height)), ensure_ascii=True))
    except ValueError as exc:
        print(json.dumps(dict(success=False, error=str(exc), error_code="invalid_face"), ensure_ascii=True))
        sys.exit(2)
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps(dict(success=False, error="稠密重建失败，请检查服务日志", error_code="dense_reconstruction_failed"), ensure_ascii=True))
        sys.exit(1)
