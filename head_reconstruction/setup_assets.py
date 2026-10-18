"""Download official public assets when missing; no downloads during inference."""
from pathlib import Path
from urllib.request import urlopen

ASSETS = Path(__file__).parent / "assets"
URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "canonical_face_model.obj": "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj",
    "LICENSE.mediapipe": "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/LICENSE",
}


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    for name, url in URLS.items():
        path = ASSETS / name
        if not path.is_file():
            with urlopen(url, timeout=60) as response:
                content = response.read()
            path.write_bytes(content)
        print(name, path.stat().st_size)
