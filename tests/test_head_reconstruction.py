"""Run with .venv-head/Scripts/python.exe -m unittest discover -s tests -p test_head_reconstruction.py."""
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask
import numpy as np
from PIL import Image
import trimesh

from head_reconstruction import api
from head_reconstruction.geometry import build_scene, canonical_face, boundary_loops, face_weights
from head_reconstruction.worker import InputError, reconstruct

ROOT = Path(__file__).resolve().parent.parent


class HeadGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, _ = build_scene()
        canonical, faces, _ = canonical_face()
        # Synthetic orthographic view with a deliberate protruding nose.
        points = (canonical - [0, 0, 4]) * [.035, -.035, -.035] + [.5, .5, 0]
        points[4, 2] -= .025
        cls.changed, _ = build_scene(np.full((512, 512, 3), 120, np.uint8), points)
        cls.weights = face_weights(canonical, faces, boundary_loops(faces)[0])

    def test_only_face_interior_changes(self):
        for name in self.base.geometry:
            a, b = self.base.geometry[name], self.changed.geometry[name]
            np.testing.assert_array_equal(a.faces, b.faces)
            if name != "face":
                np.testing.assert_array_equal(a.vertices, b.vertices)
                np.testing.assert_array_equal(a.visual.material.baseColorFactor, b.visual.material.baseColorFactor)
        a, b = self.base.geometry["face"].vertices, self.changed.geometry["face"].vertices
        np.testing.assert_array_equal(a[self.weights == 0], b[self.weights == 0])
        self.assertGreater(np.max(np.abs(a - b)), .001)

    def test_face_and_scalp_join_without_gap(self):
        canonical, faces, _ = canonical_face()
        outer = boundary_loops(faces)[0]
        face = self.changed.geometry["face"].vertices[outer]
        scalp = self.changed.geometry["fixed_scalp"].vertices[:len(outer)]
        np.testing.assert_array_equal(face, scalp)

    def test_glb_contains_normals_texture_and_named_parts(self):
        data = self.changed.export(file_type="glb", include_normals=True)
        self.assertEqual(data[:4], b"glTF")
        length = struct.unpack("<I", data[12:16])[0]
        gltf = json.loads(data[20:20+length])
        self.assertTrue(all("NORMAL" in mesh["primitives"][0]["attributes"] for mesh in gltf["meshes"]))
        self.assertTrue(gltf["images"])
        scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene", process=False)
        self.assertEqual(set(scene.geometry), set(self.changed.geometry))


class HeadApiTests(unittest.TestCase):
    def setUp(self):
        app = Flask(__name__)
        app.register_blueprint(api.head_api)
        self.client = app.test_client()

    def test_missing_upload(self):
        with patch.object(api, "readiness", return_value=True):
            response = self.client.post("/api/head/reconstruct")
        self.assertEqual(response.status_code, 400)

    def test_environment_unavailable(self):
        with patch.object(api, "readiness", return_value=False):
            response = self.client.post("/api/head/reconstruct")
        self.assertEqual(response.status_code, 503)

    def test_arbitrary_model_file_is_not_served(self):
        self.assertEqual(self.client.get("/api/head/models/worker.py").status_code, 404)
        self.assertEqual(self.client.get("/api/head/models/..%2F..%2FREADME.md").status_code, 404)

    def test_busy_request_rejected(self):
        api._slot.acquire()
        try:
            with patch.object(api, "readiness", return_value=True):
                response = self.client.post("/api/head/reconstruct", data={"image": (io.BytesIO(b"test"), "a.jpg")})
            self.assertEqual(response.status_code, 429)
        finally:
            api._slot.release()

    def test_real_inference_and_download(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(api, "OUTPUT", Path(folder)):
            photo = ROOT / "frontend/public/examples/regions/nose-result.jpg"
            with photo.open("rb") as uploaded:
                response = self.client.post("/api/head/reconstruct", data={"image": (uploaded, "face.jpg")})
            self.assertEqual(response.status_code, 200, response.json)
            model = self.client.get(response.json["download_url"])
            self.assertEqual(model.status_code, 200)
            self.assertEqual(model.data[:4], b"glTF")
            self.assertIn("attachment", model.headers["Content-Disposition"])
            model.close()
            self.assertEqual(len(list(Path(folder).iterdir())), 1, "Temporary uploads must be removed")


class HeadInputTests(unittest.TestCase):
    def test_invalid_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaises(InputError):
                reconstruct(path, Path(folder) / "out.glb")

    def test_blank_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blank.png"
            Image.new("RGB", (512, 512), "white").save(path)
            with self.assertRaises(InputError):
                reconstruct(path, Path(folder) / "out.glb")

    def test_multiple_faces(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Image.open(ROOT / "frontend/public/examples/regions/nose-result.jpg")
            image = Image.new("RGB", (1024, 512))
            image.paste(source, (0, 0))
            image.paste(source, (512, 0))
            path = Path(folder) / "two.jpg"
            image.save(path)
            with self.assertRaises(InputError):
                reconstruct(path, Path(folder) / "out.glb")


if __name__ == "__main__":
    unittest.main()
