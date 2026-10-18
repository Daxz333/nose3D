import ast
import io
from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from region_adapter import (ADAPTED_REGIONS, RegionValidationError, clean_mask,
                            enhance_region, replace_region)


def face(open_mouth=True, shift=0):
    labels = np.ones((160, 160), np.uint8)
    cv2.ellipse(labels, (48 + shift, 45), (20, 4), -8, 0, 360, 2, -1)
    cv2.ellipse(labels, (112 + shift, 45), (20, 4), 8, 0, 360, 3, -1)
    cv2.ellipse(labels, (48 + shift, 62), (14, 6), 0, 0, 360, 4, -1)
    cv2.ellipse(labels, (112 + shift, 62), (14, 6), 0, 0, 360, 5, -1)
    cv2.ellipse(labels, (80 + shift, 114), (27, 10), 0, 0, 360, 12, -1)
    labels[115:125, 53 + shift:108 + shift] = 13
    if open_mouth:
        cv2.ellipse(labels, (80 + shift, 115), (23, 5), 0, 0, 360, 11, -1)
    image = np.full((160, 160, 3), (185, 150, 125), np.uint8)
    for label, color in ((2, (35, 25, 20)), (3, (65, 40, 30)),
                         (4, (10, 30, 40)), (5, (20, 30, 50)),
                         (11, (60, 20, 25)), (12, (160, 55, 70)), (13, (170, 60, 75))):
        image[labels == label] = color
    return image, labels


class AdapterTests(unittest.TestCase):
    def test_all_regions_are_deterministic_and_leave_other_features_unchanged(self):
        target, tp = face()
        source, sp = face(shift=4)
        source[sp == 2] = (15, 12, 10)
        source[sp == 11] = (220, 210, 195)
        source[sp == 12] = (210, 40, 80)
        original_source, original_target = source.copy(), target.copy()
        for region, ids in ADAPTED_REGIONS.items():
            with self.subTest(region=region):
                result = replace_region(target, source, tp, sp, region)
                again = replace_region(target, source, tp, sp, region)
                np.testing.assert_array_equal(result.image, again.image)
                np.testing.assert_array_equal(result.image[result.alpha == 0], target[result.alpha == 0])
                protected = ~np.isin(tp, (1,) + ids)
                np.testing.assert_array_equal(result.image[protected], target[protected])
                self.assertGreater(np.count_nonzero(result.alpha), 20)
        np.testing.assert_array_equal(source, original_source)
        np.testing.assert_array_equal(target, original_target)

    def test_closed_mouth_errors_identify_which_image(self):
        opened, op = face()
        closed, cp = face(False)
        for target, source, tp, sp, side in ((opened, closed, op, cp, "来源"),
                                           (closed, opened, cp, op, "目标")):
            with self.assertRaises(RegionValidationError) as caught:
                replace_region(target, source, tp, sp, "mouth")
            self.assertEqual(caught.exception.code, "mouth_state_mismatch")
            self.assertIn(side, str(caught.exception))

    def test_mouth_aperture_mismatch_is_rejected(self):
        target, tp = face()
        source, sp = face()
        sp[sp == 11] = 12
        cv2.ellipse(sp, (80, 115), (23, 23), 0, 0, 360, 11, -1)
        with self.assertRaises(RegionValidationError):
            replace_region(target, source, tp, sp, "mouth")

    def test_mouth_keeps_recipient_lips_and_skin(self):
        target, tp = face()
        source, sp = face(shift=4)
        source[sp == 11] = (230, 220, 210)
        result = replace_region(target, source, tp, sp, "mouth")
        np.testing.assert_array_equal(result.image[tp != 11], target[tp != 11])
        self.assertGreater(result.image[115, 80, 0], 200)

    def test_brows_do_not_swap_or_copy_donor_eyes(self):
        target, tp = face()
        source, sp = face(shift=4)
        source[sp == 2] = (40, 5, 5)
        source[sp == 3] = (5, 50, 5)
        source[np.isin(sp, (4, 5))] = (255, 0, 255)
        result = replace_region(target, source, tp, sp, "brow")
        self.assertEqual(len(result.matrices), 2)
        self.assertGreater(result.image[45, 48, 0], result.image[45, 48, 1])
        self.assertGreater(result.image[45, 112, 1], result.image[45, 112, 0])
        np.testing.assert_array_equal(result.image[np.isin(tp, (4, 5))], target[np.isin(tp, (4, 5))])

    def test_thinner_brow_removes_old_outline_only(self):
        target, tp = face()
        source, sp = face()
        sp[sp == 2] = 1
        cv2.ellipse(sp, (48, 45), (20, 1), -8, 0, 360, 2, -1)
        result = replace_region(target, source, tp, sp, "l_brow")
        self.assertTrue(np.any(result.image[tp == 2] != target[tp == 2]))
        np.testing.assert_array_equal(result.image[tp == 3], target[tp == 3])

    def test_lip_preserves_pigment_and_opposite_lip(self):
        target, tp = face()
        source, sp = face(shift=4)
        source[sp == 12] = (230, 35, 75)
        result = replace_region(target, source, tp, sp, "u_lip")
        output_color = cv2.cvtColor(result.image, cv2.COLOR_RGB2LAB)[106, 80, 1:].astype(float)
        donor_color = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)[sp == 12, 1:].mean(axis=0)
        target_color = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)[tp == 12, 1:].mean(axis=0)
        self.assertLess(np.linalg.norm(output_color - donor_color), np.linalg.norm(output_color - target_color))
        np.testing.assert_array_equal(result.image[np.isin(tp, (11, 13))], target[np.isin(tp, (11, 13))])

    def test_lips_reject_open_closed_mismatch(self):
        target, tp = face()
        source, sp = face(False)
        with self.assertRaises(RegionValidationError) as caught:
            replace_region(target, source, tp, sp, "lip")
        self.assertEqual(caught.exception.code, "lip_state_mismatch")

    def test_missing_one_brow_is_not_silent_partial_success(self):
        target, tp = face()
        source, sp = face()
        sp[sp == 3] = 1
        with self.assertRaisesRegex(RegionValidationError, "右眉"):
            replace_region(target, source, tp, sp, "brow")

    def test_distant_mask_noise_is_removed(self):
        _, parsing = face()
        parsing[5:8, 5:8] = 2
        result = clean_mask(parsing == 2)
        self.assertFalse(result[6, 6])
        self.assertTrue(result[45, 48])

    def test_enhancement_leaves_unselected_pixels_exactly_intact(self):
        image, parsing = face()
        alpha = (parsing == 2).astype(np.float32)
        result = enhance_region(image, np.full((320, 320, 3), 255, np.uint8), alpha)
        np.testing.assert_array_equal(result[alpha == 0], image[alpha == 0])


class ApiTests(unittest.TestCase):
    """Exercise the actual route without loading models or cloud clients."""
    def setUp(self):
        from flask import Flask, jsonify, request, session
        from uuid import uuid1
        import os
        from PIL import Image
        import tempfile
        tree = ast.parse(Path("flask_server.py").read_text(encoding="utf-8"))
        route = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "do_something")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.warper = Mock()
        self.cloud = Mock()
        self.sr = Mock()
        class OutputPaths:
            join = staticmethod(lambda *parts: os.path.join(self.folder.name, *parts[1:]))
            basename = staticmethod(os.path.basename)
            relpath = staticmethod(os.path.relpath)
        app = Flask(__name__)
        app.secret_key = "test"
        env = dict(app=app, request=request, session=session, jsonify=jsonify,
                   REGION_CLASS_MAP={"nose": [10], "mouth": [11]}, REGION_LABELS={"mouth": "口腔"},
                   ADAPTED_REGIONS=ADAPTED_REGIONS, RegionValidationError=RegionValidationError,
                   enhance_region=enhance_region, os=Mock(path=OutputPaths, makedirs=os.makedirs),
                   uuid1=uuid1, io=io, Image=Image, np=np, cv2=cv2, warper=self.warper,
                   sr_model=self.sr, logger=Mock(), qiniu_upload=self.cloud,
                   send_note_with_image=Mock(), save_result_to_folder=Mock(return_value=os.path.join(self.folder.name, "result.jpg")),
                   local_save_path=self.folder.name, to_base64=lambda _: b"result")
        exec(compile(ast.Module(body=[route], type_ignores=[]), "flask_server.py", "exec"), env)
        self.route_env = env
        self.client = app.test_client()

    def upload(self):
        from PIL import Image
        content = io.BytesIO()
        Image.fromarray(face()[0]).save(content, format="PNG")
        return io.BytesIO(content.getvalue()), "face.png"

    def test_state_error_is_returned_and_cloud_is_not_called(self):
        self.warper.warp_adapted.side_effect = RegionValidationError("来源图片需张嘴", "mouth_state_mismatch")
        result = self.client.post("/predict", data={"region": "mouth", "face_image": self.upload(), "source_image": self.upload()}).get_json()
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "mouth_state_mismatch")
        self.assertIn("需张嘴", result["error"])
        self.cloud.assert_not_called()

    def test_missing_target_upload_returns_json(self):
        result = self.client.post("/predict", data={"source_image": self.upload()}).get_json()
        self.assertFalse(result["success"])
        self.assertIn("upload both", result["error"])

    def test_new_and_legacy_source_fields_reach_adapted_path(self):
        from region_adapter import RegionResult
        image, parsing = face()
        self.warper.warp_adapted.return_value = RegionResult(image, (parsing == 11).astype(np.float32), ())
        self.sr.inference.return_value = image
        for field in ("source_image", "nose_image"):
            with self.subTest(field=field):
                result = self.client.post("/predict", data={"region": "mouth", "face_image": self.upload(), field: self.upload()}).get_json()
                self.assertTrue(result["success"])
                self.assertEqual(result["region"], "mouth")
                self.assertEqual(result["result"], "result")
                self.warper.assert_not_called()
                self.cloud.assert_called()

    def test_nose_api_uses_pose_correction_and_reports_it(self):
        from region_adapter import RegionResult
        image, parsing = face()
        self.warper.warp_nose.return_value = RegionResult(image, (parsing == 11).astype(np.float32), ())
        self.sr.inference.return_value = image
        result = self.client.post("/predict", data={"region": "nose", "face_image": self.upload(), "source_image": self.upload()}).get_json()
        self.assertTrue(result["success"])
        self.assertTrue(result["nose_pose_corrected"])
        self.warper.warp_nose.assert_called_once()
        self.warper.assert_not_called()

    def test_prediction_without_session_reaches_validation(self):
        result = self.client.post('/predict').get_json()
        self.assertFalse(result['success'])
        self.assertIn('upload both', result['error'])
        self.assertNotIn('Unauthorized', result['error'])
        self.warper.assert_not_called()


class PublicRoutesTests(unittest.TestCase):
    def setUp(self):
        from flask import Flask, jsonify, redirect
        from api_catalog import describe_endpoints
        tree = ast.parse(Path('flask_server.py').read_text(encoding='utf-8'))
        routes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name in ('index', 'api_endpoints')]
        mappings = {}
        for node in ast.parse(Path('warper.py').read_text(encoding='utf-8')).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in ('REGION_CLASS_MAP', 'REGION_LABELS'):
                    mappings[node.targets[0].id] = ast.literal_eval(node.value)
        app = Flask(__name__)
        env = dict(app=app, jsonify=jsonify, redirect=redirect,
                   describe_endpoints=describe_endpoints, **mappings)
        exec(compile(ast.Module(body=routes, type_ignores=[]), 'flask_server.py', 'exec'), env)
        self.client = app.test_client()
        self.regions = mappings['REGION_CLASS_MAP']

    def test_root_redirects_to_public_workbench(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/app/')

    def test_catalog_is_public_and_matches_prediction_options(self):
        response = self.client.get('/api/endpoints')
        self.assertEqual(response.status_code, 200)
        catalog = response.get_json()
        self.assertEqual(catalog['authentication'], 'none')
        self.assertEqual({r['value'] for r in catalog['regions']}, set(self.regions))
        self.assertEqual(len(catalog['regions']), 11)
        prediction = next(e for e in catalog['endpoints'] if e['path'] == '/predict')
        self.assertEqual(prediction['method'], 'POST')
        parameters = {p['name']: p for p in prediction['parameters']}
        self.assertTrue(parameters['face_image']['required'])
        self.assertTrue(parameters['source_image']['required'])
        self.assertIn('nose_image', parameters['source_image']['description'])
        self.assertEqual(parameters['region']['default'], 'nose')


if __name__ == "__main__":
    unittest.main()
