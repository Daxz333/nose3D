"""Check left/right and near-90-degree donors on three frontal recipients.

Uses local model weights; does not call Flask, upload or notify anyone.
"""
import os
from pathlib import Path
import sys
import json
import argparse
import base64
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
dll_handle = os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
import cv2
import numpy as np
from PIL import Image
from nose_pose import needs_frontalisation, pose_score, transfer_profile_nose, row_bounds
from region_adapter import enhance_region


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--cached", action="store_true", help="Use inspect_nose_pose.py fixtures for quick visual iteration")
    argument_parser.add_argument("--api-only", action="store_true", help="Check three real /predict responses without repeating the image matrix")
    args = argument_parser.parse_args()
    out = Path("outputs/nose-pose")
    out.mkdir(parents=True, exist_ok=True)
    warper = sr = None
    if not args.cached:
        from warper import RegionWarper
        from super_resolution import SuperResolutionModel
        warper = RegionWarper("pretrained_model/FaceDetector.onnx", "pretrained_model/fp_512.pth", "pretrained_model/rvm_resnet50_fp16.onnx", iteration=80, learning_rate=.02, decay=.995, threshold=.6)
        sr = SuperResolutionModel("pretrained_model/GFPGANCleanv1-NoCE-C2.pth")
    samples = {}
    targets = ("55", "44", "11")
    sources = ("ce_55", "ce_66", "ce_44", "ce_22", "ce_11", "90_11", "90_22", "90_55", "90_44")
    for name in targets + sources:
        raw = np.asarray(Image.open(Path("images") / (name + ".jpg")).convert("RGB"))
        if args.cached:
            with np.load(out / (name + ".npz")) as fixture:
                crop, parsing, points = fixture["crop"], fixture["parsing"], fixture["points"]
        else:
            crop, parsing, points = warper.get_pose_parsing(raw)
        samples[name] = (raw, crop, parsing, points)
    report, rows = [], []
    for target_name in (() if args.api_only else targets):
        for source_name in sources:
            target, source = samples[target_name], samples[source_name]
            assert needs_frontalisation(source[3], target[3]), (target_name, source_name)
            result = transfer_profile_nose(target[1], source[1], target[2], source[2], pose_score(source[3]))
            if warper:
                public = warper.warp_nose(target[0], source[0], return_details=True, add_bg=True)
                np.testing.assert_array_equal(public.image, result.image)
                np.testing.assert_array_equal(warper(target[0], source[0], region="nose"), result.image)
                final = enhance_region(result.image, sr.inference(result.image), result.alpha)
            else:
                final = result.image
            protected = target[2] != 10
            np.testing.assert_array_equal(final[protected], target[1][protected])
            nose_mask = target[2] == 10
            nose_rows, _, _ = row_bounds(nose_mask)
            lower_anatomy = nose_mask & (np.indices(nose_mask.shape)[0] >= nose_rows[0] + .91 * np.ptp(nose_rows))
            np.testing.assert_array_equal(final[lower_anatomy], target[1][lower_anatomy])
            max_delta = int(np.abs(result.image.astype(np.int16) - target[1].astype(np.int16)).max())
            assert max_delta <= 24, "Remapped source produced an unbounded brightness artifact"
            changed = int(np.count_nonzero(np.any(final != target[1], axis=2)))
            assert changed > 100, "Returned original image without replacing nose"
            cv2.imwrite(str(out / (target_name + "-from-" + source_name + ".jpg")), final[..., ::-1])
            entry = dict(target=target_name, source=source_name, source_pose=pose_score(source[3]), changed_pixels=changed, passed=True, enhanced=sr is not None, lower_anatomy_preserved=True, max_surface_delta=max_delta)
            report.append(entry)
            print("PASS", entry, flush=True)
            if target_name == "55":
                row = np.concatenate([cv2.resize(item, (256, 256)) for item in (target[1], source[1], final)], axis=1)
                cv2.putText(row, target_name + " <- " + source_name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
                rows.append(row)
    if warper:
        # Call the real route with real inference and JPEG encoding while
        # replacing cloud upload/notification clients with test doubles.
        from test_region_adapter import ApiTests
        harness = ApiTests("test_nose_api_uses_pose_correction_and_reports_it")
        harness.setUp()
        api_report = []
        try:
            harness.route_env["warper"] = warper
            harness.route_env["sr_model"] = sr
            harness.route_env["to_base64"] = lambda image: base64.b64encode(cv2.imencode(".jpg", image)[1].tobytes())
            def upload(image):
                stream = io.BytesIO()
                Image.fromarray(image).save(stream, format="PNG")
                stream.seek(0)
                return stream, "fixture.png"
            for source_name in ("ce_55", "90_11", "90_44"):
                response = harness.client.post("/predict", data={
                    "region": "nose", "face_image": upload(samples["55"][0]),
                    "source_image": upload(samples[source_name][0]),
                }).get_json()
                assert response["success"] and response["nose_pose_corrected"], response
                decoded = cv2.imdecode(np.frombuffer(base64.b64decode(response["result"]), np.uint8), cv2.IMREAD_COLOR)
                assert decoded.shape == (512, 512, 3)
                api_report.append(dict(source=source_name, success=True, jpeg_shape=list(decoded.shape)))
                print("PASS /predict real JPEG", source_name, flush=True)
        finally:
            harness.doCleanups()
        (out / "api-report.json").write_text(json.dumps(api_report, indent=2), encoding="utf-8")
        # Verify frontal recipients/donors keep the existing nose branch.
        if not args.api_only:
            frontal = warper.warp_nose(samples["55"][0], samples["44"][0], return_details=True, add_bg=True)
            assert frontal.alpha is None and frontal.image.shape == (512, 512, 3)
            print("PASS frontal-to-frontal legacy branch", flush=True)
    if rows:
        cv2.imwrite(str(out / "results.jpg"), np.concatenate(rows)[..., ::-1])
    if report:
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
