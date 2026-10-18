"""Offline model smoke check; saves images locally and never uploads results.

Run from the project root: .venv/Scripts/python.exe tests/smoke_regions.py
"""
import json
import os
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

torch_lib = Path(torch.__file__).parent / "lib"
dll_handle = os.add_dll_directory(str(torch_lib)) if hasattr(os, "add_dll_directory") else None
import cv2
import numpy as np
from PIL import Image

from region_adapter import ADAPTED_REGIONS, RegionValidationError, replace_region
from region_adapter import enhance_region
from warper import RegionWarper, DEFAULT_BLEND_CONFIG, get_region_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-legacy", action="store_true", help="Check only adapted regions")
    parser.add_argument("--enhance", action="store_true", help="Also exercise the locally cached GFPGAN models")
    args = parser.parse_args()
    output = Path("outputs/region-adapter-smoke")
    output.mkdir(parents=True, exist_ok=True)
    warper = RegionWarper(
        "pretrained_model/FaceDetector.onnx", "pretrained_model/fp_512.pth",
        "pretrained_model/rvm_resnet50_fp16.onnx", iteration=80,
        learning_rate=.02, decay=.995, threshold=.6,
    )
    assert get_region_config("nose") == DEFAULT_BLEND_CONFIG
    samples = []
    for filename in ("55.jpg", "761.jpg", "44.jpg", "999.jpg", "dlrb.jpg", "dzq.jpg",
                     "jst.jpg", "tl.jpg", "zly.jpg", "66.jpg", "000.jpg", "22.jpg"):
        try:
            image = np.asarray(Image.open(Path("images") / filename).convert("RGB"))
            crop, parsing, _ = warper.get_parsing(image)
            samples.append((filename, image, crop, parsing))
            print(filename, {i: int(np.count_nonzero(parsing == i)) for i in (2, 3, 11, 12, 13)}, flush=True)
        except ValueError as error:
            print(filename, str(error), flush=True)
    report = []
    rows = []
    for region in ADAPTED_REGIONS:
        matched = None
        for target in samples:
            for source in sorted(samples, key=lambda sample: sample[0] != "44.jpg"):
                if target[0] == source[0]:
                    continue
                try:
                    result = replace_region(target[2], source[2], target[3], source[3], region)
                    matched = (target, source, result)
                    break
                except RegionValidationError:
                    continue
            if matched:
                break
        if matched is None:
            raise RuntimeError("No valid sample pair for " + region)
        target, source, result = matched
        # Exercise the public warper path as well as the model parsing pipeline.
        public = warper(target[1], source[1], region=region)
        np.testing.assert_array_equal(public, result.image)
        np.testing.assert_array_equal(result.image[result.alpha == 0], target[2][result.alpha == 0])
        cv2.imwrite(str(output / (region + ".jpg")), result.image[..., ::-1])
        views = [target[2], source[2], result.image,
                 np.repeat((result.alpha * 255).astype(np.uint8)[..., None], 3, axis=2)]
        row = np.concatenate([cv2.resize(view, (256, 256)) for view in views], axis=1)
        cv2.putText(row, region + " | " + target[0] + " <- " + source[0], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        rows.append(row)
        report.append({"region": region, "target": target[0], "source": source[0],
                       "changed_pixels": int(np.count_nonzero(result.alpha)), "passed": True})
        print("PASS", report[-1], flush=True)
    # A much darker donor is an extra stress case, not the sole visual sample.
    target = next(item for item in samples if item[0] == "55.jpg")
    source = next(item for item in samples if item[0] == "761.jpg")
    for region in ("brow", "lip"):
        result = replace_region(target[2], source[2], target[3], source[3], region)
        cv2.imwrite(str(output / (region + "-skin-contrast.jpg")), result.image[..., ::-1])
        report.append({"region": region, "target": target[0], "source": source[0], "stress_case": True, "passed": True})
    if args.enhance:
        from super_resolution import SuperResolutionModel
        sr = SuperResolutionModel("pretrained_model/GFPGANCleanv1-NoCE-C2.pth")
        for region in ("mouth", "brow", "lip"):
            item = next(item for item in report if item["region"] == region)
            target = next(sample for sample in samples if sample[0] == item["target"])
            source = next(sample for sample in samples if sample[0] == item["source"])
            result = replace_region(target[2], source[2], target[3], source[3], region)
            final = enhance_region(result.image, sr.inference(result.image), result.alpha)
            np.testing.assert_array_equal(final[result.alpha == 0], result.image[result.alpha == 0])
            cv2.imwrite(str(output / (region + "-enhanced.jpg")), final[..., ::-1])
            item["enhancement_passed"] = True
            print("PASS", region, "GFPGAN with protected surroundings", flush=True)
    # Legacy paths still run with their original segmentation/configuration.
    for region in (() if args.skip_legacy else ("nose", "eye", "l_eye", "r_eye")):
        result = warper(samples[0][1], samples[1][1], region=region, add_bg=True)
        assert result.shape == (512, 512, 3) and result.dtype == np.uint8
        cv2.imwrite(str(output / (region + ".jpg")), result[..., ::-1])
        report.append({"region": region, "passed": True})
        print("PASS", region, flush=True)
    cv2.imwrite(str(output / "contact-sheet.jpg"), np.concatenate(rows)[..., ::-1])
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
