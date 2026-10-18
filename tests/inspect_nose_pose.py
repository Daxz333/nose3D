"""Local diagnostics for the available frontal/profile fixtures."""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
dll_handle = os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
import cv2
import numpy as np
from PIL import Image
from face_detector import FaceDetector
from face_detector.utils import resize_image_with_crop_or_pad, crop_face_square
from segmentation import FaceParsing

detector = FaceDetector("pretrained_model/FaceDetector.onnx")
parser = FaceParsing("pretrained_model/fp_512.pth")
out = Path("outputs/nose-pose")
out.mkdir(parents=True, exist_ok=True)
rows = []
for name in ("55.jpg", "44.jpg", "11.jpg", "ce_55.jpg", "ce_66.jpg", "ce_44.jpg", "ce_22.jpg", "ce_11.jpg", "90_11.jpg", "90_22.jpg", "90_55.jpg", "90_44.jpg"):
    try:
        image = resize_image_with_crop_or_pad(np.asarray(Image.open(Path("images") / name).convert("RGB")), 640)
        bbox, prob, points = detector.inference(image, return_landmarks=True)
        eyes = points[1] - points[0]
        yaw = 2 * np.dot(points[2] - (points[0] + points[1]) / 2, eyes) / max(np.dot(eyes, eyes), 1)
        crop, _ = crop_face_square(image, bbox)
        crop = resize_image_with_crop_or_pad(crop, 512)
        parsing = parser.parsing_map(crop)
        np.savez(out / (Path(name).stem + ".npz"), crop=crop, parsing=parsing, points=points, yaw=yaw)
        print(name, "yaw", round(float(yaw), 3), "nose", int(np.count_nonzero(parsing == 10)), flush=True)
        mask = np.repeat((parsing == 10)[..., None], 3, axis=2)
        view = np.concatenate([cv2.resize(crop, (256, 256)), cv2.resize((crop * mask).astype(np.uint8), (256, 256))], axis=1)
        cv2.putText(view, name + " yaw %.2f" % yaw, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        rows.append(view)
    except (ValueError, IndexError) as error:
        print(name, error, flush=True)
cv2.imwrite(str(out / "fixtures.jpg"), np.concatenate(rows)[..., ::-1])
