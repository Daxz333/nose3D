"""Generate actual, enhanced interface examples locally; never upload or notify.

Run from the project root: .venv/Scripts/python.exe tests/generate_api_examples.py
"""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
torch_lib = Path(torch.__file__).parent / 'lib'
dll_handle = os.add_dll_directory(str(torch_lib)) if hasattr(os, 'add_dll_directory') else None
import cv2
import numpy as np
from PIL import Image

from region_adapter import ADAPTED_REGIONS, enhance_region
from warper import REGION_CLASS_MAP, REGION_LABELS, RegionWarper
from super_resolution import SuperResolutionModel


CASES = {
    'nose': ('44.jpg', '90_003.jpg'),
    'eye': ('55.jpg', '761.jpg'),
    'l_eye': ('55.jpg', '761.jpg'),
    'r_eye': ('55.jpg', '761.jpg'),
    'brow': ('55.jpg', '44.jpg'),
    'l_brow': ('55.jpg', '44.jpg'),
    'r_brow': ('55.jpg', '44.jpg'),
    'lip': ('55.jpg', '44.jpg'),
    'u_lip': ('55.jpg', '44.jpg'),
    'l_lip': ('55.jpg', '44.jpg'),
    'mouth': ('dzq.jpg', 'jst.jpg'),
}


def box_for(parsing, ids):
    y, x = np.nonzero(np.isin(parsing, ids))
    if not len(x):
        raise ValueError('Missing example region')
    padding = 8
    return [max(int(x.min()) - padding, 0), max(int(y.min()) - padding, 0),
            min(int(x.max()) + padding + 1, parsing.shape[1]),
            min(int(y.max()) + padding + 1, parsing.shape[0])]


def detail_box(box):
    x0, y0, x1, y1 = box
    size = min(512, max(x1 - x0, y1 - y0) + 24)
    left = int(np.clip((x0 + x1 - size) // 2, 0, 512 - size))
    top = int(np.clip((y0 + y1 - size) // 2, 0, 512 - size))
    return [left, top, left + size, top + size]


def main():
    os.chdir(ROOT)
    assets = ROOT / 'frontend/public/examples/regions'
    assets.mkdir(parents=True, exist_ok=True)
    warper = RegionWarper('pretrained_model/FaceDetector.onnx', 'pretrained_model/fp_512.pth',
                          'pretrained_model/rvm_resnet50_fp16.onnx', iteration=80,
                          learning_rate=.02, decay=.995, threshold=.6)
    sr = SuperResolutionModel('pretrained_model/GFPGANCleanv1-NoCE-C2.pth')
    samples = {}
    for name in dict.fromkeys(name for pair in CASES.values() for name in pair):
        raw = np.asarray(Image.open(ROOT / 'images' / name).convert('RGB'))
        crop, parsing, _ = warper.get_parsing(raw)
        samples[name] = (raw, crop, parsing)
        shutil.copyfile(ROOT / 'images' / name, assets / ('input-' + name))

    manifest = {}
    for region, (target_name, source_name) in CASES.items():
        target, source = samples[target_name], samples[source_name]
        if region == 'nose':
            replaced = warper.warp_nose(target[0], source[0], return_details=True, add_bg=True)
            assert replaced.alpha is not None, 'Profile example must use pose correction'
        elif region in ADAPTED_REGIONS:
            replaced = warper.warp_adapted(target[0], source[0], region, return_details=True)
        else:
            replaced = None
        if replaced is not None:
            final = enhance_region(replaced.image, sr.inference(replaced.image), replaced.alpha)
            np.testing.assert_array_equal(final[replaced.alpha == 0], target[1][replaced.alpha == 0])
        else:
            raw_result = warper(target[0], source[0], region=region, add_bg=True)
            enhanced = cv2.resize(sr.inference(raw_result), (raw_result.shape[1], raw_result.shape[0]))
            final = (enhanced * .3 + raw_result * .7).astype('uint8')
        assert final.shape == (512, 512, 3)
        changed = int(np.count_nonzero(np.any(final != target[1], axis=2)))
        assert changed > 100, 'Example must include an actual replacement'
        target_box = box_for(target[2], REGION_CLASS_MAP[region])
        source_box = box_for(source[2], REGION_CLASS_MAP[region])
        views = {}
        for role, array, box in (('target', target[1], target_box),
                                  ('source', source[1], source_box), ('result', final, target_box)):
            name = region + '-' + role + '.jpg'
            Image.fromarray(array).save(assets / name, quality=95)
            detail = region + '-' + role + '-detail.jpg'
            Image.fromarray(array).crop(detail_box(box)).save(assets / detail, quality=95)
            views[role] = {'image': name, 'detail': detail, 'box': box}
        manifest[region] = {
            'region': region, 'label': REGION_LABELS[region],
            'targetInput': 'input-' + target_name, 'sourceInput': 'input-' + source_name,
            'views': views, 'changedPixels': changed, 'enhanced': True,
        }
        print('PASS example', region, target_name, '<-', source_name, 'changed', changed, flush=True)

    assert set(manifest) == set(REGION_CLASS_MAP)
    destination = ROOT / 'frontend/src/constants/regionExamples.json'
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved all', len(manifest), 'examples:', destination, flush=True)


if __name__ == '__main__':
    main()
