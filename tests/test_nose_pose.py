import unittest

import cv2
import numpy as np

from nose_pose import (needs_frontalisation, pose_score, row_bounds,
                       transfer_profile_nose, match_surface_detail)
from region_adapter import RegionValidationError, enhance_region


def fixture(profile=False):
    parsing = np.ones((160, 160), np.uint8)
    pts = np.array([(75, 50), (90, 52), (102, 85), (90, 105), (68, 110)]) if profile else np.array([(78, 50), (82, 50), (98, 93), (88, 105), (72, 105)])
    cv2.fillPoly(parsing, [pts], 10)
    parsing[35:45, 35:60] = 4
    parsing[35:45, 100:125] = 5
    parsing[112:118, 60:100] = 12
    image = np.full((160, 160, 3), (190, 160, 135), np.uint8)
    rows, cols = np.indices(parsing.shape)
    image[parsing == 10] = np.column_stack((120 + cols[parsing == 10] % 35,
                                           85 + rows[parsing == 10] % 25,
                                           75 + cols[parsing == 10] % 10))
    return image, parsing


class NosePoseTests(unittest.TestCase):
    def test_pose_selection_in_both_directions(self):
        front = np.array([(40, 40), (120, 40), (80, 80), (60, 110), (100, 110)])
        side = front.copy()
        side[2, 0] = 130
        self.assertTrue(needs_frontalisation(side, front))
        side[2, 0] = 30
        self.assertTrue(needs_frontalisation(side, front))
        self.assertFalse(needs_frontalisation(front, front))
        self.assertFalse(needs_frontalisation(front, side))

    def test_collapsed_eye_landmarks_do_not_divide_by_zero(self):
        side = np.array([(70, 40), (70, 40), (30, 80), (55, 110), (70, 110)])
        self.assertTrue(np.isfinite(pose_score(side)))
        self.assertGreater(abs(pose_score(side)), .45)

    def test_pose_score_is_invariant_to_translation_scale_and_roll(self):
        points = np.array([(40, 40), (120, 40), (110, 80), (60, 110), (100, 110)], np.float32)
        angle = .25
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        transformed = points @ rotation.T * 2 + (30, 75)
        self.assertAlmostEqual(pose_score(points), pose_score(transformed), places=5)

    def test_invalid_landmarks_are_explicit_errors(self):
        with self.assertRaises(RegionValidationError):
            pose_score(np.full((5, 2), np.nan))

    def test_profile_transfer_changes_nose_and_protects_every_other_pixel(self):
        target, tp = fixture()
        source, sp = fixture(True)
        source[sp != 10] = (255, 0, 255)
        for yaw in (-4., -.8, .8, 4.):
            with self.subTest(yaw=yaw):
                result = transfer_profile_nose(target, source, tp, sp, yaw)
                np.testing.assert_array_equal(result.image[tp != 10], target[tp != 10])
                self.assertGreater(np.count_nonzero(result.alpha), .6 * np.count_nonzero(tp == 10))
                self.assertGreater(np.count_nonzero(np.any(result.image != target, axis=2)), 100)
                self.assertLess(np.max(result.image[tp == 10, 0]), 230)
                again = transfer_profile_nose(target, source, tp, sp, yaw)
                np.testing.assert_array_equal(result.image, again.image)

    def test_white_profile_tip_cannot_make_a_hole_in_target_columella(self):
        target, tp = fixture()
        source, sp = fixture(True)
        rows, _, _ = row_bounds(tp == 10)
        bottom = (tp == 10) & (np.indices(tp.shape)[0] >= rows[0] + .91 * np.ptp(rows))
        target[bottom] = (95, 58, 45)
        # Extreme bright profile contour was the cause of the reported defect.
        source[sp == 10] = (250, 245, 240)
        source[60:85, 78:90] = (65, 35, 25)
        result = transfer_profile_nose(target, source, tp, sp, -4)
        np.testing.assert_array_equal(result.image[bottom], target[bottom])
        np.testing.assert_array_equal(result.alpha[bottom], 0)
        self.assertLessEqual(np.abs(result.image.astype(float) - target).max(), 24)

    def test_detail_transfer_bounds_bright_outliers_without_erasing_texture(self):
        target, tp = fixture()
        mask = tp == 10
        donor = target.astype(np.float32).copy()
        donor[75:78, 79:82] = 255
        donor[80:83, 79:82] = 0
        corrected = match_surface_detail(target, donor, mask, 15)
        self.assertLessEqual(np.abs(corrected - target).max(), 24)
        self.assertGreater(np.abs(corrected[mask] - target[mask]).max(), 5)

    def test_enhancement_does_not_reintroduce_columella_hole(self):
        target, tp = fixture()
        source, sp = fixture(True)
        result = transfer_profile_nose(target, source, tp, sp, -4)
        rows, _, _ = row_bounds(tp == 10)
        bottom = (tp == 10) & (np.indices(tp.shape)[0] >= rows[0] + .91 * np.ptp(rows))
        enhanced = enhance_region(result.image, np.full_like(target, 255), result.alpha)
        np.testing.assert_array_equal(enhanced[bottom], target[bottom])

    def test_sparse_parsing_rows_are_interpolated(self):
        _, parsing = fixture(True)
        mask = parsing == 10
        mask[70:72] = False
        rows, left, right = row_bounds(mask)
        self.assertIn(70, rows)
        self.assertTrue(np.isfinite(left).all() and np.isfinite(right).all())

    def test_missing_nose_is_not_reported_as_success(self):
        target, tp = fixture()
        source, sp = fixture(True)
        sp[sp == 10] = 1
        with self.assertRaises(RegionValidationError) as error:
            transfer_profile_nose(target, source, tp, sp, 1.)
        self.assertEqual(error.exception.code, "nose_region_missing")

    def test_profile_crop_at_image_edge_does_not_wrap_negative_indices(self):
        # Import only the crop helper, without loading the detector/model stack.
        import importlib.util
        spec = importlib.util.spec_from_file_location("crop_utils", "face_detector/utils.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        image = np.zeros((160, 160, 3), np.uint8)
        crop, bbox = module.crop_face_square(image, (-15, -10, 70, 140))
        self.assertGreater(crop.size, 0)
        self.assertGreaterEqual(min(bbox), 0)

    def test_enhancement_cannot_change_eyes_lips_or_background(self):
        target, tp = fixture()
        source, sp = fixture(True)
        result = transfer_profile_nose(target, source, tp, sp, 1.)
        enhanced = enhance_region(result.image, np.full((320, 320, 3), 255, np.uint8), result.alpha)
        np.testing.assert_array_equal(enhanced[tp != 10], target[tp != 10])


if __name__ == "__main__":
    unittest.main()
