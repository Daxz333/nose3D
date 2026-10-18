"""Deterministic semantic alignment for brows, lips and the oral cavity.

All coordinates are OpenCV (x, y). Parsing labels, rather than image colour,
define what can be copied, so neighbouring eyes/lips cannot enter the patch.
"""
from dataclasses import dataclass

import cv2
import numpy as np


ADAPTED_REGIONS = {
    "mouth": (11,), "brow": (2, 3), "l_brow": (2,), "r_brow": (3,),
    "lip": (12, 13), "u_lip": (12,), "l_lip": (13,),
}
LABELS = {2: "左眉", 3: "右眉", 11: "口腔", 12: "上唇", 13: "下唇"}


class RegionValidationError(ValueError):
    def __init__(self, message, code="region_not_detected"):
        super().__init__(message)
        self.code = code


@dataclass
class RegionResult:
    image: np.ndarray
    alpha: np.ndarray
    matrices: tuple


def clean_mask(mask):
    """Keep the main component and nearby fragments, discard distant speckles."""
    mask = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask.astype(bool)
    main = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[main, cv2.CC_STAT_AREA]
    near = cv2.dilate((labels == main).astype(np.uint8), kernel(3)) > 0
    keep = [main]
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= max(3, area * .02) and np.any(near & (labels == index)):
            keep.append(index)
    return np.isin(labels, keep)


def kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def required_mask(parsing, ids, side, label):
    mask = clean_mask(np.isin(parsing, ids))
    if np.count_nonzero(mask) < max(8, parsing.size * .00003):
        raise RegionValidationError("%s未检测到%s，请使用无遮挡、清晰的正面人脸照片" % (side, label))
    return mask


def mask_frame(mask):
    points = np.column_stack(np.nonzero(mask)[::-1]).astype(np.float32)
    center = np.median(points, axis=0)
    _, vectors = np.linalg.eigh(np.cov(points.T))
    axis = vectors[:, -1]
    if axis[0] < 0:
        axis = -axis
    # Thin regions must remain approximately horizontal, never flip vertically.
    angle = np.clip(np.arctan2(axis[1], axis[0]), -np.pi / 6, np.pi / 6)
    basis = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    local = (points - center) @ basis
    low, high = np.percentile(local, [2, 98], axis=0)
    center = center + basis @ ((low + high) / 2)
    return center, basis, np.maximum(high - low, 2)


def align_masks(source, target, brow=False):
    sc, sb, ss = mask_frame(source)
    tc, tb, ts = mask_frame(target)
    scales = ts / ss
    if scales[0] < .4 or scales[0] > 2.5:
        raise RegionValidationError("两张图片的部位大小差异过大，请选择姿态和距离相近的照片", "region_shape_mismatch")
    if brow:
        # Match each brow's centre and length; retain donor thickness and arch.
        scales[1] = np.clip(scales[1], .7, 1.4)
    else:
        scales[1] = np.clip(scales[1], .5, 2.)
    linear = tb @ np.diag(scales) @ sb.T
    return np.column_stack((linear, tc - linear @ sc)).astype(np.float32)


def validate_mouth(source, target):
    cavities = []
    for parsing, side in ((source, "来源图片"), (target, "目标图片")):
        cavity = clean_mask(parsing == 11)
        if np.count_nonzero(cavity) < max(12, parsing.size * .00006):
            raise RegionValidationError("%s为闭嘴状态或未检测到口腔；口腔替换需要两张照片都张嘴" % side, "mouth_state_mismatch")
        mouth = required_mask(parsing, (11, 12, 13), side, "嘴巴")
        _, _, extent = mask_frame(mouth)
        _, _, opening = mask_frame(cavity)
        cavities.append((cavity, opening[1] / extent[0]))
    ratio = cavities[0][1] / max(cavities[1][1], .001)
    if not .4 <= ratio <= 2.5:
        raise RegionValidationError("两张照片的张嘴幅度差异过大，请选择张嘴幅度相近的照片", "mouth_state_mismatch")
    return cavities[0][0], cavities[1][0]


def replace_region(target, source, target_parsing, source_parsing, region):
    if region not in ADAPTED_REGIONS:
        raise ValueError("不支持的部位: %s" % region)
    if target.shape != source.shape or target.shape[:2] != target_parsing.shape or source.shape[:2] != source_parsing.shape:
        raise ValueError("人脸切片与解析掩码尺寸不一致")
    brow = "brow" in region
    ids = ADAPTED_REGIONS[region]
    if region in ("lip", "u_lip", "l_lip"):
        minimum = max(12, target_parsing.size * .00006)
        source_open = np.count_nonzero(clean_mask(source_parsing == 11)) >= minimum
        target_open = np.count_nonzero(clean_mask(target_parsing == 11)) >= minimum
        if source_open != target_open:
            raise RegionValidationError("嘴唇替换需要两张照片的张嘴 / 闭嘴状态一致，请重新选择图片", "lip_state_mismatch")
    if region == "mouth":
        sm, tm = validate_mouth(source_parsing, target_parsing)
        parts = [(sm, tm, (11,))]
    else:
        # Match named sides individually. Never align both disconnected brows
        # with one unconstrained nearest-neighbour optimisation.
        parts = [(required_mask(source_parsing, (i,), "来源图片", LABELS[i]),
                  required_mask(target_parsing, (i,), "目标图片", LABELS[i]), (i,)) for i in ids]

    mixed = target.astype(np.float32).copy()
    total_alpha = np.zeros(target.shape[:2], np.float32)
    matrices = []
    size = (target.shape[1], target.shape[0])
    for sm, tm, part_ids in parts:
        matrix = align_masks(sm, tm, brow)
        matrices.append(matrix)
        donor = cv2.warpAffine(source, matrix, size).astype(np.float32)
        donor_labels = cv2.warpAffine(source_parsing, matrix, size, flags=cv2.INTER_NEAREST)
        donor_mask = cv2.warpAffine(sm.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST) > 0
        if region == "mouth":
            # Keep the recipient's lips, skin and mouth outline exactly intact.
            support = tm & donor_mask
            if np.count_nonzero(support) < .5 * np.count_nonzero(tm):
                raise RegionValidationError("口腔形状无法可靠对齐，请选择张嘴幅度相近的照片", "region_shape_mismatch")
            alpha = support.astype(np.float32)
            alpha *= np.clip(cv2.distanceTransform(tm.astype(np.uint8), cv2.DIST_L2, 3) / 1.5, 0, 1)
        else:
            radius = 2 if brow else 1
            union = donor_mask | tm
            support = cv2.dilate(union.astype(np.uint8), kernel(radius)) > 0
            # Only skin and the selected labels may be edited or copied.
            allowed = (1,) + part_ids
            support &= np.isin(target_parsing, allowed)
            valid = np.isin(donor_labels, allowed)
            if brow:
                # Erase exposed old brow when donor is thinner, without copying
                # the donor's neighbouring eye or eyebrow on the other side.
                erase = tm
                base = cv2.inpaint(target, erase.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
                mixed[erase] = base[erase]
                total_alpha[erase] = 1
            support &= valid
            # Correct only the surrounding skin's illumination. Do not match
            # lip/brow averages, which destroys donor pigmentation and texture.
            ring = (cv2.dilate(union.astype(np.uint8), kernel(7)) > 0) & ~union
            ring &= (target_parsing == 1) & (donor_labels == 1)
            brow_opacity = np.ones(support.shape, np.float32)
            if np.count_nonzero(ring) >= 16:
                target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
                donor_lab = cv2.cvtColor(donor.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
                skin_source = np.median(donor_lab[ring], axis=0)
                skin_target = np.median(target_lab[ring], axis=0)
                if brow:
                    # Parsing includes skin between hairs. Extract a soft hair
                    # matte from contrast to donor skin, avoiding solid blocks.
                    darkness = skin_source[0] - donor_lab[..., 0]
                    brow_opacity[donor_mask] = np.clip((darkness[donor_mask] - 3) / 25, 0, 1)
                shift = skin_target - skin_source
                shift[1:] = np.clip(shift[1:], -12, 12)
                corrected = donor_lab + shift
                # Lip/hair brightness has a different relation to skin across
                # people. Match feature luminance separately, retaining donor
                # chroma and high-frequency texture instead of whitening it
                # with the surrounding skin's potentially much larger shift.
                quantile = 25 if brow else 50
                feature_shift = (np.percentile(target_lab[..., 0][tm], quantile)
                                 - np.percentile(donor_lab[..., 0][donor_mask], quantile))
                feature_light = donor_lab[..., 0] + feature_shift
                weight = brow_opacity if brow else np.ones(support.shape, np.float32)
                corrected[..., 0][donor_mask] = (
                    feature_light * weight + corrected[..., 0] * (1 - weight)
                )[donor_mask]
                donor_lab = np.clip(corrected, 0, 255).astype(np.uint8)
                donor = cv2.cvtColor(donor_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
            alpha = cv2.GaussianBlur(support.astype(np.float32), (5, 5), 0)
            alpha *= support
            # Feather the semantic boundary as well as the surrounding skin;
            # a hard alpha=1 at the label edge produces visible cutout seams.
            edge = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
            alpha *= np.clip(edge / (3 if brow else 2), 0, 1)
            if brow:
                matte = donor_mask.astype(np.float32) * brow_opacity
                alpha = cv2.GaussianBlur(matte, (5, 5), 0) * support
            else:
                feature_edge = cv2.distanceTransform(donor_mask.astype(np.uint8), cv2.DIST_L2, 3)
                alpha *= np.clip(feature_edge / 3, 0, 1)
        mixed = donor * alpha[..., None] + mixed * (1 - alpha[..., None])
        total_alpha = np.maximum(total_alpha, alpha)
    return RegionResult(np.clip(np.rint(mixed), 0, 255).astype(np.uint8), total_alpha, tuple(matrices))


def enhance_region(result, enhanced, alpha):
    """Keep enhancement within the replacement area, including single sides."""
    enhanced = cv2.resize(enhanced, (result.shape[1], result.shape[0]))
    weight = np.asarray(alpha, np.float32)[..., None] * .3
    return np.clip(enhanced * weight + result * (1 - weight), 0, 255).astype(np.uint8)
