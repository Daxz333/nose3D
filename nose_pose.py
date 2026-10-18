"""Profile-to-frontal 2D nose transfer with bounded semantic remapping.

A single profile does not reveal the far side. The visible surface supplies
bridge/tip detail; recipient nostrils and columella remain an anatomical prior.
This is an approximation, not a recovered 3D nose.
"""
import cv2
import numpy as np

from region_adapter import RegionResult, RegionValidationError, clean_mask


def pose_score(points):
    points = np.asarray(points, np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise RegionValidationError("人脸关键点无效，请重新选择清晰图片", "nose_pose_invalid")
    eye_axis = points[1] - points[0]
    length = float(np.linalg.norm(eye_axis))
    # Floor the inter-eye distance with mouth width: detector eye landmarks
    # can collapse onto one visible eye in a full profile.
    floor = max(float(np.linalg.norm(points[4] - points[3])) * .5, 2.)
    if length < 1:
        axis = np.array([1., 0.], np.float32)
    else:
        axis = eye_axis / length
    offset = float(np.dot(points[2] - (points[0] + points[1]) / 2, axis))
    return float(np.clip(2 * offset / max(length, floor), -4, 4))


def needs_frontalisation(source_points, target_points):
    return abs(pose_score(source_points)) >= .45 and abs(pose_score(target_points)) <= .35


def row_bounds(mask):
    """Interpolate sparse parsing rows; use only their contiguous nose span."""
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) < 8 or np.count_nonzero(mask) < 32:
        raise RegionValidationError("未检测到足够清晰的鼻子区域，请选择鼻子完整可见的图片", "nose_region_missing")
    left, right = [], []
    for y in rows:
        xs = np.flatnonzero(mask[y])
        # Taking the longest run avoids including holes or detached fragments.
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
        run = max(runs, key=len)
        left.append(run[0])
        right.append(run[-1])
    all_rows = np.arange(rows[0], rows[-1] + 1)
    return all_rows, np.interp(all_rows, rows, left), np.interp(all_rows, rows, right)


def smoothstep(low, high, values):
    phase = np.clip((values - low) / (high - low), 0, 1)
    return phase * phase * (3 - 2 * phase)


def anatomical_confidence(target, mask, rows):
    """Do not invent hidden nostrils or stitch profile edges into the septum."""
    height = max(float(rows[-1] - rows[0]), 1.)
    position = (np.arange(mask.shape[0], dtype=np.float32)[:, None] - rows[0]) / height
    top = smoothstep(0, .12, position)
    bottom = 1 - smoothstep(.72, .91, position)
    # Protect visible nostril shadows even if the parsing mask's bottom edge
    # extends below the actual nose. Limit this prior to the lower nose.
    luminance = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = max(9, min(31, int(height / 6) | 1))
    local = cv2.GaussianBlur(luminance, (window, window), 0)
    shadow = smoothstep(8, 25, local - luminance)
    shadow *= smoothstep(.6, .78, position)
    shadow = cv2.GaussianBlur(shadow, (7, 7), 0)
    return (top * bottom * (1 - shadow) * mask).astype(np.float32)


def match_surface_detail(target, donor, support, blur_size):
    """Transfer bounded surface detail, not highlights from the silhouette."""
    valid = support.astype(np.float32)
    density = cv2.GaussianBlur(valid, (blur_size, blur_size), 0)
    difference = (donor - target.astype(np.float32)) * valid[..., None]
    low_frequency = cv2.GaussianBlur(difference, (blur_size, blur_size), 0) / np.maximum(density[..., None], 1e-5)
    residual = difference - low_frequency
    # Avoid pasting a sharp, high-resolution texture patch onto a soft target.
    gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
    target_detail = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    texture = float(np.percentile(np.abs(target_detail[support]), 75))
    sigma = float(np.clip(1.2 - texture / 6, .25, 1.2))
    residual = cv2.GaussianBlur(residual * valid[..., None], (5, 5), sigma)
    # Winsorise the remapping residual. In particular a background-like bright
    # pixel must never become a new white hole in the columella/nostril area.
    limit = float(np.clip(8 + texture * 2, 12, 24))
    residual = np.clip(residual, -limit, limit)
    return np.clip(target.astype(np.float32) + residual, 0, 255)


def transfer_profile_nose(target, source, target_parsing, source_parsing, source_yaw):
    if target.shape != source.shape or target.shape[:2] != target_parsing.shape or source.shape[:2] != source_parsing.shape:
        raise ValueError("人脸切片与解析掩码尺寸不一致")
    target_mask = clean_mask(target_parsing == 10)
    source_mask = clean_mask(source_parsing == 10)
    ty, tl, tr = row_bounds(target_mask)
    sy, sl, sr = row_bounds(source_mask)
    height, width = target_mask.shape
    map_x = np.full((height, width), -1, np.float32)
    map_y = np.full_like(map_x, -1)
    for index, y in enumerate(ty):
        donor_y = sy[0] + index / max(len(ty) - 1, 1) * (sy[-1] - sy[0])
        donor_left = np.interp(donor_y, sy, sl)
        donor_right = np.interp(donor_y, sy, sr)
        x = np.flatnonzero(target_mask[y])
        u = (x - tl[index]) / max(tr[index] - tl[index], 1)
        # The ridge is near the profile silhouette. Mirror its visible side
        # into both wings instead of stretching background into a frontal nose.
        # A cosine ridge has zero derivative at the midline; the old triangular
        # fold created a crease and sampled too close to the profile outline.
        mirrored = .2 + .6 * np.cos(np.pi * (u - .5)) ** 2
        if source_yaw < 0:
            mirrored = 1 - mirrored
        sample = mirrored
        map_x[y, x] = donor_left + sample * (donor_right - donor_left)
        map_y[y, x] = donor_y

    # Normalise by remapped semantic coverage: bilinear samples at the profile
    # outline must not pick up black padding, hair or background.
    coverage = cv2.remap(source_mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
    weighted = cv2.remap(source.astype(np.float32) * source_mask[..., None], map_x, map_y, cv2.INTER_LINEAR)
    donor = weighted / np.maximum(coverage[..., None], 1e-6)
    support = target_mask & (coverage > .5)
    fraction = np.count_nonzero(support) / max(np.count_nonzero(target_mask), 1)
    if fraction < .9:
        raise RegionValidationError("侧鼻可见区域不足，无法可靠对齐到正脸，请更换鼻子完整可见的照片", "nose_coverage_insufficient")

    blur_size = max(9, min(51, int(np.ptp(ty) / 3) | 1))
    donor = match_surface_detail(target, donor, support, blur_size)
    feather = max(2., min(6., float(np.max(tr - tl)) * .08))
    alpha = np.clip(cv2.distanceTransform(target_mask.astype(np.uint8), cv2.DIST_L2, 3) / feather, 0, 1)
    alpha *= support * anatomical_confidence(target, target_mask, ty)
    image = donor * alpha[..., None] + target.astype(np.float32) * (1 - alpha[..., None])
    return RegionResult(np.clip(np.rint(image), 0, 255).astype(np.uint8), alpha.astype(np.float32), ())
