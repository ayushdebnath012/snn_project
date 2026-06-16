"""Numpy augmentations used by the point-cloud dataset loaders."""

from __future__ import annotations

import math

import numpy as np


def _rand_z_rotation() -> np.ndarray:
    theta = float(np.random.uniform(0.0, 2.0 * math.pi))
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _rand_so3_rotation() -> np.ndarray:
    # QR gives an orthonormal basis; flip if needed to keep det=+1.
    q, _ = np.linalg.qr(np.random.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def _rand_tilt_rotation(max_angle: float) -> np.ndarray:
    ax, ay = np.random.uniform(-max_angle, max_angle, size=2)
    cx, sx = math.cos(float(ax)), math.sin(float(ax))
    cy, sy = math.cos(float(ay)), math.sin(float(ay))
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float32,
    )
    ry = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    return (ry @ rx).astype(np.float32)


def _rotation_from_cfg(cfg) -> np.ndarray:
    if getattr(cfg, "aug_rotate_so3", False):
        return _rand_so3_rotation()
    if getattr(cfg, "aug_rotate_z", False):
        rot = _rand_z_rotation()
    else:
        rot = np.eye(3, dtype=np.float32)

    tilt = float(getattr(cfg, "aug_tilt", 0.0))
    if tilt > 0:
        rot = _rand_tilt_rotation(tilt) @ rot
    return rot.astype(np.float32)


def _scale_from_cfg(cfg) -> np.ndarray:
    lo = float(getattr(cfg, "aug_scale_lo", 1.0))
    hi = float(getattr(cfg, "aug_scale_hi", 1.0))
    if lo == 1.0 and hi == 1.0:
        return np.ones((1, 3), dtype=np.float32)
    if getattr(cfg, "aug_anisotropic_scale", False):
        return np.random.uniform(lo, hi, size=(1, 3)).astype(np.float32)
    scale = float(np.random.uniform(lo, hi))
    return np.full((1, 3), scale, dtype=np.float32)


def _elastic_warp(xyz: np.ndarray, cfg) -> np.ndarray:
    if not getattr(cfg, "aug_elastic", False) or len(xyz) == 0:
        return xyz

    strength = float(
        getattr(cfg, "aug_elastic_strength", getattr(cfg, "aug_elastic_sigma", 0.0))
    )
    if strength <= 0:
        return xyz

    anchors = min(max(1, int(getattr(cfg, "aug_elastic_anchors", 12))), len(xyz))
    span = float(np.ptp(xyz, axis=0).max())
    bandwidth = float(getattr(cfg, "aug_elastic_bandwidth", 0.35)) * max(span, 1e-6)
    clip = float(getattr(cfg, "aug_elastic_clip", strength * 2.0))

    choice = np.random.choice(len(xyz), size=anchors, replace=False)
    anchor_xyz = xyz[choice]
    disp = np.random.normal(0.0, strength, size=(anchors, 3)).astype(np.float32)
    disp[:, 2] *= float(getattr(cfg, "aug_elastic_z_scale", 1.0))
    dist2 = ((xyz[:, None, :] - anchor_xyz[None, :, :]) ** 2).sum(axis=-1)
    weights = np.exp(-dist2 / (2.0 * bandwidth * bandwidth)).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    warp = weights @ disp
    return (xyz + np.clip(warp, -clip, clip)).astype(np.float32)


def _apply_xyz_aug(xyz: np.ndarray, cfg) -> np.ndarray:
    out = xyz.astype(np.float32, copy=True)

    rot = _rotation_from_cfg(cfg)
    scale = _scale_from_cfg(cfg)
    out = (out @ rot.T) * scale

    translate = float(getattr(cfg, "aug_translate", 0.0))
    if translate > 0:
        out += np.random.uniform(-translate, translate, size=(1, 3)).astype(np.float32)

    sigma = float(getattr(cfg, "aug_jitter_sigma", 0.0))
    if sigma > 0:
        clip = float(getattr(cfg, "aug_jitter_clip", 0.05))
        noise = np.clip(np.random.normal(0.0, sigma, out.shape), -clip, clip)
        out += noise.astype(np.float32)

    return _elastic_warp(out, cfg)


def _point_dropout(points: np.ndarray, prob: float) -> np.ndarray:
    if prob <= 0 or len(points) == 0:
        return points
    out = points.copy()
    drop = np.random.random(len(out)) < prob
    if np.any(drop):
        keep = np.where(~drop)[0]
        fill = keep[0] if len(keep) else 0
        out[drop] = out[fill]
    return out


def _dropout_sources(length: int, prob: float) -> np.ndarray:
    """Return indices that replace dropped entries with retained entries."""
    source = np.arange(length)
    if prob <= 0 or length == 0:
        return source
    drop = np.random.random(length) < prob
    if np.all(drop):
        drop[np.random.randint(0, length)] = False
    keep = np.where(~drop)[0]
    if np.any(drop):
        source[drop] = np.random.choice(keep, int(drop.sum()), replace=True)
    return source


def augment_slices(slices: np.ndarray, cfg) -> np.ndarray:
    """Augment classification slices while preserving shape."""
    out = slices.astype(np.float32, copy=True)
    flat_xyz = out[..., :3].reshape(-1, 3)
    out[..., :3] = _apply_xyz_aug(flat_xyz, cfg).reshape(out.shape[0], out.shape[1], 3)

    prob = float(getattr(cfg, "aug_point_dropout", 0.0))
    if prob > 0:
        for i in range(out.shape[0]):
            out[i] = _point_dropout(out[i], prob)

    slice_prob = float(getattr(cfg, "aug_slice_dropout", 0.0))
    if slice_prob > 0 and out.shape[0] > 1:
        drop = np.random.random(out.shape[0]) < slice_prob
        if np.any(drop):
            keep = np.where(~drop)[0]
            fill = keep[0] if len(keep) else 0
            out[drop] = out[fill]

    return out


def augment_seg(slices: np.ndarray, pts_features: np.ndarray, cfg,
                labels: np.ndarray = None, sid_arr: np.ndarray = None):
    """Apply shared geometry and label-aligned dropout for segmentation."""
    out_slices = slices.astype(np.float32, copy=True)
    out_pts = pts_features.astype(np.float32, copy=True)
    out_labels = None if labels is None else labels.copy()
    out_sid = None if sid_arr is None else sid_arr.copy()

    rot = _rotation_from_cfg(cfg)
    scale = _scale_from_cfg(cfg)

    translate = float(getattr(cfg, "aug_translate", 0.0))
    shift = (
        np.random.uniform(-translate, translate, size=(1, 3)).astype(np.float32)
        if translate > 0
        else np.zeros((1, 3), dtype=np.float32)
    )

    def transform(xyz: np.ndarray) -> np.ndarray:
        aug = (xyz @ rot.T) * scale + shift
        sigma = float(getattr(cfg, "aug_jitter_sigma", 0.0))
        if sigma > 0:
            clip = float(getattr(cfg, "aug_jitter_clip", 0.05))
            aug += np.clip(np.random.normal(0.0, sigma, aug.shape), -clip, clip).astype(
                np.float32
            )
        return aug.astype(np.float32)

    out_slices[..., :3] = transform(out_slices[..., :3].reshape(-1, 3)).reshape(
        out_slices.shape[0], out_slices.shape[1], 3
    )
    out_pts[:, :3] = transform(out_pts[:, :3])

    if getattr(cfg, "aug_elastic", False):
        slice_xyz = out_slices[..., :3].reshape(-1, 3)
        n_slice_xyz = len(slice_xyz)
        all_xyz = np.concatenate([slice_xyz, out_pts[:, :3]], axis=0)
        all_xyz = _elastic_warp(all_xyz, cfg)
        out_slices[..., :3] = all_xyz[:n_slice_xyz].reshape(
            out_slices.shape[0], out_slices.shape[1], 3
        )
        out_pts[:, :3] = all_xyz[n_slice_xyz:]

    if getattr(cfg, "use_normals", False):
        out_slices[..., 3:6] = out_slices[..., 3:6] @ rot.T
        out_pts[:, 3:6] = out_pts[:, 3:6] @ rot.T
        slice_norm = np.linalg.norm(out_slices[..., 3:6], axis=-1, keepdims=True)
        point_norm = np.linalg.norm(out_pts[:, 3:6], axis=-1, keepdims=True)
        out_slices[..., 3:6] /= np.maximum(slice_norm, 1e-12)
        out_pts[:, 3:6] /= np.maximum(point_norm, 1e-12)

    point_drop = float(getattr(cfg, "aug_point_dropout", 0.0))
    if point_drop > 0:
        for i in range(out_slices.shape[0]):
            out_slices[i] = _point_dropout(out_slices[i], point_drop)
        source = _dropout_sources(len(out_pts), point_drop)
        out_pts = out_pts[source]
        if out_labels is not None:
            out_labels = out_labels[source]
        if out_sid is not None:
            out_sid = out_sid[source]

    slice_drop = float(getattr(cfg, "aug_slice_dropout", 0.0))
    if slice_drop > 0:
        source = _dropout_sources(out_slices.shape[0], slice_drop)
        out_slices = out_slices[source]

    color_drop = float(getattr(cfg, "aug_color_drop", 0.0))
    if (not getattr(cfg, "use_normals", False) and out_pts.shape[1] >= 6
            and color_drop > 0 and np.random.random() < color_drop):
        out_pts[:, 3:6] = 0.0
        if out_slices.shape[-1] >= 6:
            out_slices[..., 3:6] = 0.0

    color_jitter = float(getattr(cfg, "aug_color_jitter", 0.0))
    if (not getattr(cfg, "use_normals", False) and out_pts.shape[1] >= 6
            and color_jitter > 0):
        noise = np.random.normal(0.0, color_jitter, out_pts[:, 3:6].shape).astype(
            np.float32
        )
        out_pts[:, 3:6] = np.clip(out_pts[:, 3:6] + noise, 0.0, 1.0)
        if out_slices.shape[-1] >= 6:
            noise_s = np.random.normal(
                0.0, color_jitter, out_slices[..., 3:6].shape
            ).astype(np.float32)
            out_slices[..., 3:6] = np.clip(out_slices[..., 3:6] + noise_s, 0.0, 1.0)

    return out_slices, out_pts, out_labels, out_sid
