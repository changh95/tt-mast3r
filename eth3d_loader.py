"""Parse ETH3D DSLR (undistorted) scenes in COLMAP format.

One function: `load_eth3d_scene(scene_root)` returns a list of
per-image records `{name, image_path, extrinsic (3,4), K (3,3),
image_hw (H, W)}`, ready to feed the same PairViewer pipeline we
use for CO3D.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


def _quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP stores unit quaternion as (QW, QX, QY, QZ) with rotation
    acting on column vectors: x_cam = R @ x_world + t. Matches OpenCV's
    extrinsic convention, so we just build the rotation matrix and
    stack with t as-is — no axis flips needed (unlike CO3D's Py3D).
    """
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    return R


def _parse_cameras_txt(path: Path) -> dict:
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            W = int(parts[2]); H = int(parts[3])
            params = [float(p) for p in parts[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = params
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params
                K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            else:
                raise ValueError(f"ETH3D undistorted should have PINHOLE, got {model}")
            cams[cam_id] = {"W": W, "H": H, "K": K}
    return cams


def _parse_images_txt(path: Path) -> list:
    """Return list of dicts {name, qvec, tvec, camera_id}.

    `images.txt` has two lines per image — the pose header, then the 2D
    keypoint observations. Only the header is needed.
    """
    out = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw, qx, qy, qz = (float(parts[k]) for k in (1, 2, 3, 4))
        tx, ty, tz = (float(parts[k]) for k in (5, 6, 7))
        camera_id = int(parts[8])
        name = parts[9]
        out.append({"name": name, "qvec": (qw, qx, qy, qz),
                    "tvec": (tx, ty, tz), "camera_id": camera_id})
        i += 1   # skip the 2D-points line
    return out


def load_eth3d_scene(scene_root: Path) -> list:
    """Return list of image records sorted by filename."""
    scene_root = Path(scene_root)
    cal = scene_root / "dslr_calibration_undistorted"
    cams = _parse_cameras_txt(cal / "cameras.txt")
    imgs = _parse_images_txt(cal / "images.txt")
    records = []
    for im in imgs:
        cam = cams[im["camera_id"]]
        R = _quat_to_R(*im["qvec"])
        t = np.array(im["tvec"], dtype=np.float64)
        extri = np.concatenate([R, t[:, None]], axis=-1)
        records.append({
            "name": im["name"],
            # Actual files live under `images/<name>` in the distributed archive.
            "image_path": scene_root / "images" / im["name"],
            "extrinsic": extri,
            "K": cam["K"],
            "image_hw": (cam["H"], cam["W"]),
        })
    records.sort(key=lambda r: r["name"])
    return records


def gt_K_for_preproc(record: dict, preproc: dict) -> np.ndarray:
    """Project ETH3D pixel-space K into the pad-to-square + resize grid
    that `eval_mast3r.load_image_for_dust3r` produces.
    """
    W = preproc["orig_W"]; H = preproc["orig_H"]
    K = record["K"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2] + preproc["offx"], K[1, 2] + preproc["offy"]
    scale = preproc["out_size"] / preproc["pad_side"]
    return np.array([
        [fx * scale, 0,          cx * scale],
        [0,          fy * scale, cy * scale],
        [0, 0, 1],
    ], dtype=np.float64)
