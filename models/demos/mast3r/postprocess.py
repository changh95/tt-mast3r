"""Host-side pre/post-processing for the DUSt3R port (no ttnn, no device).

These helpers used to live in the CO3D eval harness (``eval_mast3r.py``); they
moved here so the serving app -- and the eval scripts -- import them from the
package instead of from a script. Semantics are unchanged.

* :func:`preprocess_image`     pad-to-square (gray 128) -> bicubic 512x512 -> [-1, 1]
* :func:`activate_pts3d`       DUSt3R ``depth_mode='exp'`` activation of the xyz channels
* :func:`activate_conf`        ``conf_mode=('exp', 1, inf)`` -> ``1 + exp(c)``
* :func:`pair_viewer_pose`     DUSt3R PairViewer 2-view pose (focal median vote + PnP-RANSAC)

``cv2`` (opencv-python-headless) is imported lazily inside :func:`pnp_pose`, so
importing this module never requires OpenCV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

#: The only input geometry the port was validated for (32x32 tokens, B=1 pair).
IMG_SIZE = 512
PAD_GRAY = (128, 128, 128)


# ---------- image preprocessing for DUSt3R (512x512) ----------

def preprocess_image(img: Image.Image, size: int = IMG_SIZE):
    """Pad to square (gray 128) then resize to ``size x size``, normalise to
    [-1, 1]. Preserves image content; PCC port-vs-ref stays > 0.99 xyz on real
    CO3D apple pairs. Returns the ``(1, 3, size, size)`` float32 tensor plus a
    preprocess record that lets callers project intrinsics into the padded grid
    (and map results back to the original image).
    """
    img = img.convert("RGB")
    W, H = img.size
    s = max(W, H)
    canvas = Image.new("RGB", (s, s), PAD_GRAY)
    offx, offy = (s - W) // 2, (s - H) // 2
    canvas.paste(img, (offx, offy))
    canvas = canvas.resize((size, size), Image.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, {"orig_W": W, "orig_H": H, "offx": offx, "offy": offy,
                    "pad_side": s, "out_size": size}


def load_image_for_dust3r(path, size: int = IMG_SIZE):
    """File-path convenience wrapper around :func:`preprocess_image` (eval scripts)."""
    with Image.open(Path(path)) as im:
        return preprocess_image(im, size)


def intrinsics_to_canonical(K: np.ndarray, preproc: dict) -> np.ndarray:
    """Project a pinhole ``K`` given in ORIGINAL image pixels into the padded,
    resized ``out_size x out_size`` grid the network saw (same arithmetic as the
    CO3D/ETH3D GT-K projection in the eval scripts).
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    scale = preproc["out_size"] / preproc["pad_side"]
    fx, fy = K[0, 0] * scale, K[1, 1] * scale
    cx = (K[0, 2] + preproc["offx"]) * scale
    cy = (K[1, 2] + preproc["offy"]) * scale
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ---------- DUSt3R output postprocess ----------

def activate_pts3d(raw: torch.Tensor) -> torch.Tensor:
    """Apply DUSt3R's depth_mode='exp' activation to the 3 xyz channels.

    raw: (B, 4, H, W) -- channels 0..2 are xyz (pre-activation), 3 is conf.
    Returns (B, H, W, 3) float pointmap. Both heads express points in the
    camera-1 frame (DUSt3R convention).
    """
    xyz = raw[:, :3].permute(0, 2, 3, 1).contiguous()   # (B, H, W, 3)
    d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return xyz * (torch.expm1(d) / d)


def activate_conf(raw: torch.Tensor) -> torch.Tensor:
    """conf_mode = ('exp', 1, +inf) -> 1 + exp(c). Returns (B, H, W)."""
    return 1.0 + raw[:, 3].exp()


# ---------- pose recovery (DUSt3R PairViewer, 2-view global aligner) ----------

def estimate_focal(pts3d: np.ndarray, pp: np.ndarray) -> float:
    """Median-vote focal estimator matching DUSt3R's `estimate_focal_knowing_depth`
    (focal_mode='median'). `pts3d` is an (H, W, 3) pointmap in the camera's own
    frame; `pp` is the (2,) principal point in pixel coords (typically image
    centre). Pixel (u, v) with 3D (X, Y, Z) gives f = |(u - cx) * Z / X| and
    similarly in y -- we take the median over all pixels and assume fx == fy.
    """
    H, W, _ = pts3d.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixels = np.stack([xs, ys], axis=-1).astype(np.float64) - pp[None, None]  # (H, W, 2)
    z = pts3d[..., 2]
    xy = pts3d[..., :2]
    # f = pixels * z / xy, with |xy| floor to avoid exploding at the optical axis.
    denom = np.where(np.abs(xy) < 1e-6, np.sign(xy) * 1e-6 + 1e-12, xy)
    f_votes = pixels * z[..., None] / denom  # (H, W, 2)
    f_votes = f_votes[np.isfinite(f_votes)]
    f = float(np.median(np.abs(f_votes)))
    # Clamp to a sane focal range (same convention as DUSt3R: 0.5x -- 4x image size).
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))   # ~0.866 x max side
    return float(np.clip(f, 0.2 * focal_base, 8.0 * focal_base))


def pnp_pose(pts3d: np.ndarray, K: np.ndarray, conf: np.ndarray,
             conf_pct: float = 50.0, reproj_err: float = 5.0):
    """PnP-RANSAC on (pts3d, pixel grid, K). Returns (3, 4) extrinsic or None.

    Needs OpenCV (``opencv-python-headless``); imported here so the package
    imports without it.
    """
    import cv2  # lazy: only the pose path needs OpenCV

    H, W, _ = pts3d.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pts2d = np.stack([xs, ys], axis=-1).astype(np.float64).reshape(-1, 2)
    pts3d_flat = pts3d.reshape(-1, 3).astype(np.float64)
    c = conf.reshape(-1)

    keep = c >= np.percentile(c, 100.0 - conf_pct)
    keep &= np.isfinite(pts3d_flat).all(-1)
    if keep.sum() < 50:
        return None
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        pts3d_flat[keep].reshape(-1, 1, 3),
        pts2d[keep].reshape(-1, 1, 2),
        K, distCoeffs=None,
        iterationsCount=500, reprojectionError=reproj_err,
        confidence=0.9999, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return np.concatenate([R, tvec.reshape(3, 1)], axis=-1)


def pair_viewer_pose(raw_ii, raw_ji, raw_jj, raw_ij, img_size: int,
                     K_j_override: Optional[np.ndarray] = None,
                     conf_pct: float = 50.0):
    """DUSt3R PairViewer: given the 4 pointmaps produced by symmetric forwards
    on (i, j) and (j, i), recover view-j's relative pose in view-i's frame.

    raw_ii = head-1 output of forward(i, j) -- pts3d of view i in view i's frame
    raw_ji = head-2 output of forward(i, j) -- pts3d of view j in view i's frame
    raw_jj = head-1 output of forward(j, i) -- pts3d of view j in view j's frame
    raw_ij = head-2 output of forward(j, i) -- pts3d of view i in view j's frame

    When `K_j_override` is given, skip the DUSt3R focal-estimation head and
    use those calibrated intrinsics (in the ``img_size`` grid) for PnP.
    Returns ``(R (3,3), t (3,), f_i, f_j)`` or ``None`` when PnP fails.
    """
    pts_ii = activate_pts3d(raw_ii)[0].cpu().numpy()
    pts_ji = activate_pts3d(raw_ji)[0].cpu().numpy()
    pts_jj = activate_pts3d(raw_jj)[0].cpu().numpy()
    conf_ji = activate_conf(raw_ji)[0].cpu().numpy()

    pp = np.array([img_size / 2.0, img_size / 2.0], dtype=np.float64)
    f_i = estimate_focal(pts_ii, pp)
    f_j = estimate_focal(pts_jj, pp)

    if K_j_override is not None:
        K_j = np.asarray(K_j_override, dtype=np.float64)
    else:
        K_j = np.array([[f_j, 0, pp[0]], [0, f_j, pp[1]], [0, 0, 1]], dtype=np.float64)
    extri_j = pnp_pose(pts_ji, K_j, conf_ji, conf_pct=conf_pct)
    if extri_j is None:
        return None
    return extri_j[:, :3], extri_j[:, 3], f_i, f_j
