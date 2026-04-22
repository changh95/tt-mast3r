#!/usr/bin/env python3
"""Generate media/ demo: run the ttnn port on one CO3D apple pair,
save the preprocessed source images, render the predicted 3D point
cloud, and dump pose-accuracy numbers vs CO3D GT.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_TT_METAL_ROOT = "/home/ttuser/experiments/medgemma/tt-metal"
sys.path.insert(0, _TT_METAL_ROOT)
sys.path.insert(1, os.path.join(_TT_METAL_ROOT, "ttnn"))
sys.path.insert(2, os.path.join(_TT_METAL_ROOT, "tools"))
os.chdir(_TT_METAL_ROOT)

_MAST3R_ROOT = "/home/ttuser/experiments/tt-mast3r/models/demos/mast3r"
sys.path.insert(0, _MAST3R_ROOT)
_REPO_ROOT = "/home/ttuser/experiments/tt-mast3r"
sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import ttnn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reference.torch_dust3r import load_checkpoint
from tt.ttnn_dust3r import dust3r_forward, _make_positions_cached
from eval_mast3r import (
    load_image_for_dust3r, activate_pts3d, activate_conf,
    pair_viewer_pose, co3d_gt_K, co3d_to_opencv_extrinsic,
    extrinsic_to_rel, rel_rotation_angle_deg, rel_translation_angle_deg,
    load_co3d_annotations,
)


def tensor_to_rgb_img(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) in [-1, 1] → (H, W, 3) uint8 RGB."""
    arr = t[0].permute(1, 2, 0).numpy()
    arr = np.clip((arr * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return arr


def render_pointcloud(pts_list, rgb_list, conf_list, out_path: Path,
                      conf_pct: float = 85.0):
    """Render the union of the predicted pointmaps as a 3D scatter,
    coloured by per-pixel RGB from the source images, viewed from an
    off-axis angle. Saves to `out_path` as a 12" × 6" PNG.
    """
    pts = np.concatenate([p.reshape(-1, 3) for p in pts_list], axis=0)
    rgb = np.concatenate([c.reshape(-1, 3) for c in rgb_list], axis=0) / 255.0
    conf = np.concatenate([c.reshape(-1) for c in conf_list], axis=0)

    thr = np.percentile(conf, 100.0 - conf_pct)
    keep = conf >= thr
    pts = pts[keep]
    rgb = rgb[keep]
    # Downsample for speed.
    if pts.shape[0] > 40000:
        idx = np.random.default_rng(0).choice(pts.shape[0], 40000, replace=False)
        pts, rgb = pts[idx], rgb[idx]

    # Centre + scale so the scene fits a unit cube.
    pts = pts - pts.mean(axis=0, keepdims=True)
    pts = pts / max(np.percentile(np.abs(pts), 95), 1e-6)

    fig = plt.figure(figsize=(12, 6))
    for i, (elev, azim, title) in enumerate([
        (20, -60, "view A"), (20, 30, "view B"),
    ], start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=rgb, s=2, marker=".", alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.1,
                facecolor="white")
    plt.close(fig)


def main():
    co3d_root = Path("/home/ttuser/experiments/vggt/co3d_data")
    category = "apple"
    seq = "540_79043_153212"
    i, j = 0, 40   # GT rot ≈ 64°, mid-range baseline — representative of the
                   # pairs exercised in the 12-pair aggregate eval.

    media = Path(_REPO_ROOT) / "media"
    media.mkdir(parents=True, exist_ok=True)

    print(f"# loading CO3D annotations: {category} / {seq}")
    by_seq = load_co3d_annotations(co3d_root, category)
    anns = by_seq[seq]
    ann_i, ann_j = anns[i], anns[j]

    img_i, pp_i = load_image_for_dust3r(co3d_root / ann_i["image"]["path"], 512)
    img_j, pp_j = load_image_for_dust3r(co3d_root / ann_j["image"]["path"], 512)

    # Save preprocessed inputs.
    Image.fromarray(tensor_to_rgb_img(img_i)).save(media / "source_1.png")
    Image.fromarray(tensor_to_rgb_img(img_j)).save(media / "source_2.png")
    print(f"# saved media/source_1.png, media/source_2.png")

    device = ttnn.open_device(device_id=0, l1_small_size=32 * 1024)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    try:
        state = load_checkpoint()
        print("# running ttnn port forward (i, j) and (j, i)")
        # Warm the program cache.
        _ = dust3r_forward(img_i, img_j, state, device)

        tt_ii, tt_ji = dust3r_forward(img_i, img_j, state, device)
        tt_jj, tt_ij = dust3r_forward(img_j, img_i, state, device)

        pts_ii = activate_pts3d(tt_ii)[0].cpu().numpy()
        pts_ji = activate_pts3d(tt_ji)[0].cpu().numpy()
        conf_ii = activate_conf(tt_ii)[0].cpu().numpy()
        conf_ji = activate_conf(tt_ji)[0].cpu().numpy()

        rgb_i = tensor_to_rgb_img(img_i)
        rgb_j = tensor_to_rgb_img(img_j)

        # Point cloud render — use both heads' pointmaps (both in view-i frame)
        # coloured by their source pixels.
        print("# rendering point cloud → media/output.png")
        render_pointcloud(
            pts_list=[pts_ii, pts_ji],
            rgb_list=[rgb_i, rgb_j],
            conf_list=[conf_ii, conf_ji],
            out_path=media / "output.png",
        )

        # PairViewer pose recovery (both estimated-focal and known-focal).
        print("# running PairViewer pose recovery")
        K_j_gt = co3d_gt_K(ann_j["viewpoint"], pp_j)
        est = pair_viewer_pose(tt_ii, tt_ji, tt_jj, tt_ij, img_size=512)
        gt_ = pair_viewer_pose(tt_ii, tt_ji, tt_jj, tt_ij, img_size=512,
                               K_j_override=K_j_gt)

        gt_extri_i = co3d_to_opencv_extrinsic(ann_i["viewpoint"])
        gt_extri_j = co3d_to_opencv_extrinsic(ann_j["viewpoint"])
        gt_R, gt_t = extrinsic_to_rel(gt_extri_i, gt_extri_j)

        lines = [
            f"# tt-mast3r demo pair — CO3Dv2 `{category}` scene `{seq}` frames ({i}, {j})",
            f"# source_1.png = frame {i}, source_2.png = frame {j}  (both centre-padded to 512×512)",
            "",
            "## PairViewer pose (port output) vs CO3D GT relative pose",
            "",
            "| variant | rot err | tr-dir err | focal_i | focal_j |",
            "|---|---:|---:|---:|---:|",
        ]
        for tag, recovered in (("estimated focal", est), ("known focal (CO3D K)", gt_)):
            if recovered is None:
                lines.append(f"| {tag} | PnP FAILED | — | — | — |")
                continue
            R, t, f_i, f_j = recovered
            r_err = rel_rotation_angle_deg(R, gt_R)
            t_err = rel_translation_angle_deg(t, gt_t)
            lines.append(
                f"| {tag} | **{r_err:.2f}°** | **{t_err:.2f}°** | "
                f"{f_i:.1f} px | {f_j:.1f} px |"
            )
        lines.append("")
        lines.append(f"CO3D GT focal (view j, 512 px grid): {K_j_gt[0,0]:.1f}")
        lines.append(f"CO3D GT relative rotation magnitude: {rel_rotation_angle_deg(gt_R, np.eye(3)):.2f}°")
        block = "\n".join(lines)
        (media / "pose_accuracy.md").write_text(block + "\n")
        print("\n" + block)
        print(f"\n# saved media/pose_accuracy.md")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
