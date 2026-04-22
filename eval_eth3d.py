#!/usr/bin/env python3
"""DUSt3R / MASt3R TT port PairViewer eval on ETH3D DSLR (undistorted).

Scene is a calibrated, COLMAP-format multi-view capture (indoor/outdoor
scene-level content at ~6000×4000, 3:2 aspect). Pad-to-square on 3:2 is
~33 % gray — much less than CO3D apple's 55 % gray at 2.22:1 — which is
the main reason absolute pose accuracy typically improves here.

Usage:
    python3 eval_eth3d.py --scene /path/to/kicker --pairs 8 --device-id 0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_TT_METAL_ROOT = "/home/ttuser/experiments/medgemma/tt-metal"
sys.path.insert(0, _TT_METAL_ROOT)
sys.path.insert(1, os.path.join(_TT_METAL_ROOT, "ttnn"))
os.chdir(_TT_METAL_ROOT)

_REPO = "/home/ttuser/experiments/tt-mast3r"
sys.path.insert(0, _REPO)
sys.path.insert(0, f"{_REPO}/models/demos/mast3r")

import numpy as np
import torch
import ttnn

from reference.torch_dust3r import load_checkpoint, load_dust3r
from tt.ttnn_dust3r import dust3r_forward
from eval_mast3r import (
    load_image_for_dust3r, activate_pts3d, activate_conf, pcc,
    pair_viewer_pose, extrinsic_to_rel, rel_rotation_angle_deg,
    rel_translation_angle_deg, rra_rta_at, auc_deg, depth_rel_error,
    chamfer_distance_norm,
)
from eth3d_loader import load_eth3d_scene, gt_K_for_preproc


def pick_pairs(records: list, n: int, max_rot_deg: float = 60.0) -> list:
    """Pick `n` pairs spread across the scene, filtered to reasonable
    pair-wise rotation (< `max_rot_deg`) so PairViewer isn't asked for
    near-180° flips. First pass: sort by GT rotation, pick quantiles.
    """
    if len(records) < 2:
        return []
    # Score every (i, j) pair by GT rotation magnitude.
    scored = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            R_rel, _ = extrinsic_to_rel(records[i]["extrinsic"], records[j]["extrinsic"])
            ang = rel_rotation_angle_deg(R_rel, np.eye(3))
            if ang < max_rot_deg:
                scored.append((ang, i, j))
    scored.sort()
    if not scored:
        return []
    # Pick `n` pairs spread across the angle distribution.
    step = max(1, len(scored) // n)
    picks = [scored[k] for k in range(0, len(scored), step)][:n]
    return [(i, j, ang) for ang, i, j in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True,
                    help="Path to ETH3D DSLR scene dir (has dslr_calibration_undistorted/, images/).")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--max-rot-deg", type=float, default=60.0)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=512)
    args = ap.parse_args()

    print(f"# loading ETH3D scene: {args.scene.name}")
    records = load_eth3d_scene(args.scene)
    print(f"# {len(records)} images")

    pairs = pick_pairs(records, args.pairs, max_rot_deg=args.max_rot_deg)
    print(f"# picked {len(pairs)} pairs with GT rot < {args.max_rot_deg}°")

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32 * 1024)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    try:
        state = load_checkpoint()
        ref_model = load_dust3r(state).eval()
        print("# warming TT program cache")
        dummy = torch.zeros(1, 3, args.img_size, args.img_size)
        _ = dust3r_forward(dummy, dummy, state, device)

        results = []
        for (i, j, gt_ang) in pairs:
            rec_i = records[i]; rec_j = records[j]
            img_i, pp_i = load_image_for_dust3r(rec_i["image_path"], args.img_size)
            img_j, pp_j = load_image_for_dust3r(rec_j["image_path"], args.img_size)

            with torch.no_grad():
                ref_ii, ref_ji = ref_model(img_i, img_j)
                ref_jj, ref_ij = ref_model(img_j, img_i)
            tt_ii, tt_ji = dust3r_forward(img_i, img_j, state, device)
            tt_jj, tt_ij = dust3r_forward(img_j, img_i, state, device)

            pcc_scores = {
                "h1_xyz": pcc(ref_ii[:, :3], tt_ii[:, :3]),
                "h2_xyz": pcc(ref_ji[:, :3], tt_ji[:, :3]),
                "h1_conf": pcc(ref_ii[:, 3], tt_ii[:, 3]),
                "h2_conf": pcc(ref_ji[:, 3], tt_ji[:, 3]),
            }

            # GT relative pose (world→cam conventions in both stores).
            gt_R, gt_t = extrinsic_to_rel(rec_i["extrinsic"], rec_j["extrinsic"])

            K_j_gt = gt_K_for_preproc(rec_j, pp_j)

            def err(pose):
                if pose is None:
                    return (float("nan"), float("nan"))
                R, t, _, _ = pose
                return (rel_rotation_angle_deg(R, gt_R),
                        rel_translation_angle_deg(t, gt_t))

            ref_est = pair_viewer_pose(ref_ii, ref_ji, ref_jj, ref_ij, args.img_size)
            tt_est = pair_viewer_pose(tt_ii, tt_ji, tt_jj, tt_ij, args.img_size)
            ref_gt = pair_viewer_pose(ref_ii, ref_ji, ref_jj, ref_ij, args.img_size, K_j_gt)
            tt_gt = pair_viewer_pose(tt_ii, tt_ji, tt_jj, tt_ij, args.img_size, K_j_gt)

            r_ref_e, t_ref_e = err(ref_est)
            r_tt_e, t_tt_e = err(tt_est)
            r_ref_g, t_ref_g = err(ref_gt)
            r_tt_g, t_tt_g = err(tt_gt)

            print(f"\n--- pair ({rec_i['name'][-12:]}, {rec_j['name'][-12:]})  "
                  f"GT rot={gt_ang:.1f}°")
            print(f"pcc h1 xyz={pcc_scores['h1_xyz']:.4f} conf={pcc_scores['h1_conf']:.4f}   "
                  f"h2 xyz={pcc_scores['h2_xyz']:.4f} conf={pcc_scores['h2_conf']:.4f}")
            print(f"ref  est  rot={r_ref_e:6.2f}° tr={t_ref_e:6.2f}°    "
                  f"gt  rot={r_ref_g:6.2f}° tr={t_ref_g:6.2f}°")
            print(f"port est  rot={r_tt_e:6.2f}° tr={t_tt_e:6.2f}°    "
                  f"gt  rot={r_tt_g:6.2f}° tr={t_tt_g:6.2f}°")

            results.append({
                "pair": (rec_i["name"], rec_j["name"]),
                "gt_rot": gt_ang,
                "pcc": pcc_scores,
                "ref_rot_est": r_ref_e, "ref_tr_est": t_ref_e,
                "tt_rot_est": r_tt_e, "tt_tr_est": t_tt_e,
                "ref_rot_gt": r_ref_g, "ref_tr_gt": t_ref_g,
                "tt_rot_gt": r_tt_g, "tt_tr_gt": t_tt_g,
            })

        # Summary
        print(f"\n=== ETH3D scene {args.scene.name} summary across {len(results)} pair(s) ===")
        for k in ("h1_xyz", "h2_xyz", "h1_conf", "h2_conf"):
            v = [r["pcc"][k] for r in results]
            print(f"mean pcc_{k:<8} : {np.mean(v):.4f}   min {np.min(v):.4f}")

        for variant in ("est", "gt"):
            for tag in ("ref", "tt"):
                rots = np.array([r[f"{tag}_rot_{variant}"] for r in results])
                trs = np.array([r[f"{tag}_tr_{variant}"] for r in results])
                valid = ~(np.isnan(rots) | np.isnan(trs))
                if not valid.any():
                    print(f"{tag:<4} {variant:<3}: no valid poses"); continue
                vr = rots[valid]; vt = trs[valid]
                joint = np.minimum(vr, vt)
                rra = rra_rta_at(vr); rta = rra_rta_at(vt)
                auc = auc_deg(joint)
                print(f"{tag:<4} {variant:<3} ({valid.sum()}/{len(results)} OK)  "
                      f"rot_med={np.median(vr):5.2f}°  tr_med={np.median(vt):5.2f}°  "
                      f"RRA@5={rra['at_5']:5.1f} @15={rra['at_15']:5.1f}  "
                      f"RTA@5={rta['at_5']:5.1f} @15={rta['at_15']:5.1f}  AUC30={auc:5.1f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
