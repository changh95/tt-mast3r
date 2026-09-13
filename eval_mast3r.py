#!/usr/bin/env python3
"""DUSt3R / MASt3R TT port correctness + absolute-pose evaluation on CO3Dv2.

Two families of metrics per (scene, view-pair):

  1. Port-vs-reference on real CO3D images:
       - per-head PCC (full / xyz / conf)
       - depth relative error (port vs ref) on the activated z-channel,
         masked to the reference's top-conf pixels
       - normalised Chamfer distance between port and ref 3D point clouds

  2. Absolute pose error vs CO3D ground truth via the `PairViewer`
     recovery (DUSt3R's minimal 2-view global aligner):
       - run the model symmetrically: forward on (i, j) gets X^{i,i}
         and X^{j,i}; forward on (j, i) gets X^{j,j} and X^{i,j}.
       - per-view focal estimation from each own-frame pointmap via the
         median closed-form (`estimate_focal_knowing_depth`).
       - view-j extrinsic recovered from PnP-RANSAC on X^{j,i} + view-j
         pixel grid + estimated K_j; view-i is identity by construction.
       - Relative pose compared to CO3D GT → RRA@5/15°, RTA@5/15°, AUC@30°
         (same metric family as tt-vggt's eval_vggt.py).

Usage:
    python3 eval_mast3r.py --co3d-root co3d_data \\
        --category apple --pairs 6 --device-id 0
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

# Match test_mast3r.py's import shim: the port is imported with its package spelling
# from this repo's code/ dir; ttnn comes from the active environment.
_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models.demos.mast3r.reference.torch_dust3r import load_checkpoint, load_dust3r  # noqa: E402
# Pre/post-processing + PairViewer pose live in the package now (the serving app
# imports them from there); re-exported so eval_eth3d.py / make_demo.py keep working.
from models.demos.mast3r.postprocess import (  # noqa: E402,F401
    load_image_for_dust3r, activate_pts3d, activate_conf,
    estimate_focal, pnp_pose, pair_viewer_pose,
)


# ---------- CO3D annotations ----------

def load_co3d_annotations(co3d_root: Path, category: str):
    ann_path = co3d_root / category / "frame_annotations.jgz"
    with gzip.open(ann_path, "rb") as f:
        annos = json.load(f)
    by_seq: dict = {}
    for entry in annos:
        by_seq.setdefault(entry["sequence_name"], []).append(entry)
    for seq in by_seq:
        by_seq[seq].sort(key=lambda e: e["frame_number"])
    return by_seq




# ---------- image preprocessing for DUSt3R (512×512) ----------
# load_image_for_dust3r -> models.demos.mast3r.postprocess

def co3d_gt_K(viewpoint: dict, preproc: dict) -> np.ndarray:
    """Build (3, 3) K in the `size × size` preprocessed-image pixel grid from
    CO3D's NDC-normalised focal + principal point, accounting for
    pad-to-square + resize.
    """
    W, H = preproc["orig_W"], preproc["orig_H"]
    fx_ndc, fy_ndc = viewpoint["focal_length"]
    px_ndc, py_ndc = viewpoint["principal_point"]
    half_min = min(W, H) / 2.0
    fx = fx_ndc * half_min
    fy = fy_ndc * half_min
    cx = W / 2.0 - px_ndc * half_min + preproc["offx"]
    cy = H / 2.0 - py_ndc * half_min + preproc["offy"]
    scale = preproc["out_size"] / preproc["pad_side"]
    fx *= scale; fy *= scale; cx *= scale; cy *= scale
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ---------- DUSt3R output postprocess ----------
# activate_pts3d / activate_conf -> models.demos.mast3r.postprocess


# ---------- metrics ----------

def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float((a @ b).item() / denom) if denom > 0 else 0.0


def depth_rel_error(pred_z: np.ndarray, ref_z: np.ndarray, mask: np.ndarray):
    """|pred_z - ref_z| / max(|ref_z|, eps), averaged + median over `mask`."""
    valid = mask & np.isfinite(pred_z) & np.isfinite(ref_z) & (np.abs(ref_z) > 1e-6)
    if valid.sum() == 0:
        return float("nan"), float("nan")
    rel = np.abs(pred_z[valid] - ref_z[valid]) / np.abs(ref_z[valid])
    return float(rel.mean()), float(np.median(rel))


# ---------- pose recovery (DUSt3R PairViewer, 2-view global aligner) ----------
# estimate_focal / pnp_pose / pair_viewer_pose -> models.demos.mast3r.postprocess


# ---------- CO3D GT pose / relative-pose metrics ----------

def co3d_to_opencv_extrinsic(viewpoint: dict) -> np.ndarray:
    R = np.array(viewpoint["R"], dtype=np.float64).T
    T = np.array(viewpoint["T"], dtype=np.float64)
    flip = np.diag([-1.0, -1.0, 1.0])
    return np.concatenate([(flip @ R), (flip @ T)[:, None]], axis=-1)


def rel_rotation_angle_deg(R1, R2):
    Rrel = R1 @ R2.T
    tr = np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def rel_translation_angle_deg(t1, t2):
    n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos = float(np.clip(np.dot(t1 / n1, t2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def extrinsic_to_rel(extri_i, extri_j):
    Ri, ti = extri_i[:, :3], extri_i[:, 3]
    Rj, tj = extri_j[:, :3], extri_j[:, 3]
    return Rj @ Ri.T, tj - (Rj @ Ri.T) @ ti


def rra_rta_at(errs, thresholds=(5, 15)):
    errs = np.asarray([e for e in errs if np.isfinite(e)])
    if len(errs) == 0:
        return {f"at_{t}": 0.0 for t in thresholds}
    return {f"at_{t}": float((errs < t).mean() * 100.0) for t in thresholds}


def auc_deg(errs, max_thr=30.0, nbins=30):
    errs = np.asarray([e for e in errs if np.isfinite(e)])
    if len(errs) == 0:
        return 0.0
    thrs = np.linspace(0.0, max_thr, nbins + 1)
    cdf = np.array([(errs < t).mean() for t in thrs])
    return float(np.trapz(cdf, thrs) / max_thr * 100.0)


def chamfer_distance_norm(a: np.ndarray, b: np.ndarray, subsample: int = 8192):
    """Symmetric L2 Chamfer distance between point clouds `a`, `b` (both
    (N, 3)), normalised by the median |a|. Uses random subsample for speed.
    """
    rng = np.random.default_rng(0)
    if a.shape[0] > subsample:
        a = a[rng.choice(a.shape[0], subsample, replace=False)]
    if b.shape[0] > subsample:
        b = b[rng.choice(b.shape[0], subsample, replace=False)]
    # Pairwise distances (K, M)
    da = np.sqrt(((a[:, None] - b[None, :]) ** 2).sum(-1))
    cd = da.min(axis=1).mean() + da.min(axis=0).mean()
    norm = np.linalg.norm(a, axis=-1).clip(min=1e-6).mean()
    return float(cd / (2.0 * norm))


# ---------- per-pair evaluation ----------

def eval_pair(i: int, j: int, anns_i: dict, anns_j: dict, co3d_root: Path,
              ref_model, tt_forward, state, device, img_size: int = 512,
              conf_keep_pct: float = 30.0):
    path_i = co3d_root / anns_i["image"]["path"]
    path_j = co3d_root / anns_j["image"]["path"]
    img_i, _ = load_image_for_dust3r(path_i, img_size)
    img_j, preproc_j = load_image_for_dust3r(path_j, img_size)

    # Symmetric forward passes (PairViewer needs both view-i-native and
    # view-j-native pointmaps to estimate focal per camera).
    with torch.no_grad():
        ref_ii, ref_ji = ref_model(img_i, img_j)
        ref_jj, ref_ij = ref_model(img_j, img_i)
    tt_ii, tt_ji = tt_forward(img_i, img_j, state, device)
    tt_jj, tt_ij = tt_forward(img_j, img_i, state, device)

    # --- 1. PCC per head per channel group (on the (i, j) forward) ---
    pcc_scores = {
        "head1_full": pcc(ref_ii, tt_ii),
        "head2_full": pcc(ref_ji, tt_ji),
        "head1_xyz":  pcc(ref_ii[:, :3], tt_ii[:, :3]),
        "head2_xyz":  pcc(ref_ji[:, :3], tt_ji[:, :3]),
        "head1_conf": pcc(ref_ii[:, 3],  tt_ii[:, 3]),
        "head2_conf": pcc(ref_ji[:, 3],  tt_ji[:, 3]),
    }

    # --- 2. depth rel error + Chamfer under reference high-conf mask ---
    ref_pts_ii = activate_pts3d(ref_ii)[0].cpu().numpy()
    ref_pts_ji = activate_pts3d(ref_ji)[0].cpu().numpy()
    tt_pts_ii = activate_pts3d(tt_ii)[0].cpu().numpy()
    tt_pts_ji = activate_pts3d(tt_ji)[0].cpu().numpy()
    ref_c_ii = activate_conf(ref_ii)[0].cpu().numpy()
    ref_c_ji = activate_conf(ref_ji)[0].cpu().numpy()

    def hi_conf_mask(c):
        return c >= np.percentile(c, 100.0 - conf_keep_pct)
    m1 = hi_conf_mask(ref_c_ii)
    m2 = hi_conf_mask(ref_c_ji)
    d1_mean, d1_med = depth_rel_error(tt_pts_ii[..., 2], ref_pts_ii[..., 2], m1)
    d2_mean, d2_med = depth_rel_error(tt_pts_ji[..., 2], ref_pts_ji[..., 2], m2)
    c1 = chamfer_distance_norm(tt_pts_ii[m1], ref_pts_ii[m1])
    c2 = chamfer_distance_norm(tt_pts_ji[m2], ref_pts_ji[m2])

    # --- 3. PairViewer pose recovery vs CO3D GT, with two focal variants ---
    # Estimated-focal: DUSt3R's PairViewer as written in the paper.
    # Known-focal:     CO3D GT intrinsics — isolates PnP / pointmap quality
    #                  from the focal-estimation head, matching how real
    #                  multi-view benchmarks evaluate "pose error given K".
    K_j_gt = co3d_gt_K(anns_j["viewpoint"], preproc_j)

    ref_est = pair_viewer_pose(ref_ii, ref_ji, ref_jj, ref_ij, img_size)
    tt_est = pair_viewer_pose(tt_ii, tt_ji, tt_jj, tt_ij, img_size)
    ref_gt = pair_viewer_pose(ref_ii, ref_ji, ref_jj, ref_ij, img_size, K_j_gt)
    tt_gt = pair_viewer_pose(tt_ii, tt_ji, tt_jj, tt_ij, img_size, K_j_gt)

    gt_i = co3d_to_opencv_extrinsic(anns_i["viewpoint"])
    gt_j = co3d_to_opencv_extrinsic(anns_j["viewpoint"])
    gt_R, gt_t = extrinsic_to_rel(gt_i, gt_j)

    def pose_err(pose):
        if pose is None:
            return float("nan"), float("nan"), float("nan"), float("nan")
        R, t, f_i, f_j = pose
        return (rel_rotation_angle_deg(R, gt_R),
                rel_translation_angle_deg(t, gt_t),
                f_i, f_j)

    ref_rot_e, ref_tr_e, ref_fi, ref_fj = pose_err(ref_est)
    tt_rot_e, tt_tr_e, tt_fi, tt_fj = pose_err(tt_est)
    ref_rot_k, ref_tr_k, _, _ = pose_err(ref_gt)
    tt_rot_k, tt_tr_k, _, _ = pose_err(tt_gt)

    return {
        "pcc": pcc_scores,
        "depth1_rel_mean": d1_mean, "depth1_rel_med": d1_med,
        "depth2_rel_mean": d2_mean, "depth2_rel_med": d2_med,
        "chamfer1": c1, "chamfer2": c2,
        # estimated-focal PairViewer
        "ref_rot_est": ref_rot_e, "ref_tr_est": ref_tr_e,
        "tt_rot_est": tt_rot_e, "tt_tr_est": tt_tr_e,
        "ref_focal_i": ref_fi, "ref_focal_j": ref_fj,
        "tt_focal_i": tt_fi, "tt_focal_j": tt_fj,
        # known-focal (CO3D GT K) PairViewer
        "ref_rot_gt": ref_rot_k, "ref_tr_gt": ref_tr_k,
        "tt_rot_gt": tt_rot_k, "tt_tr_gt": tt_tr_k,
        "K_gt_fx": float(K_j_gt[0, 0]),
    }


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--co3d-root", type=Path,
                    default=Path("co3d_data"))
    ap.add_argument("--category", default="apple")
    ap.add_argument("--seqs", default="",
                    help="Comma-separated seq names (empty = all).")
    ap.add_argument("--pairs", type=int, default=6,
                    help="Pairs of views to evaluate per scene.")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"# loading CO3Dv2 annotations: {args.category}")
    by_seq = load_co3d_annotations(args.co3d_root, args.category)
    seqs = args.seqs.split(",") if args.seqs else list(by_seq.keys())
    seqs = [s for s in seqs if s in by_seq]
    print(f"# sequences: {seqs}")

    import ttnn
    from models.demos.mast3r.tt.ttnn_dust3r import dust3r_forward, fused_config, release_device_caches
    # TT_FUSED (+ sub-knobs) is read once, here. The traced fused path needs an explicit
    # trace region (this tree's default of 0 = dynamic trace-allocation mode, which no gate
    # of this port runs in); legacy / MAST3R_TRACE=0 keep the plain open.
    fused = fused_config()
    print(f"# port path: {'fused ' + str(fused.summary()) if fused.enabled else 'legacy (TT_FUSED=0)'}")
    device = ttnn.open_device(device_id=args.device_id,
                              **fused.open_device_kwargs(l1_small_size=32 * 1024))
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    try:
        state = load_checkpoint()
        print("# loading reference DUSt3R")
        ref_model = load_dust3r(state).eval()

        # Warm the TT program cache with one dummy pair so the first timed
        # inference isn't dominated by ttnn kernel compilation.
        print("# warming TT program cache")
        dummy = torch.zeros(1, 3, args.img_size, args.img_size)
        _ = dust3r_forward(dummy, dummy, state, device)

        per_pair = []
        for seq in seqs:
            anns = by_seq[seq]
            if len(anns) < 2:
                continue
            # Pick pairs with diverse baselines: views evenly spaced.
            step = max(1, len(anns) // (args.pairs + 1))
            idxs = [k * step for k in range(args.pairs + 1)]
            pairs = [(idxs[k], idxs[k + 1]) for k in range(args.pairs)]
            for (i, j) in pairs:
                print(f"\n--- seq {seq}  pair ({i}, {j})")
                r = eval_pair(i, j, anns[i], anns[j], args.co3d_root,
                              ref_model, dust3r_forward, state, device,
                              img_size=args.img_size)
                r["seq"] = seq
                r["pair"] = (i, j)
                per_pair.append(r)
                _print_row(r)

        _print_summary(per_pair)
    finally:
        # Release the port's device caches (weights, LUTs; with TT_FUSED=1 also the metal
        # trace and the persistent buffers) before close_device -- avoids the teardown
        # crash of freeing them against a closed device.
        try:
            release_device_caches()
        except Exception as e:  # pragma: no cover - best effort at shutdown
            print(f"# release_device_caches failed: {e!r}")
        ttnn.close_device(device)


def _print_row(r):
    pc = r["pcc"]
    print(f"pcc head1: full={pc['head1_full']:.4f} xyz={pc['head1_xyz']:.4f} conf={pc['head1_conf']:.4f}")
    print(f"pcc head2: full={pc['head2_full']:.4f} xyz={pc['head2_xyz']:.4f} conf={pc['head2_conf']:.4f}")
    print(f"depth rel err (port vs ref):  "
          f"h1 mean={r['depth1_rel_mean']:.4f} med={r['depth1_rel_med']:.4f}  "
          f"h2 mean={r['depth2_rel_mean']:.4f} med={r['depth2_rel_med']:.4f}")
    print(f"chamfer (normalised):  h1={r['chamfer1']:.4f}  h2={r['chamfer2']:.4f}")
    print(f"ref  est-focal  rot={r['ref_rot_est']:.2f}°  tr={r['ref_tr_est']:.2f}°  "
          f"fi={r['ref_focal_i']:.0f} fj={r['ref_focal_j']:.0f}   "
          f"(gt fj={r['K_gt_fx']:.0f})")
    print(f"port est-focal  rot={r['tt_rot_est']:.2f}°  tr={r['tt_tr_est']:.2f}°  "
          f"fi={r['tt_focal_i']:.0f} fj={r['tt_focal_j']:.0f}")
    print(f"ref  gt-focal   rot={r['ref_rot_gt']:.2f}°  tr={r['ref_tr_gt']:.2f}°")
    print(f"port gt-focal   rot={r['tt_rot_gt']:.2f}°  tr={r['tt_tr_gt']:.2f}°")


def _print_summary(results):
    if not results:
        print("no results")
        return
    print(f"\n=== summary across {len(results)} pair(s) ===")
    pcc_keys = ["head1_full", "head2_full", "head1_xyz", "head2_xyz", "head1_conf", "head2_conf"]
    for k in pcc_keys:
        vals = [r["pcc"][k] for r in results]
        print(f"mean pcc_{k:<14} : {np.mean(vals):.4f}   min {np.min(vals):.4f}")
    for k in ("depth1_rel_mean", "depth1_rel_med", "depth2_rel_mean", "depth2_rel_med",
              "chamfer1", "chamfer2"):
        vals = [r[k] for r in results if np.isfinite(r[k])]
        if not vals:
            continue
        print(f"mean {k:<20} : {np.mean(vals):.4f}   median {np.median(vals):.4f}")

    # Pose summary vs CO3D GT — separately for DUSt3R-estimated and known focal.
    for variant in ("est", "gt"):
        for tag in ("ref", "tt"):
            rots = [r[f"{tag}_rot_{variant}"] for r in results]
            trs = [r[f"{tag}_tr_{variant}"] for r in results]
            valid = [(rr, tt) for rr, tt in zip(rots, trs)
                     if np.isfinite(rr) and np.isfinite(tt)]
            if not valid:
                print(f"{tag:<4} {variant:<3}: no valid PairViewer poses"); continue
            vr = np.array([x[0] for x in valid])
            vt = np.array([x[1] for x in valid])
            joint = np.minimum(vr, vt)
            rra = rra_rta_at(vr); rta = rra_rta_at(vt); auc = auc_deg(joint)
            print(f"{tag:<4} {variant:<3} ({len(valid)}/{len(results)} OK)  "
                  f"rot_med={np.median(vr):.2f}°  tr_med={np.median(vt):.2f}°  "
                  f"RRA@5={rra['at_5']:.1f} @15={rra['at_15']:.1f}  "
                  f"RTA@5={rta['at_5']:.1f} @15={rta['at_15']:.1f}  AUC30={auc:.1f}")


if __name__ == "__main__":
    main()
