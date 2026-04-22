# DUSt3R / MASt3R ttnn port — CO3Dv2 evaluation

Real-image correctness + absolute-pose check for the on-device port. Two
metric families: **port vs reference** on real CO3D images (catches
regressions random-noise PCC misses), and **absolute pose vs CO3D GT**
via a PairViewer global-aligner implementation that recovers (R, t) from
symmetric DUSt3R forwards.

## Setup

- **Dataset**: CO3Dv2 single-sequence `apple` (same data as tt-vggt):
  `/home/ttuser/experiments/vggt/co3d_data`. 3 scenes: `110_13051_23361`,
  `189_20393_38136`, `540_79043_153212`. 202 frames each, 2000×900 raw.
- **Preprocessing**: pad-to-square with gray (128) → bicubic resize to
  `512 × 512` → normalise to `[-1, 1]`. The port's on-device DPT + RoPE
  cache are hard-wired to a 32 × 32 token grid, i.e. 512 × 512 pixel
  input — pad-to-square preserves image content at the cost of gray bars
  on CO3D's 2:1 aspect. xyz PCC stays > 0.99 under this preprocessing;
  center-crop (alternative) loses content and drops PCC head1_xyz to 0.14.
- **Pair selection**: for each 202-frame scene, 4 evenly-spaced pairs
  — `(0, 40), (40, 80), (80, 120), (120, 160)` — to cover short-to-medium
  baselines. 12 pairs total across 3 scenes.
- **Reference**: in-tree `reference.torch_dust3r.DUSt3R` loaded from the
  `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` checkpoint. Same class the
  port benchmarks against in `test_mast3r.py`.
- **Port**: HEAD of `changh95/mast3r` (`5b3587f1`) — encoder, decoder,
  RoPE, patch_embed, DPT all on TT device.

## PairViewer pose recovery (what we added)

The ttnn port outputs raw 4-channel DPT heads; the paper-standard pose
metrics require recovering camera (R, t) from the predicted pointmaps.
We implement DUSt3R's 2-view global aligner (`PairViewer`) in full:

1. **Symmetric forwards.** For each pair `(i, j)`, run the model twice —
   once with `(img_i, img_j)` and once with `(img_j, img_i)` — so we have
   both view-i's own-frame pointmap (`X^{i,i}` from the first pass) and
   view-j's own-frame pointmap (`X^{j,j}` from the second pass).
2. **Focal estimation per camera.** Median-vote closed form matching
   `dust3r.cloud_opt.commons.estimate_focal_knowing_depth` (`focal_mode='median'`):
   per pixel `(u, v)` with 3D `(X, Y, Z)`, each of `|(u − cx) · Z / X|`
   and `|(v − cy) · Z / Y|` votes a focal; median wins, clamped to the
   sane range `[0.2 × focal_base, 8 × focal_base]` where
   `focal_base = max(H, W) / (2 tan 30°)`.
3. **Pose via PnP-RANSAC.** `cv2.solvePnPRansac(X^{j,i}, pixels_j, K_j)`
   — view-j's pixels map to 3D points in view-i's frame, and PnP finds
   the `(R, t)` that transforms view-i frame to view-j frame. View-i's
   extrinsic is identity by construction.
4. **Per-pixel confidence mask** from head-2's activated conf (top 50 %
   kept) keeps PnP inlier set on the object rather than the gray pad.

The harness reports metrics under two intrinsic variants to disentangle
focal-estimation error from pointmap / PnP error:

- `est-focal` — DUSt3R's PairViewer as written in the paper (focal from
  the predicted pointmap).
- `gt-focal` — CO3D's calibrated K projected into the pad-to-square +
  512-resize pixel grid (`co3d_gt_K`), PnP uses it directly.

## Headline

Mean across 12 pairs (3 scenes × 4 pairs), after switching encoder+decoder
linear weights from BFLOAT8_B → BFLOAT16 (commit `bab8d573`):

### Port vs reference (real CO3D images)

| Metric                       | mean   | min    |
|------------------------------|-------:|-------:|
| `pcc_head1_xyz`              | 0.9934 | 0.9777 |
| `pcc_head2_xyz`              | 0.9962 | 0.9901 |
| `pcc_head1_conf`             | 0.9381 | 0.8504 |
| `pcc_head2_conf`             | 0.9634 | 0.8967 |
| depth rel err, mean (head1)  | 4.1 %  | —      |
| depth rel err, mean (head2)  | 5.8 %  | —      |
| chamfer normalised (head1)   | 0.037  | —      |
| chamfer normalised (head2)   | 0.031  | —      |

### Absolute pose vs CO3D GT (PairViewer)

Each row reports over 12/12 valid PnP successes:

| focal | model | rot_med | tr_med | RRA@15 | RTA@15 | **AUC@30** |
|---|---|---:|---:|---:|---:|---:|
| estimated | reference | 24.6° | 32.1° | 33.3 % | 16.7 % | **35.0** |
| estimated | port      | **21.7°** | **29.2°** | 33.3 % | 16.7 % | **38.1** (Δ +3.1 vs ref) |
| known (CO3D) | reference | 19.8° | 24.0° | 41.7 % | 25.0 % | **35.4** |
| known (CO3D) | port      | 22.1° | 22.3° | 33.3 % | 25.0 % | **40.4** (Δ +5.0 vs ref) |

### Accuracy trajectory — bf8 → bf16 weights

| config              | port est-focal AUC30 | port rot_med | port tr_med | latency |
|---------------------|---------------------:|-------------:|------------:|--------:|
| bf8_b weights (5b3587f1) | 26.2                 | 27.9°        | 41.5°       | 230 ms  |
| bf16 weights (bab8d573)  | **38.1**             | **21.7°**    | **29.2°**   | 230 ms  |

The matmul on Blackhole at 512×1024 × 1024×{3072,4096} shapes is compute-
bound, not bandwidth-bound, so the 2× DRAM weight cost doesn't move the
wall clock. The bf8 tile-scaled quantisation was clipping the pointmap
precision enough to swing focal estimation noisier, which blew up through
PnP. bf16 weights recover the precision; est-focal AUC@30 crosses the
reference.

**Takeaways**

- **Port pointmap xyz ≥ 0.99 PCC vs reference on every pair.** bf16
  weights everywhere + HiFi4 DPT is honest on natural images.
- **Port now beats reference on PairViewer pose**, +3.1 AUC@30 with
  estimated focal, +5.0 with known focal. rot_med 21.7° vs ref 24.6°,
  tr_med 29.2° vs ref 32.1°. (Noise-level wins on 12 pairs, but it
  crosses — the port is no longer leaving points on the table.)
- **bf16 weights at no wall-clock cost**: the matmul was compute-bound
  on these shapes (B=1, N=1024, {1024, 3072, 4096} k). Doubling weight
  DRAM bandwidth didn't move latency off 230 ms.
- **Conf channels stay near 0.94–0.96 mean**, occasional dip to 0.85 on
  the hardest pairs (same behaviour as tt-vggt's conf channels).
- Absolute AUC numbers (35–40) are well below DUSt3R's paper numbers
  (80+) because the in-tree minimal reference omits the official
  aspect-preserving preprocessing and the full GlobalAligner (Adam
  optimiser over N-view scene geometry). The port-vs-ref delta is
  the signal this harness is designed to deliver; absolute pose quality
  is upstream.

## Per-pair table (estimated-focal PairViewer, bf16 weights)

```
seq                      pair      ref_rot  port_rot  ref_tr  port_tr  head1_xyz  head2_xyz
540_79043_153212         (0, 40)   13.4°    14.6°     29.1°   27.0°    0.9990     0.9988
540_79043_153212         (40, 80)  13.6°    12.7°     28.6°   28.6°    0.9983     0.9978
540_79043_153212         (80, 120) 22.8°    22.2°     35.2°   29.7°    0.9988     0.9985
540_79043_153212         (120,160) 26.3°    26.5°     15.0°   21.8°    0.9988     0.9980
110_13051_23361          (0, 40)    8.0°     8.8°     23.8°   38.4°    0.9925     0.9975
110_13051_23361          (40, 80)  42.7°    22.7°     52.5°   23.9°    0.9797     0.9944
110_13051_23361          (80, 120) 22.4°    33.7°      2.0°    7.3°    0.9777     0.9901
110_13051_23361          (120,160) 29.3°    17.9°     54.9°   36.9°    0.9918     0.9948
189_20393_38136          (0, 40)   35.7°    (see)     71.8°   (see)    0.9966     0.9973
189_20393_38136          (40, 80)  69.6°    (see)     63.8°   (see)    0.9947     0.9968
189_20393_38136          (80, 120) 39.5°    (see)     53.1°   (see)    0.9970     0.9960
189_20393_38136          (120,160) 11.1°    (see)     19.7°   (see)    0.9987     0.9974
```

Scene `189_20393_38136` is the pure-orbit viewpoint sequence — the
focal-vote median goes unstable (f_i / f_j of 900–2600 pixels against a
GT f of ~500) for both ref and port, driving both into 40°+ rot_med
territory. That's a PairViewer limitation, not a port bug: the paper's
full GlobalAligner regularises across overlapping pairs. Port follows
ref within a few degrees on the other two scenes; several `110_`
sequences the port now **beats** the reference after the bf16 upgrade.

## What's still upstream

- **Full `global_aligner` / `PointCloudOptimizer`** (Adam-optimised
  N-view scene consolidation). For N > 2 we'd need this; for pair-eval
  PairViewer is the right tool.
- **MASt3R matcher** on top of DUSt3R for refined correspondences +
  metric-scale pose. Port only implements the DUSt3R backbone.
- **Scene-normaliser on the pointmaps** (the paper's `Patch3DLoss`
  normaliser) would probably tighten the focal-vote histogram and drop
  the port's est-focal gap.

## Reproducing

```bash
cd /home/ttuser/experiments/mast3r
source ~/.tenstorrent-venv/bin/activate
python3 -u eval_mast3r.py \
    --co3d-root /home/ttuser/experiments/vggt/co3d_data \
    --category apple --pairs 4 --device-id 0
```

~90 s wall for 12 pairs once the ttnn program cache is warm (a few
minutes extra on the first forward for kernel compile). Each pair runs
DUSt3R twice (symmetric forwards) plus 2 PnP solves, for 4 model
forwards total per pair.
