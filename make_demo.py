#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-view 3D point-map demo for mast3r-p150 -> media/pointmap.png.

Feeds one image pair through the SERVED model (default: ``POST /predict`` of a running
``tt-model serve`` / ``tt serve changh95/mast3r-p150`` on ``http://127.0.0.1:20000``,
``output_format: npz``) or, with ``--local``, through the ttnn port directly on the device
in this process, and renders ONE figure: the two inputs on the left and the fused 3D point
map on the right from two viewpoints -- the union of both predicted pointmaps in the shared
camera-1 frame (DUSt3R convention), coloured by the source pixels, top ``--keep-pct`` %
confidence per view, gray-padding pixels dropped, equal axes.

    python3 make_demo.py                                   # media/source_1.png + source_2.png -> server
    python3 make_demo.py a.png b.png --url http://host:20000 --out media/pointmap.png
    python3 make_demo.py --local                           # ttnn on /dev/tenstorrent/0 (needs ttnn + weights)
    python3 make_demo.py --save-response /path/prefix      # also keep the raw npz + JSON response

Prints per-view confidence stats and the two-cloud coherence number (median nearest-neighbour
distance between the view-1 and view-2 clouds / scene scale; small = one coherent scene).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# media/ sits beside this file (GitHub layout) or one level up (HF repo layout:
# code/make_demo.py + media/); MAST3R_REPO_ROOT overrides.
_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.environ.get("MAST3R_REPO_ROOT") or (
    _CODE_ROOT if os.path.isdir(os.path.join(_CODE_ROOT, "media"))
    else os.path.dirname(_CODE_ROOT)
)
_MEDIA = Path(_REPO_ROOT) / "media"
PAD_GRAY = (128, 128, 128)


# ------------------------------------------------------------------ preprocessing replay

def canonical_view(img: Image.Image, rec: dict):
    """Rebuild the ``out_size x out_size`` image the network saw from a ``preprocess``
    record (server response or ``postprocess.preprocess_image``) and the mask of pixels
    that came from the photo (False on the gray padding).  Keys accepted in either the
    server (``orig_w``) or the module (``orig_W``) spelling."""
    W = int(rec.get("orig_w", rec.get("orig_W")))
    H = int(rec.get("orig_h", rec.get("orig_H")))
    offx, offy, side, out = int(rec["offx"]), int(rec["offy"]), int(rec["pad_side"]), int(rec["out_size"])
    img = img.convert("RGB")
    if rec.get("mode", "pad") == "pad":
        canvas = Image.new("RGB", (side, side), PAD_GRAY)
        canvas.paste(img, (offx, offy))
        valid = Image.new("L", (side, side), 0)
        valid.paste(255, (offx, offy, offx + W, offy + H))
    else:  # crop: offsets are <= 0
        x0, y0 = -offx, -offy
        canvas = img.crop((x0, y0, x0 + side, y0 + side))
        valid = Image.new("L", (side, side), 255)
    rgb = np.asarray(canvas.resize((out, out), Image.BICUBIC), dtype=np.uint8)
    mask = np.asarray(valid.resize((out, out), Image.NEAREST)) > 127
    return rgb, mask


# ------------------------------------------------------------------ model calls

def predict_served(url: str, paths, save_prefix: str | None = None, return_pose: bool = False,
                   wait_s: float = 600.0):
    """POST the pair to the server; returns (pts3d1, pts3d2, conf1, conf2, response-without-npz)."""
    def get(u, timeout=10.0):
        with urllib.request.urlopen(u, timeout=timeout) as r:
            return json.loads(r.read().decode())

    t0 = time.perf_counter()
    while True:
        try:
            if get(url + "/health").get("status") == "ok":
                break
        except Exception:  # noqa: BLE001 - not up yet
            pass
        if time.perf_counter() - t0 > wait_s:
            raise SystemExit(f"server at {url} not ready after {wait_s:.0f} s")
        time.sleep(2.0)
    b64 = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in paths]
    body = {"image1": b64[0], "image2": b64[1], "output_format": "npz", "return_pose": return_pose}
    req = urllib.request.Request(url + "/predict", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600.0) as r:
        resp = json.loads(r.read().decode())
    npz_bytes = base64.b64decode(resp.pop("npz_b64"))
    if save_prefix:
        Path(save_prefix + ".npz").write_bytes(npz_bytes)
        Path(save_prefix + "_resp.json").write_text(json.dumps(resp, indent=1))
    npz = np.load(io.BytesIO(npz_bytes))
    return (npz["pts3d1"].astype(np.float32), npz["pts3d2"].astype(np.float32),
            npz["conf1"].astype(np.float32), npz["conf2"].astype(np.float32), resp)


def predict_local(paths, return_pose: bool = False):
    """Run the ttnn port on the device in this process (mirrors models/server/app.py)."""
    if _CODE_ROOT not in sys.path:
        sys.path.insert(0, _CODE_ROOT)
    import torch
    import ttnn
    from models.demos.mast3r.postprocess import (
        IMG_SIZE, activate_conf, activate_pts3d, pair_viewer_pose, preprocess_image, preprocess_mode,
    )
    from models.demos.mast3r.reference.torch_dust3r import load_checkpoint
    from models.demos.mast3r.tt import ttnn_dust3r as port

    mode = preprocess_mode()
    imgs, recs = zip(*[preprocess_image(Image.open(p), IMG_SIZE, mode) for p in paths])
    fused = port.fused_config()
    device = ttnn.open_device(**fused.open_device_kwargs(device_id=int(os.environ.get("TT_DEVICE_ID", "0")),
                                                         l1_small_size=32 * 1024))
    try:
        state = load_checkpoint()
        with torch.no_grad():
            _ = port.dust3r_forward(imgs[0], imgs[1], state, device)  # compile / capture
            t0 = time.perf_counter()
            out_ii, out_ji = port.dust3r_forward(imgs[0], imgs[1], state, device)
            fwd_ms = (time.perf_counter() - t0) * 1e3
            pose = None
            if return_pose:
                out_jj, out_ij = port.dust3r_forward(imgs[1], imgs[0], state, device)
                res = pair_viewer_pose(out_ii, out_ji, out_jj, out_ij, IMG_SIZE)
                if res is not None:
                    R, t, f1, f2 = res
                    pose = {"R": R.tolist(), "t": t.tolist(), "focal_1": float(f1), "focal_2": float(f2)}
        pts1 = activate_pts3d(out_ii)[0].numpy(); pts2 = activate_pts3d(out_ji)[0].numpy()
        conf1 = activate_conf(out_ii)[0].numpy(); conf2 = activate_conf(out_ji)[0].numpy()
    finally:
        try:
            port.release_device_caches()
        finally:
            ttnn.close_device(device)
    resp = {"model": "mast3r-p150 (local ttnn)", "frame": "camera_1", "fused": fused.summary(),
            "preprocess": [{"mode": r["mode"], "orig_w": r["orig_W"], "orig_h": r["orig_H"], "offx": r["offx"],
                            "offy": r["offy"], "pad_side": r["pad_side"], "out_size": r["out_size"]} for r in recs],
            "pose": pose, "timing_ms": {"forward": fwd_ms}}
    return pts1, pts2, conf1, conf2, resp


# ------------------------------------------------------------------ point cloud + metrics

def select_points(pts, rgb, conf, valid, keep_pct: float, conf_min: float):
    """Finite, non-padding points in the top ``keep_pct`` % of confidence within the photo,
    and never below ``conf_min`` (DUSt3R conf = 1 + exp(c) >= 1; ~1 = the network gave up,
    typically far background seen through a window)."""
    finite = np.isfinite(pts).all(-1) & np.isfinite(conf) & valid
    thr = float(np.percentile(conf[finite], 100.0 - keep_pct)) if finite.any() else np.inf
    thr = max(thr, conf_min)
    keep = finite & (conf >= thr)
    return pts[keep].reshape(-1, 3), rgb[keep].reshape(-1, 3).astype(np.float32) / 255.0, thr


def nn_median(a: np.ndarray, b: np.ndarray, n: int = 4000, seed: int = 0) -> float:
    """Median distance from a random subset of ``a`` to its nearest neighbour in ``b``."""
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(n, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(4 * n, len(b)), replace=False)]
    d = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), 256):
        blk = a[i:i + 256]
        d[i:i + 256] = np.sqrt(((blk[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d))


def coherence(p1: np.ndarray, p2: np.ndarray) -> dict:
    """Two-cloud coherence: symmetric median NN distance / scene scale (95th pct radius)."""
    allp = np.concatenate([p1, p2], 0)
    scale = float(np.percentile(np.linalg.norm(allp - np.median(allp, 0), axis=1), 95))
    d12, d21 = nn_median(p1, p2), nn_median(p2, p1)
    return {"scene_scale": scale, "nn_med_12": d12, "nn_med_21": d21,
            "sym_med_over_scale": 0.5 * (d12 + d21) / max(scale, 1e-9)}


# ------------------------------------------------------------------ figure

DEFAULT_VIEWS = ((35, -62), (14, 38))   # (elev, azim) of the two 3-D panels


def render(images, clouds, colours, out_path: Path, caption: str, dpi: int = 150, max_points: int = 90000,
           point_size: float = 2.0, zoom: float = 1.45, views=DEFAULT_VIEWS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)

    pts = np.concatenate(clouds, 0)
    col = np.concatenate(colours, 0)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, col = pts[idx], col[idx]
    centre = np.median(pts, 0)
    scale = float(np.percentile(np.linalg.norm(pts - centre, axis=1), 95))
    q = (pts - centre) / max(scale, 1e-9)
    # camera frame (x right, y down, z forward) -> plot frame (x right, z forward, up)
    X, Y, Z = q[:, 0], q[:, 2], -q[:, 1]

    fig = plt.figure(figsize=(15, 6.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.35, 1.35], height_ratios=[1, 1],
                          left=0.01, right=0.99, top=0.93, bottom=0.07, wspace=0.0, hspace=0.12)
    for r, (im, title) in enumerate(zip(images, ("view 1 (reference camera)", "view 2"))):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    for c, ((elev, azim), title) in enumerate(zip(views, ("fused point map, viewpoint A",
                                                           "fused point map, viewpoint B")), start=1):
        ax = fig.add_subplot(gs[:, c], projection="3d")
        ax.scatter(X, Y, Z, c=col, s=point_size, marker=".", linewidths=0, alpha=0.95, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        lim = 1.0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)
    fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=9.5, color="#333333")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image1", nargs="?", default=str(_MEDIA / "source_1.png"), help="view 1 = reference camera")
    ap.add_argument("image2", nargs="?", default=str(_MEDIA / "source_2.png"))
    ap.add_argument("--url", default="http://127.0.0.1:20000", help="served model (POST /predict, npz)")
    ap.add_argument("--local", action="store_true", help="run the ttnn port on the device in this process")
    ap.add_argument("--out", default=str(_MEDIA / "pointmap.png"))
    ap.add_argument("--keep-pct", type=float, default=70.0, help="top confidence %% kept per view")
    ap.add_argument("--conf-min", type=float, default=3.0, help="never plot points below this confidence")
    ap.add_argument("--zoom", type=float, default=1.45, help="3-D panel zoom")
    ap.add_argument("--views", default=";".join(f"{e},{a}" for e, a in DEFAULT_VIEWS),
                    help="elev,azim of the two 3-D panels, ';'-separated")
    ap.add_argument("--from-response", default=None,
                    help="prefix of a saved --save-response (<prefix>.npz + <prefix>_resp.json): re-render, no model call")
    ap.add_argument("--pose", action="store_true", help="also recover the PairViewer pose (2 forwards) and print it")
    ap.add_argument("--save-response", default=None, help="prefix: write <prefix>.npz + <prefix>_resp.json (server mode)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    paths = [args.image1, args.image2]
    t0 = time.perf_counter()
    if args.from_response:
        npz = np.load(args.from_response + ".npz")
        pts1, pts2, conf1, conf2 = (npz[k].astype(np.float32) for k in ("pts3d1", "pts3d2", "conf1", "conf2"))
        resp = json.loads(Path(args.from_response + "_resp.json").read_text())
    elif args.local:
        pts1, pts2, conf1, conf2, resp = predict_local(paths, return_pose=args.pose)
    else:
        pts1, pts2, conf1, conf2, resp = predict_served(args.url, paths, args.save_response, args.pose)
    print(f"# {resp.get('model')} frame={resp.get('frame')} in {time.perf_counter() - t0:.1f} s; "
          f"timing_ms={json.dumps({k: round(v, 1) for k, v in resp.get('timing_ms', {}).items()})}")

    images = [Image.open(p).convert("RGB") for p in paths]
    canon = [canonical_view(im, rec) for im, rec in zip(images, resp["preprocess"])]
    clouds, colours = [], []
    for i, (pts, conf, (rgb, valid)) in enumerate(zip((pts1, pts2), (conf1, conf2), canon), start=1):
        p, c, thr = select_points(pts, rgb, conf, valid, args.keep_pct, args.conf_min)
        clouds.append(p); colours.append(c)
        v = conf[valid]
        print(f"# view {i}: preprocess={resp['preprocess'][i - 1]} photo pixels {valid.mean():.3f}; "
              f"conf min/median/max {v.min():.2f}/{np.median(v):.2f}/{v.max():.2f}; kept {len(p)} pts (conf >= {thr:.2f}); "
              f"z median {np.median(p[:, 2]):.3f}")
    coh = coherence(clouds[0], clouds[1])
    print(f"# coherence: median NN distance view1<->view2 = {coh['sym_med_over_scale']:.4f} of the scene scale "
          f"(scale {coh['scene_scale']:.3f}, nn12 {coh['nn_med_12']:.4f}, nn21 {coh['nn_med_21']:.4f})")
    pose = resp.get("pose")
    if pose:
        R = np.asarray(pose["R"])
        rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        print(f"# PairViewer pose: rotation {rot:.2f} deg, focals {pose['focal_1']:.0f} / {pose['focal_2']:.0f} px (512 grid)")

    caption = (f"{Path(paths[0]).name} + {Path(paths[1]).name}  ->  both predicted pointmaps in the camera-1 frame, "
               f"coloured by source pixels, top {args.keep_pct:.0f} % confidence per view (conf >= {args.conf_min:g}); "
               f"device forward {resp.get('timing_ms', {}).get('forward', float('nan')):.0f} ms")
    views = [tuple(float(x) for x in v.split(",")) for v in args.views.split(";")]
    render(images, clouds, colours, Path(args.out), caption, dpi=args.dpi, zoom=args.zoom, views=views)
    print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
