#!/usr/bin/env python3
"""DUSt3R/MASt3R end-to-end benchmark on a single Tenstorrent Blackhole chip.

Outputs the fields PROGRAM.md expects so they can be grepped:
    inference_speed: <frames/sec>
    accuracy:        <PCC * 100, threshold 99.0>
    peak_dram:       <MB, best-effort>
    pcc:             <raw PCC, for context>
    latency_ms:      <wall-clock per call>

Usage:
    python3 test_mast3r.py                      # end_to_end (default)
    python3 test_mast3r.py --layer end_to_end --runs 3
    python3 test_mast3r.py --layer full_encoder
    python3 test_mast3r.py --layer dpt_head_device   # the on-device DPT head (dpt_head = torch bf16 host reference)

The fused path is the default since the 2026-09-13 device validation (TT_FUSED unset or 1;
read once by the port, see models/demos/mast3r/tt/fused.py): full_encoder / full_decoder /
dpt_head_device run the fused ops eagerly; end_to_end runs the traced TtDust3r graph unless
MAST3R_TRACE=0. TT_FUSED=0 selects the legacy eager graph. The device is opened with the
trace region the configuration needs (FusedConfig.open_device_kwargs) and the port's device
caches are released before close_device.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# ttnn must be importable from the active environment (a tt-metal python_env, or
# PYTHONPATH=<tt-metal>:<tt-metal>/ttnn). The port is imported with its package
# spelling (models.demos.mast3r.*) from this repo's code/ dir.
_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import torch  # noqa: E402

from models.demos.mast3r.reference.torch_dust3r import (  # noqa: E402
    load_checkpoint,
    load_patch_embed,
    load_encoder_block,
    load_encoder,
    load_decoder_block,
    load_decoder,
    load_dpt_head,
    load_dust3r,
    make_positions,
)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 0.0
    return float((a @ b).item() / denom)


def device_dram_mb(device) -> float:
    """Best-effort peak DRAM in MB. Returns 0.0 if APIs unavailable."""
    try:
        import ttnn  # noqa
        # Try a few common API paths.
        for attr in ("dram_size_per_bank", "dram_size"):
            fn = getattr(device, attr, None)
            if callable(fn):
                # Not really peak — but indicates capacity. Skip.
                pass
        # Allocator stats (preferred if exposed):
        try:
            import ttnn._ttnn as _t
            stats = _t.device.allocator_statistics(device, _t.tensor.BufferType.DRAM)
            return float(stats.peak_bytes) / (1024 * 1024)
        except Exception:
            pass
        try:
            stats = ttnn.get_memory_per_bank_dram_allocation_stats(device)
            return float(stats.peak_bytes) / (1024 * 1024)
        except Exception:
            pass
    except Exception:
        pass
    return 0.0


def print_result(layer: str, pcc_val: float, latency_ms: float,
                 status: str, peak_dram_mb: float = 0.0):
    speed = (1000.0 / latency_ms) if latency_ms > 0 else 0.0
    accuracy = max(0.0, min(100.0, pcc_val * 100.0))
    print(f"--- layer: {layer}")
    print(f"pcc: {pcc_val:.4f}")
    print(f"latency_ms: {latency_ms:.2f}")
    print(f"inference_speed: {speed:.4f}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"peak_dram: {peak_dram_mb:.2f}")
    print(f"status: {status}")
    print("---")


# ---------- per-layer runners ----------

def run_patch_embed(device, runs: int):
    from models.demos.mast3r.tt.ttnn_dust3r import patch_embed as tt_patch_embed

    torch.manual_seed(0)
    state = load_checkpoint()
    ref = load_patch_embed(state)
    img = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        ref_out, _pos, _hw = ref(img)

    w = state["patch_embed.proj.weight"]
    b = state["patch_embed.proj.bias"]

    _ = tt_patch_embed(img, w, b, device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tt_out = tt_patch_embed(img, w, b, device)
        times.append((time.perf_counter() - t0) * 1000)
    return ref_out, tt_out, min(times)


def run_encoder_block(device, idx: int, runs: int):
    from models.demos.mast3r.tt.ttnn_dust3r import encoder_block as tt_encoder_block

    torch.manual_seed(0)
    state = load_checkpoint()
    pe = load_patch_embed(state)
    blk = load_encoder_block(state, idx)
    img = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        tokens, pos, _hw = pe(img)
        ref_out = blk(tokens, pos)

    weights = {
        f"{k}": state[f"enc_blocks.{idx}.{k}"] for k in [
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
            "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias",
            "mlp.fc1.weight", "mlp.fc1.bias",
            "mlp.fc2.weight", "mlp.fc2.bias",
        ]
    }
    _ = tt_encoder_block(tokens, pos, weights, device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tt_out = tt_encoder_block(tokens, pos, weights, device)
        times.append((time.perf_counter() - t0) * 1000)
    return ref_out, tt_out, min(times)


def run_decoder_block(device, idx: int, branch: int, runs: int):
    from models.demos.mast3r.tt.ttnn_dust3r import decoder_block as tt_decoder_block

    torch.manual_seed(0)
    state = load_checkpoint()
    ref = load_decoder_block(state, idx, branch=branch)
    B, N, D = 1, 32 * 32, 768
    x = torch.randn(B, N, D) * 0.02
    y = torch.randn(B, N, D) * 0.02
    pos = make_positions(B, 32, 32, x.device)
    with torch.no_grad():
        ref_out = ref(x, y, pos, pos)

    prefix = "dec_blocks." if branch == 1 else "dec_blocks2."
    keys = [
        "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
        "norm3.weight", "norm3.bias", "norm_y.weight", "norm_y.bias",
        "attn.qkv.weight", "attn.qkv.bias",
        "attn.proj.weight", "attn.proj.bias",
        "cross_attn.projq.weight", "cross_attn.projq.bias",
        "cross_attn.projk.weight", "cross_attn.projk.bias",
        "cross_attn.projv.weight", "cross_attn.projv.bias",
        "cross_attn.proj.weight", "cross_attn.proj.bias",
        "mlp.fc1.weight", "mlp.fc1.bias",
        "mlp.fc2.weight", "mlp.fc2.bias",
    ]
    weights = {k: state[f"{prefix}{idx}.{k}"] for k in keys}

    _ = tt_decoder_block(x, y, pos, pos, weights, device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tt_out = tt_decoder_block(x, y, pos, pos, weights, device)
        times.append((time.perf_counter() - t0) * 1000)
    return ref_out, tt_out, min(times)


def run_dpt_head(device, branch: int, runs: int):
    from models.demos.mast3r.tt.ttnn_dust3r import dpt_head as tt_dpt_head

    torch.manual_seed(0)
    state = load_checkpoint()
    ref = load_dpt_head(state, branch=branch)
    B, N = 1, 32 * 32
    f0 = torch.randn(B, N, 1024) * 0.02
    f1 = torch.randn(B, N, 768) * 0.02
    f2 = torch.randn(B, N, 768) * 0.02
    f3 = torch.randn(B, N, 768) * 0.02
    feats = [f0, f1, f2, f3]
    hw = (32, 32)
    with torch.no_grad():
        ref_out = ref(feats, hw)
    _ = tt_dpt_head(feats, hw, state, branch, device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tt_out = tt_dpt_head(feats, hw, state, branch, device)
        times.append((time.perf_counter() - t0) * 1000)
    return ref_out, tt_out, min(times)


def run_dpt_head_device(device, branch: int, runs: int):
    """The on-device DPT head (``dpt_head_device``, the function ``dust3r_forward`` /
    ``TtDust3r`` call) on the same synthetic taps as ``run_dpt_head``. The four taps are
    uploaded once as ``(B, N, D)`` bf16 TILE tensors; with ``TT_FUSED=1`` this A/Bs the
    TILE tap reshape, the fused conv relu and the cached prepared conv weights eagerly
    (``MAST3R_DPT_FUSE=0`` isolates the first two from the weight cache)."""
    import ttnn
    from models.demos.mast3r.tt.ttnn_dust3r import dpt_head_device as tt_dpt_head_device

    torch.manual_seed(0)
    state = load_checkpoint()
    ref = load_dpt_head(state, branch=branch)
    B, N = 1, 32 * 32
    f0 = torch.randn(B, N, 1024) * 0.02
    f1 = torch.randn(B, N, 768) * 0.02
    f2 = torch.randn(B, N, 768) * 0.02
    f3 = torch.randn(B, N, 768) * 0.02
    feats = [f0, f1, f2, f3]
    hw = (32, 32)
    Hh, Wh = hw[0] * 16, hw[1] * 16
    with torch.no_grad():
        ref_out = ref(feats, hw)
    tt_feats = [ttnn.from_torch(f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for f in feats]

    def forward():
        out = tt_dpt_head_device(tt_feats, hw, state, branch, device)   # NHWC flat (1, 1, B*Hh*Wh, 4)
        return ttnn.to_torch(out).reshape(B, Hh, Wh, 4).permute(0, 3, 1, 2).contiguous().float()

    _ = forward()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tt_out = forward()
        times.append((time.perf_counter() - t0) * 1000)
    for t in tt_feats:
        ttnn.deallocate(t)
    return ref_out, tt_out, min(times)


def run_full_encoder(device, runs: int):
    from models.demos.mast3r.tt.ttnn_dust3r import full_encoder as tt_full_encoder

    torch.manual_seed(0)
    state = load_checkpoint()
    ref = load_encoder(state)
    img = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        ref_out, _pos, _hw = ref(img)
    _ = tt_full_encoder(img, state, device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tt_out = tt_full_encoder(img, state, device)
        times.append((time.perf_counter() - t0) * 1000)
    return ref_out, tt_out, min(times)


def run_full_decoder(device, runs: int):
    from models.demos.mast3r.tt.ttnn_dust3r import full_decoder as tt_full_decoder

    torch.manual_seed(0)
    state = load_checkpoint()
    ref = load_decoder(state)
    B, N, D = 1, 32 * 32, 1024
    f1 = torch.randn(B, N, D) * 0.02
    f2 = torch.randn(B, N, D) * 0.02
    pos = make_positions(B, 32, 32, f1.device)
    with torch.no_grad():
        ref_out1, ref_out2, _, _ = ref(f1, f2, pos)
    _ = tt_full_decoder(f1, f2, pos, state, device)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tt_out1, tt_out2, _, _ = tt_full_decoder(f1, f2, pos, state, device)
        times.append((time.perf_counter() - t0) * 1000)
    p1 = pcc(ref_out1, tt_out1)
    p2 = pcc(ref_out2, tt_out2)
    print(f"# decoder branch PCCs: b1={p1:.4f} b2={p2:.4f}")
    return torch.stack([ref_out1, ref_out2]), torch.stack([tt_out1, tt_out2]), min(times)


def run_end_to_end(device, runs: int):
    from models.demos.mast3r.tt.ttnn_dust3r import dust3r_forward

    torch.manual_seed(0)
    state = load_checkpoint()
    ref = load_dust3r(state)
    img1 = torch.randn(1, 3, 512, 512)
    img2 = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        ref_out1, ref_out2 = ref(img1, img2)

    _ = dust3r_forward(img1, img2, state, device)  # warm
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tt_out1, tt_out2 = dust3r_forward(img1, img2, state, device)
        times.append((time.perf_counter() - t0) * 1000)
    p1 = pcc(ref_out1, tt_out1)
    p2 = pcc(ref_out2, tt_out2)
    print(f"# e2e head PCCs: head1={p1:.4f} head2={p2:.4f}")
    return torch.stack([ref_out1, ref_out2]), torch.stack([tt_out1, tt_out2]), min(times)


LAYER_DISPATCH = {
    "patch_embed": lambda d, r: run_patch_embed(d, r),
    "full_encoder": lambda d, r: run_full_encoder(d, r),
    "full_decoder": lambda d, r: run_full_decoder(d, r),
    "dpt_head": lambda d, r: run_dpt_head(d, 1, r),
    "dpt_head_2": lambda d, r: run_dpt_head(d, 2, r),
    "dpt_head_device": lambda d, r: run_dpt_head_device(d, 1, r),
    "dpt_head_device_2": lambda d, r: run_dpt_head_device(d, 2, r),
    "end_to_end": lambda d, r: run_end_to_end(d, r),
    **{f"encoder_block_{i}": (lambda d, r, i=i: run_encoder_block(d, i, r)) for i in range(24)},
    **{f"decoder_block_{i}": (lambda d, r, i=i: run_decoder_block(d, i, 1, r)) for i in range(12)},
    **{f"decoder2_block_{i}": (lambda d, r, i=i: run_decoder_block(d, i, 2, r)) for i in range(12)},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="end_to_end")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of timed runs (best-of-N reported)")
    parser.add_argument("--device-id", type=int, default=0,
                        help="Tenstorrent device id (per PROGRAM.md, this user's chip)")
    args = parser.parse_args()

    import ttnn
    from models.demos.mast3r.tt.ttnn_dust3r import fused_config, release_device_caches

    # The port reads TT_FUSED (+ sub-knobs) once, here. With the traced fused path the
    # device needs an explicit trace region (otherwise this tree's default of 0 selects the
    # dynamic trace-allocation mode, which no gate of this port runs in).
    fused = fused_config()
    print(f"# port path: {'fused ' + str(fused.summary()) if fused.enabled else 'legacy (TT_FUSED=0)'}")
    # l1_small_size needed by ttnn.conv2d (sliding window state buffer).
    device = ttnn.open_device(device_id=args.device_id,
                              **fused.open_device_kwargs(l1_small_size=32 * 1024))
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    try:
        if args.layer not in LAYER_DISPATCH:
            print_result(args.layer, 0.0, 0.0, "crash")
            print(f"ERROR: unknown layer '{args.layer}'")
            return 2
        try:
            ref_out, tt_out, latency_ms = LAYER_DISPATCH[args.layer](device, args.runs)
            pcc_val = pcc(ref_out, tt_out)
            peak = device_dram_mb(device)
            status = "PASS" if pcc_val >= 0.99 else "FAIL"
            print_result(args.layer, pcc_val, latency_ms, status, peak)
            return 0 if status == "PASS" else 1
        except Exception:
            traceback.print_exc()
            print_result(args.layer, 0.0, 0.0, "crash")
            return 3
    finally:
        # Drop the port's device-resident caches (weights, LUTs, and with TT_FUSED=1 the
        # metal trace + persistent input / output buffers) BEFORE closing the device;
        # otherwise ttnn frees them against a closed device at interpreter exit.
        try:
            release_device_caches()
        except Exception:
            traceback.print_exc()
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
