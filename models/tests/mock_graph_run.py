# SPDX-License-Identifier: Apache-2.0
"""Run the DUSt3R device graph on the host against ``fake_ttnn`` and print op counts.

    python models/tests/mock_graph_run.py            # honours TT_FUSED & sub-knobs from the env

Installs ``fake_ttnn`` as ``sys.modules["ttnn"]`` *before* importing ``ttnn_dust3r``, builds a
state dict of empty bf16 tensors with the checkpoint's shapes, calls ``dust3r_forward``
twice (1st: caches / eager warm run / trace capture, 2nd: steady state) and prints one JSON
line: op counts of both calls, op counts recorded INSIDE the trace capture, the list of
violations (host<->device transfers or host conv weights inside the capture) and the
resolved ``FusedConfig``. Consumed by ``test_fused_host.py``; no device, no numerics.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import torch  # noqa: E402

from models.tests import fake_ttnn  # noqa: E402

sys.modules["ttnn"] = fake_ttnn
from models.demos.mast3r.tt import ttnn_dust3r as port  # noqa: E402  (imports the fake)

ENC_D, DEC_D, ENC_DEPTH, DEC_DEPTH = 1024, 768, 24, 12


def fake_state() -> dict:
    e = lambda *shape: torch.empty(*shape, dtype=torch.bfloat16)  # noqa: E731
    s = {"patch_embed.proj.weight": e(ENC_D, 3, 16, 16), "patch_embed.proj.bias": e(ENC_D),
         "enc_norm.weight": e(ENC_D), "enc_norm.bias": e(ENC_D),
         "decoder_embed.weight": e(DEC_D, ENC_D), "decoder_embed.bias": e(DEC_D),
         "dec_norm.weight": e(DEC_D), "dec_norm.bias": e(DEC_D)}
    for i in range(ENC_DEPTH):
        p = f"enc_blocks.{i}."
        for n in ("norm1", "norm2"):
            s[p + n + ".weight"], s[p + n + ".bias"] = e(ENC_D), e(ENC_D)
        s[p + "attn.qkv.weight"], s[p + "attn.qkv.bias"] = e(3 * ENC_D, ENC_D), e(3 * ENC_D)
        s[p + "attn.proj.weight"], s[p + "attn.proj.bias"] = e(ENC_D, ENC_D), e(ENC_D)
        s[p + "mlp.fc1.weight"], s[p + "mlp.fc1.bias"] = e(4 * ENC_D, ENC_D), e(4 * ENC_D)
        s[p + "mlp.fc2.weight"], s[p + "mlp.fc2.bias"] = e(ENC_D, 4 * ENC_D), e(ENC_D)
    for pre in ("dec_blocks.", "dec_blocks2."):
        for i in range(DEC_DEPTH):
            p = f"{pre}{i}."
            for n in ("norm1", "norm2", "norm3", "norm_y"):
                s[p + n + ".weight"], s[p + n + ".bias"] = e(DEC_D), e(DEC_D)
            s[p + "attn.qkv.weight"], s[p + "attn.qkv.bias"] = e(3 * DEC_D, DEC_D), e(3 * DEC_D)
            for n in ("attn.proj", "cross_attn.projq", "cross_attn.projk", "cross_attn.projv", "cross_attn.proj"):
                s[p + n + ".weight"], s[p + n + ".bias"] = e(DEC_D, DEC_D), e(DEC_D)
            s[p + "mlp.fc1.weight"], s[p + "mlp.fc1.bias"] = e(4 * DEC_D, DEC_D), e(4 * DEC_D)
            s[p + "mlp.fc2.weight"], s[p + "mlp.fc2.bias"] = e(DEC_D, 4 * DEC_D), e(DEC_D)
    for b in (1, 2):
        p = f"downstream_head{b}.dpt."
        s[p + "act_postprocess.0.0.weight"], s[p + "act_postprocess.0.0.bias"] = e(96, ENC_D, 1, 1), e(96)
        s[p + "act_postprocess.0.1.weight"], s[p + "act_postprocess.0.1.bias"] = e(96, 96, 4, 4), e(96)
        s[p + "act_postprocess.1.0.weight"], s[p + "act_postprocess.1.0.bias"] = e(192, DEC_D, 1, 1), e(192)
        s[p + "act_postprocess.1.1.weight"], s[p + "act_postprocess.1.1.bias"] = e(192, 192, 2, 2), e(192)
        s[p + "act_postprocess.2.0.weight"], s[p + "act_postprocess.2.0.bias"] = e(384, DEC_D, 1, 1), e(384)
        s[p + "act_postprocess.3.0.weight"], s[p + "act_postprocess.3.0.bias"] = e(DEC_D, DEC_D, 1, 1), e(DEC_D)
        s[p + "act_postprocess.3.1.weight"], s[p + "act_postprocess.3.1.bias"] = e(DEC_D, DEC_D, 3, 3), e(DEC_D)
        for i, cin in zip((1, 2, 3, 4), (96, 192, 384, 768)):
            s[p + f"scratch.layer{i}_rn.weight"] = e(256, cin, 3, 3)
        for r in (1, 2, 3, 4):
            for cu in (1, 2):
                for k in (1, 2):
                    s[p + f"scratch.refinenet{r}.resConfUnit{cu}.conv{k}.weight"] = e(256, 256, 3, 3)
                    s[p + f"scratch.refinenet{r}.resConfUnit{cu}.conv{k}.bias"] = e(256)
            s[p + f"scratch.refinenet{r}.out_conv.weight"], s[p + f"scratch.refinenet{r}.out_conv.bias"] = e(256, 256, 1, 1), e(256)
        s[p + "head.0.weight"], s[p + "head.0.bias"] = e(128, 256, 3, 3), e(128)
        s[p + "head.2.weight"], s[p + "head.2.bias"] = e(128, 128, 3, 3), e(128)
        s[p + "head.4.weight"], s[p + "head.4.bias"] = e(4, 128, 1, 1), e(4)
    return s


def capture_failure_probe() -> int:
    """``--fail-capture``: make the graph raise INSIDE the trace capture on the first call and
    check that ``TtDust3r.capture`` leaves nothing behind (no persistent input, no trace, the
    device out of capture mode) and that the next call captures normally with exactly one
    persistent buffer. Prints one JSON line."""
    state = fake_state()
    device = fake_ttnn.FakeDevice()
    img1, img2 = torch.zeros(1, 3, 512, 512), torch.zeros(1, 3, 512, 512)
    port.reset_fused_config()
    cfg = port.fused_config()
    model = port.get_model(state, device)
    real_graph = model._graph
    calls = [0]

    def failing_graph(tt_img):
        calls[0] += 1
        if calls[0] == 2:                      # 1 = eager warm run, 2 = inside the capture
            raise RuntimeError("synthetic capture failure")
        return real_graph(tt_img)

    model._graph = failing_graph
    fake_ttnn.reset()
    raised = None
    try:
        model(img1, img2)
    except RuntimeError as e:
        raised = str(e)
    failed = dict(fake_ttnn.CALLS)
    after_failure = {
        "raised": raised, "capturing": fake_ttnn.CAPTURING[0],
        "persistent_in": model._persistent_in is not None, "in_shape": model._in_shape,
        "trace_captured": model.trace_captured, "outs": model._outs is not None,
    }

    model._graph = real_graph
    fake_ttnn.reset()
    o1, o2 = model(img1, img2)
    retry = dict(fake_ttnn.CALLS)
    assert tuple(o1.shape) == tuple(o2.shape) == (1, 4, 512, 512)
    after_retry = {"persistent_in": model._persistent_in is not None, "trace_captured": model.trace_captured,
                   "capturing": fake_ttnn.CAPTURING[0], "violations": list(fake_ttnn.VIOLATIONS)}
    n_released = port.release_device_caches()
    print(json.dumps({"cfg": cfg.summary(), "failed": failed, "after_failure": after_failure,
                      "retry": retry, "after_retry": after_retry, "released": n_released,
                      "caches_empty": not port._MODEL_CACHE}))
    return 0


def main() -> int:
    if "--fail-capture" in sys.argv[1:]:
        return capture_failure_probe()
    state = fake_state()
    device = fake_ttnn.FakeDevice()
    img1, img2 = torch.zeros(1, 3, 512, 512), torch.zeros(1, 3, 512, 512)
    port.reset_fused_config()
    cfg = port.fused_config()

    fake_ttnn.reset()
    o1, o2 = port.dust3r_forward(img1, img2, state, device)
    first = dict(fake_ttnn.CALLS)
    captured = Counter(op for op, capturing, _ in fake_ttnn.LOG if capturing)
    conv_host_in_capture = sum(1 for op, cap, info in fake_ttnn.LOG
                               if cap and op in ("conv2d", "conv_transpose2d") and info["host_weight"])
    violations = list(fake_ttnn.VIOLATIONS)
    assert tuple(o1.shape) == tuple(o2.shape) == (1, 4, 512, 512), (o1.shape, o2.shape)

    captured_log = [(op, info) for op, cap, info in fake_ttnn.LOG if cap]

    fake_ttnn.reset()
    o1, o2 = port.dust3r_forward(img1, img2, state, device)
    second = dict(fake_ttnn.CALLS)
    violations += fake_ttnn.VIOLATIONS
    assert tuple(o1.shape) == (1, 4, 512, 512)

    # "graph" = the ops one forward issues on the device: the captured trace when tracing,
    # else the steady-state (second) call.
    graph_log = captured_log if captured_log else [(op, info) for op, _, info in fake_ttnn.LOG]
    graph = Counter(op for op, _ in graph_log)
    graph.pop("end_trace_capture", None)
    stats = {
        "conv_host_weight_calls": sum(1 for op, info in graph_log if op in ("conv2d", "conv_transpose2d") and info["host_weight"]),
        "sdpa_with_program_config": sum(1 for op, info in graph_log if op == "sdpa" and info["program_config"]),
        "conv2d_with_fused_relu": sum(1 for op, info in graph_log if op == "conv2d" and info["relu"]),
        "unaligned_slices": sum(1 for op, info in graph_log if op == "slice" and not info["aligned"]),
        "total_ttnn_calls": sum(graph.values()),
    }

    n_released = port.release_device_caches()
    model_released = not port._MODEL_CACHE and not port._FUSED_DEVICE_CACHE

    print(json.dumps({
        "cfg": cfg.summary(), "first": first, "second": second, "captured": dict(captured),
        "graph": dict(graph), "graph_stats": stats,
        "conv_host_weight_calls_in_capture": conv_host_in_capture,
        "violations": violations, "released": n_released, "caches_empty": model_released,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
