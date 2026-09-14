# SPDX-License-Identifier: Apache-2.0
"""Host-only (torch, no device) tests for the ``TT_FUSED=1`` path of the DUSt3R port.

Run from the repo's ``code/`` dir with the tree's python (pytest is available there)::

    cd code && PYTHONPATH=. TT_METAL_HOME=$TREE $TREE/python_env/bin/python -m pytest -q models/tests/test_fused_host.py

What is proven here (everything the device pass then only has to *time*):

* the two single-kernel RoPE reformulations (``rotary_embedding_llama`` with ``P_il`` +
  ``trans_mat``, legacy ``rotary_embedding`` with ``P_qs``) reproduce DUSt3R's RoPE2D
  attention scores to fp64 round-off, and the per-token outputs equal the permuted
  reference outputs exactly;
* the q/k weight/bias row permutation is a pure gather (``torch.equal``) and produces the
  permuted projection;
* the constant tables the port uploads (cos/sin LUT ``[1,1,N,64]``, ``trans_mat``
  ``[1,1,32,32]``, ones vectors) have the shapes/dtypes/values the ops validate;
* ``MAST3R_ROPE_LUT=bf16`` reproduces the legacy port's (angle-rounded) tables bit for bit;
* the other exact claims: device im2col == unfold, ones-vector residual == plain add,
  relu commutes with bf16 rounding (conv+relu fusion);
* knob plumbing: ``TT_FUSED=0`` -> legacy config, unset / ``1`` -> the device-validated fused
  default (llama RoPE, fp32 LUT, dit residual matmuls, SDPA 128/256, trace);
  ``ttnn_dust3r.fused_config()`` (if ttnn is importable) follows the same rule.

No ttnn tensor is created anywhere in this file (host ``from_torch`` opens the driver in
this tree). Only the last test imports ttnn at all, and it is skipped when unavailable.
"""
from __future__ import annotations

import inspect
import os
import sys

import pytest
import torch
import torch.nn.functional as F

_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from models.demos.mast3r.tt import fused as fz  # noqa: E402

DH, HEADS_ENC, HEADS_DEC, D_ENC, D_DEC, N, HP, WP = 64, 16, 12, 1024, 768, 1024, 32, 32
BASE = 100.0


def _positions(B: int) -> torch.Tensor:
    ys, xs = torch.arange(HP), torch.arange(WP)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gy, gx), dim=-1).reshape(HP * WP, 2).unsqueeze(0).expand(B, -1, -1).contiguous()


# ------------------------------------------------------------------ knob plumbing

def test_knob_off_is_legacy():
    """``TT_FUSED=0`` (the knob) is the legacy graph: every field at its legacy value."""
    for env in ({"TT_FUSED": "0"}, {"TT_FUSED": "false"}, {"TT_FUSED": "no"}, {"TT_FUSED": " 0 "}):
        cfg = fz.FusedConfig.from_env(env)
        assert cfg == fz.FusedConfig(), env
        assert not cfg.enabled and cfg.rope == "slices" and cfg.mm == "linear"
        assert cfg.sdpa_chunks is None and not cfg.trace and not cfg.dpt_fuse and not cfg.conv_weight_cache
        assert not cfg.fused_rope and not cfg.fused_residual_mm and not cfg.minimal_mm


def test_knob_unset_is_fused_default():
    """Unset / empty / ``1``: the device-validated default (2026-09-13)."""
    for env in ({}, {"TT_FUSED": ""}, {"TT_FUSED": "1"}, {"TT_FUSED": "true"}):
        cfg = fz.FusedConfig.from_env(env)
        assert cfg.enabled and cfg.rope == "llama" and cfg.rope_lut == "fp32" and cfg.fused_rope, env
        assert cfg.mm == fz.DEFAULT_MM_MODE == "dit" and cfg.fused_residual_mm and not cfg.minimal_mm
        assert cfg.sdpa_chunks == fz.DEFAULT_SDPA_CHUNKS == (128, 256)
        assert cfg.trace and cfg.trace_region_size == fz.DEFAULT_TRACE_REGION_SIZE
        assert cfg.dpt_fuse and cfg.conv_weight_cache
    assert fz.FusedConfig.from_env({}) == fz.FusedConfig.from_env({"TT_FUSED": "1"})


def test_knob_on_defaults():
    # Sub-knobs must be ignored while the master knob is off.
    off = fz.FusedConfig.from_env({"TT_FUSED": "0", "MAST3R_ROPE": "legacy", "MAST3R_FUSED_MM": "dit",
                                   "MAST3R_SDPA_CHUNKS": "128,128"})
    assert off == fz.FusedConfig()
    cfg = fz.FusedConfig.from_env({"TT_FUSED": "1"})
    assert cfg.enabled and cfg.rope == "llama" and cfg.rope_lut == "fp32" and cfg.fused_rope
    assert cfg.mm == "dit" and cfg.fused_residual_mm and cfg.sdpa_chunks == (128, 256)
    assert cfg.trace and cfg.trace_region_size == fz.DEFAULT_TRACE_REGION_SIZE
    assert cfg.dpt_fuse and cfg.conv_weight_cache
    assert isinstance(cfg.summary(), dict) and cfg.summary()["rope"] == "llama"
    # The pre-validation fused configuration (plain ttnn.linear + add, ttnn's SDPA chunks) is
    # still reachable for A/B attribution.
    plain = fz.FusedConfig.from_env({"TT_FUSED": "1", "MAST3R_FUSED_MM": "linear", "MAST3R_SDPA_CHUNKS": "default"})
    assert plain.mm == "linear" and not plain.fused_residual_mm and plain.sdpa_chunks is None
    for word in fz.SDPA_DEFAULT_WORDS:
        assert fz.FusedConfig.from_env({"TT_FUSED": "1", "MAST3R_SDPA_CHUNKS": word}).sdpa_chunks is None


def test_knob_sub_knobs():
    env = {"TT_FUSED": "1", "MAST3R_ROPE": "legacy", "MAST3R_ROPE_LUT": "bf16", "MAST3R_FUSED_MM": "minimal",
           "MAST3R_SDPA_CHUNKS": "128,256", "MAST3R_TRACE": "0", "MAST3R_TRACE_REGION": "268435456",
           "MAST3R_DPT_FUSE": "0"}
    cfg = fz.FusedConfig.from_env(env)
    assert cfg.rope == "legacy" and cfg.rope_lut == "bf16" and cfg.mm == "minimal" and cfg.minimal_mm
    assert cfg.fused_residual_mm and cfg.sdpa_chunks == (128, 256) and not cfg.trace
    assert cfg.trace_region_size == 256 * 1024 * 1024 and not cfg.dpt_fuse and cfg.conv_weight_cache
    assert fz.FusedConfig.from_env({"TT_FUSED": "1", "MAST3R_ROPE": "slices"}).rope == "slices"
    assert fz.FusedConfig.from_env({"TT_FUSED": "1", "MAST3R_FUSED_MM": "dit"}).fused_residual_mm
    for bad in ({"MAST3R_ROPE": "x"}, {"MAST3R_FUSED_MM": "x"}, {"MAST3R_SDPA_CHUNKS": "100,128"},
                {"MAST3R_SDPA_CHUNKS": "128"}, {"MAST3R_TRACE_REGION": "0"}, {"MAST3R_ROPE_LUT": "x"}):
        with pytest.raises(ValueError):
            fz.FusedConfig.from_env({"TT_FUSED": "1", **bad})
    assert fz.parse_sdpa_chunks(None) is None and fz.parse_sdpa_chunks("") is None
    assert fz.parse_sdpa_chunks(None, default=(128, 256)) == (128, 256) and fz.parse_sdpa_chunks("", default=(64, 64)) == (64, 64)
    assert fz.parse_sdpa_chunks("default", default=(128, 256)) is None and fz.parse_sdpa_chunks("0") is None
    assert fz.parse_sdpa_chunks("128x128") == (128, 128)


# ------------------------------------------------------------- permutation tables

def test_channel_perms_are_permutations():
    for mode in ("llama", "legacy"):
        p = fz.rope_channel_perm(DH, mode)
        assert p.shape == (DH,) and p.dtype == torch.long
        assert torch.equal(torch.sort(p).values, torch.arange(DH))
    il = fz.rope_channel_perm(DH, "llama").tolist()
    assert il[:8] == [0, 16, 1, 17, 2, 18, 3, 19] and il[30:36] == [15, 31, 32, 48, 33, 49] and il[-2:] == [47, 63]
    # each 32-wide half maps onto itself (the trans_mat rotates within one 32-wide tile)
    assert set(il[:32]) == set(range(32)) and set(il[32:]) == set(range(32, 64))
    qs = fz.rope_channel_perm(DH, "legacy").tolist()
    assert qs == list(range(0, 16)) + list(range(32, 48)) + list(range(16, 32)) + list(range(48, 64))
    ex = fz.expand_head_perm(fz.rope_channel_perm(DH, "llama"), HEADS_ENC)
    assert ex.shape == (D_ENC,) and torch.equal(torch.sort(ex).values, torch.arange(D_ENC))
    assert torch.equal(ex[DH:2 * DH] - DH, ex[:DH])
    with pytest.raises(ValueError):
        fz.rope_channel_perm(DH, "slices")


def test_trans_mat():
    T = fz.rope_trans_mat()
    assert T.shape == (1, 1, 32, 32) and T.dtype == torch.float32
    assert set(T.unique().tolist()) == {-1.0, 0.0, 1.0}
    assert torch.equal(T.to(torch.bfloat16).float(), T)             # exact in bf16
    T2 = T[0, 0]
    assert torch.equal(T2 @ T2, -torch.eye(32))                       # a rotation by 90 degrees per pair
    x = torch.randn(5, 32)
    r = x @ T2
    assert torch.equal(r[:, 0::2], -x[:, 1::2]) and torch.equal(r[:, 1::2], x[:, 0::2])
    assert (T2 != 0).sum(dim=1).eq(1).all()                          # one nonzero per row/column: exact in HiFi4


# --------------------------------------------------------- RoPE identities (fp64)

@pytest.mark.parametrize("angle_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["llama", "legacy"])
def test_rope_reformulation_matches_reference_scores(mode, angle_dtype):
    torch.manual_seed(0)
    B, H = 2, 4
    pos = _positions(B)
    q = torch.randn(B, H, N, DH, dtype=torch.float64)
    k = torch.randn(B, H, N, DH, dtype=torch.float64)
    # reference: DUSt3R RoPE2D with the SAME angle rounding (the identity is about the
    # rotation structure, not the table precision)
    D = DH // 2
    cos1, sin1 = fz.rope_cos_sin_tables(D, HP, BASE, angle_dtype)
    cos1, sin1 = cos1.double(), sin1.double()

    def ref_rope(t):
        def rot(x):
            a, b = x.chunk(2, dim=-1)
            return torch.cat((-b, a), dim=-1)
        y, x = t[..., :D], t[..., D:]
        cy, sy = cos1[pos[..., 0]][:, None], sin1[pos[..., 0]][:, None]
        cx, sx = cos1[pos[..., 1]][:, None], sin1[pos[..., 1]][:, None]
        return torch.cat((y * cy + rot(y) * sy, x * cx + rot(x) * sx), dim=-1)

    ref_scores = ref_rope(q) @ ref_rope(k).transpose(-2, -1)

    P = fz.rope_channel_perm(DH, mode)
    cos, sin = fz.rope_lut_permuted(pos[0], DH, mode, BASE, angle_dtype, out_dtype=torch.float64)
    assert cos.shape == (1, 1, N, DH) and sin.shape == (1, 1, N, DH)
    qp, kp = q[..., P], k[..., P]                       # what the folded weights produce
    if mode == "llama":
        T = fz.rope_trans_mat()
        qo, ko = fz.rope_llama_emulated(qp, cos, sin, T), fz.rope_llama_emulated(kp, cos, sin, T)
    else:
        qo, ko = fz.rope_legacy_emulated(qp, cos, sin), fz.rope_legacy_emulated(kp, cos, sin)
    # (a) per-token: fused output == permuted reference output (same elementwise products)
    assert torch.equal(qo, ref_rope(q)[..., P]) and torch.equal(ko, ref_rope(k)[..., P])
    # (b) scores invariant under the shared channel permutation
    d = (qo @ ko.transpose(-2, -1) - ref_scores).abs().max().item()
    assert d < 1e-12, d


def test_rope2d_reference_copy_matches_port_reference():
    """``fz.rope2d_reference`` is a self-contained copy of ``torch_dust3r.RoPE2D``; check it
    against the real one when the reference module is importable (it needs safetensors /
    huggingface_hub at import time)."""
    try:
        from models.demos.mast3r.reference.torch_dust3r import RoPE2D
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reference module not importable here: {e}")
    torch.manual_seed(1)
    pos = _positions(1)
    t = torch.randn(1, 3, N, DH)
    assert torch.equal(RoPE2D(BASE)(t, pos), fz.rope2d_reference(t, pos, BASE))


# ----------------------------------------------------------------- weight folds

@pytest.mark.parametrize("mode", ["llama", "legacy"])
def test_weight_row_fold_is_exact(mode):
    torch.manual_seed(2)
    x = torch.randn(3, N, D_DEC, dtype=torch.float64)
    P = fz.expand_head_perm(fz.rope_channel_perm(DH, mode), HEADS_DEC)

    qkv_w, qkv_b = torch.randn(3 * D_DEC, D_DEC, dtype=torch.float64), torch.randn(3 * D_DEC, dtype=torch.float64)
    w2, b2 = fz.permute_qkv_rows(qkv_w, qkv_b, HEADS_DEC, mode)
    idx = torch.cat([P, P + D_DEC, torch.arange(2 * D_DEC, 3 * D_DEC)])
    assert torch.equal(w2, qkv_w[idx]) and torch.equal(b2, qkv_b[idx])          # pure row gather
    assert torch.equal(w2[2 * D_DEC:], qkv_w[2 * D_DEC:])                      # v rows untouched
    ref = x @ qkv_w.t() + qkv_b
    got = x @ w2.t() + b2
    assert (got - ref[..., idx]).abs().max().item() < 1e-12                    # q/k columns permuted per head
    assert torch.allclose(got[..., 2 * D_DEC:], ref[..., 2 * D_DEC:], atol=1e-12)

    q_w, q_b = torch.randn(D_DEC, D_DEC, dtype=torch.float64), torch.randn(D_DEC, dtype=torch.float64)
    wq, bq = fz.permute_q_rows(q_w, q_b, HEADS_DEC, mode)
    assert torch.equal(wq, q_w[P]) and torch.equal(bq, q_b[P])

    kv_w, kv_b = torch.randn(2 * D_DEC, D_DEC, dtype=torch.float64), torch.randn(2 * D_DEC, dtype=torch.float64)
    wkv, bkv = fz.permute_kv_rows(kv_w, kv_b, HEADS_DEC, mode)
    assert torch.equal(wkv[:D_DEC], kv_w[P]) and torch.equal(bkv[:D_DEC], kv_b[P])
    assert torch.equal(wkv[D_DEC:], kv_w[D_DEC:]) and torch.equal(bkv[D_DEC:], kv_b[D_DEC:])  # v half untouched

    # the encoder shape too (16 heads x 64 = 1024)
    W = torch.randn(3 * D_ENC, D_ENC)
    We, _ = fz.permute_qkv_rows(W, torch.zeros(3 * D_ENC), HEADS_ENC, mode)
    Pe = fz.expand_head_perm(fz.rope_channel_perm(DH, mode), HEADS_ENC)
    assert torch.equal(We[:D_ENC], W[Pe]) and torch.equal(We[D_ENC:2 * D_ENC], W[D_ENC + Pe])


def test_fold_composes_with_split_heads():
    """The permutation is within each head's 64-channel block, so it commutes with the
    ``(B, N, 3, H, dh)`` head split the device op performs on the fused qkv output."""
    torch.manual_seed(3)
    B, D, H = 1, D_DEC, HEADS_DEC
    qkv = torch.randn(B, N, 3 * D)
    P = fz.rope_channel_perm(DH, "llama")
    idx = torch.cat([fz.expand_head_perm(P, H), fz.expand_head_perm(P, H) + D, torch.arange(2 * D, 3 * D)])
    heads_ref = qkv.reshape(B, N, 3, H, DH).permute(2, 0, 3, 1, 4)          # (3, B, H, N, dh)
    heads_perm = qkv[..., idx].reshape(B, N, 3, H, DH).permute(2, 0, 3, 1, 4)
    assert torch.equal(heads_perm[0], heads_ref[0][..., P])
    assert torch.equal(heads_perm[1], heads_ref[1][..., P])
    assert torch.equal(heads_perm[2], heads_ref[2])


# -------------------------------------------------------------- constant tables

@pytest.mark.parametrize("mode", ["llama", "legacy"])
def test_lut_shape_dtype_values(mode):
    pos = _positions(1)[0]
    cos, sin = fz.rope_lut_permuted(pos, DH, mode, BASE)
    for t in (cos, sin):
        assert t.shape == (1, 1, N, DH) and t.dtype == torch.bfloat16 and t.is_contiguous()
        assert t.float().abs().max().item() <= 1.0
    assert torch.equal(cos[0, 0, 0], torch.ones(DH, dtype=torch.bfloat16))     # pos (0,0): cos=1, sin=0
    assert torch.equal(sin[0, 0, 0], torch.zeros(DH, dtype=torch.bfloat16))
    c = cos.float()[0, 0]
    if mode == "llama":     # pair structure the kernel expects: cos[2i] == cos[2i+1]
        assert torch.equal(c[:, 0::2], c[:, 1::2])
        # first half from pos_y, second from pos_x: tokens in one row share the y tables
        assert torch.equal(c[0:WP, :32], c[0:1, :32].expand(WP, -1)) and not torch.equal(c[0, 32:], c[1, 32:])
    else:                   # quarter swap: [cy, cx, cy, cx]
        assert torch.equal(c[:, 0:16], c[:, 32:48]) and torch.equal(c[:, 16:32], c[:, 48:64])
    # tile arithmetic the ops validate (no padding on head_dim, tile-aligned seq)
    assert DH % 32 == 0 and DH <= 128 and N % 32 == 0 and DH % 64 == 0


def test_lut_bf16_angle_reproduces_legacy_port_tables():
    """Legacy ``ttnn_dust3r._rope_cos_sin`` rounds the angle to bf16 BEFORE cos/sin
    (``emb.to(dtype)`` then ``.cos()``); ``MAST3R_ROPE_LUT=bf16`` must reproduce those tables
    exactly (A/B attribution), while ``fp32`` matches the reference's fp32 angles."""
    D, max_pos = DH // 2, HP
    # verbatim legacy math (dtype = bfloat16 as the device LUT builder passes it)
    inv_freq = 1.0 / (BASE ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    seq = torch.arange(max_pos, dtype=torch.float32)
    emb = torch.cat((torch.einsum("i,j->ij", seq, inv_freq),) * 2, dim=-1).to(torch.bfloat16)
    legacy_cos, legacy_sin = emb.cos(), emb.sin()
    cos_b, sin_b = fz.rope_cos_sin_tables(D, max_pos, BASE, torch.bfloat16)
    assert torch.equal(cos_b.to(torch.bfloat16), legacy_cos) and torch.equal(sin_b.to(torch.bfloat16), legacy_sin)
    cos_f, _ = fz.rope_cos_sin_tables(D, max_pos, BASE, torch.float32)
    assert not torch.equal(cos_f.to(torch.bfloat16), legacy_cos)              # the legacy quirk is real
    # magnitude of the legacy angle error (documentation of the accuracy risk, not a gate)
    ang_err = (torch.cat((torch.einsum("i,j->ij", seq, inv_freq),) * 2, -1) - emb.float()).abs().max().item()
    assert 0.03 < ang_err < 0.07, ang_err


def test_ones_vector_residual_formulation():
    torch.manual_seed(4)
    for Nn in (D_ENC, D_DEC):
        ones = fz.residual_scale_ones(Nn)
        assert ones.shape == (1, 1, Nn) and Nn % 32 == 0
        h, W = torch.randn(2, 64, 4 * Nn, dtype=torch.float64), torch.randn(4 * Nn, Nn, dtype=torch.float64)
        b, res = torch.randn(1, 1, Nn, dtype=torch.float64), torch.randn(2, 64, Nn, dtype=torch.float64)
        y = h @ W + b
        assert torch.equal(res + 1.0 * y * ones.double(), res + y)           # x*1.0 is exact


def test_minimal_matmul_block_divisibility():
    """MinimalMatmulConfig(8,4,4,2,2) (qkv/fc1/cq/ckv) and (4,4,4,2,2) (dit residual): every
    M / K / N tile count of the model's matmuls divides the block sizes, the subblocks
    divide the blocks, and biases are tile-aligned."""
    shapes = {  # (M, K, N) after folding batch into M
        "enc_qkv": (2048, 1024, 3072), "enc_proj": (2048, 1024, 1024), "enc_fc1": (2048, 1024, 4096),
        "enc_fc2": (2048, 4096, 1024), "dec_qkv": (1024, 768, 2304), "dec_proj": (1024, 768, 768),
        "dec_cq": (1024, 768, 768), "dec_ckv": (1024, 768, 1536), "dec_cproj": (1024, 768, 768),
        "dec_fc1": (1024, 768, 3072), "dec_fc2": (1024, 3072, 768),
    }
    for name, (M, K, Nn) in shapes.items():
        for mb, kb, nb, sh, sw in ((8, 4, 4, 2, 2), (4, 4, 4, 2, 2)):
            assert mb % sh == 0 and nb % sw == 0
            assert (M // 32) % mb == 0 and (K // 32) % kb == 0 and (Nn // 32) % nb == 0, name
        assert Nn % 32 == 0


def test_sdpa_work_items():
    """SDPA program config 128/128: q chunks per (b, h) and total work items on a ~110-core grid."""
    for B, H in ((2, HEADS_ENC), (1, HEADS_DEC)):
        assert N % 128 == 0
        items = B * H * (N // 128)
        assert items in (256, 96)


# --------------------------------------------------------- other exact reformulations

def test_device_im2col_matches_unfold_and_conv():
    torch.manual_seed(5)
    img = torch.randn(2, 3, 64, 64, dtype=torch.float64)
    w = torch.randn(8, 3, 16, 16, dtype=torch.float64)
    b = torch.randn(8, dtype=torch.float64)
    cols = fz.patch_im2col(img, 16)                                       # (B, N, 768)
    assert torch.equal(cols, F.unfold(img, 16, stride=16).transpose(1, 2))
    lin = cols @ w.reshape(8, -1).t() + b
    conv = F.conv2d(img, w, b, stride=16).flatten(2).transpose(1, 2)
    assert torch.allclose(lin, conv, atol=1e-12)


def test_relu_commutes_with_bf16_rounding():
    """``Conv2dConfig(activation=relu)`` applies relu before the bf16 pack; the legacy path
    applies ``ttnn.relu`` after. Equal (round-to-nearest preserves sign; only the sign of
    zero can differ, which ``torch.equal`` treats as equal)."""
    torch.manual_seed(6)
    x = torch.randn(1 << 16) * 3
    x[:64] = torch.linspace(-1e-3, 1e-3, 64)                                # values that round to +-0
    assert torch.equal(torch.relu(x).to(torch.bfloat16), torch.relu(x.to(torch.bfloat16)))


def test_tile_reshape_of_taps_is_a_view():
    """``ttnn.reshape`` is a 0-cost view in TILE when the last two dims do not change:
    ``(1, N, C) -> (1, 1, N, C)`` keeps them (the legacy path went RM and back)."""
    for C in (1024, 768):
        src, dst = (1, N, C), (1, 1, N, C)
        assert src[-2:] == dst[-2:] and N % 32 == 0 and C % 32 == 0


# ------------------------------------------ the device graph on a fake ttnn (no device)

_HARNESS = os.path.join(os.path.dirname(__file__), "mock_graph_run.py")


def _run_graph(*argv, **env):
    """Run ``mock_graph_run.py`` in a subprocess (it replaces ``ttnn`` with ``fake_ttnn``)."""
    import json
    import subprocess

    e = {k: v for k, v in os.environ.items() if not (k == "TT_FUSED" or k.startswith("MAST3R_"))}
    e.update(env)
    e["PYTHONPATH"] = _CODE_ROOT
    r = subprocess.run([sys.executable, _HARNESS, *argv], env=e, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1])


ENC_BLOCKS, DEC_BLOCKS = 24, 24                      # 24 encoder blocks; 12 layers x 2 branches
ROPE_APPS = ENC_BLOCKS * 2 + DEC_BLOCKS * 4          # 144 q/k rotations per pair
SDPA_CALLS = ENC_BLOCKS + DEC_BLOCKS * 2             # 72
RESIDUAL_LINEARS = ENC_BLOCKS * 2 + DEC_BLOCKS * 3   # 120 (proj/fc2; proj/cproj/fc2)
PLAIN_LINEARS = ENC_BLOCKS * 2 + DEC_BLOCKS * 4      # 144 (qkv/fc1; qkv/cq/ckv/fc1)
OTHER_LINEARS = 1 + 2 + 9 * 2                        # patch embed, decoder embed x2, DPT 1x1 x9 per head


def test_graph_legacy_when_knob_off():
    """``TT_FUSED=0``: the legacy graph is untouched: per forward the image is uploaded inside
    the graph, RoPE is 4 slices + 2 neg + concat + 2 mul + add per application (none of the
    four 16-wide quarter slices is tile-aligned: they start or end at column 16 / 48, so the
    TILE slice falls back to row-major), every conv receives HOST weights, no trace."""
    d = _run_graph(TT_FUSED="0")
    assert d["cfg"]["enabled"] is False and not d["violations"] and d["caches_empty"]
    g, s = d["graph"], d["graph_stats"]
    assert "begin_trace_capture" not in d["first"] and "rotary_embedding_llama" not in g
    assert g["from_torch"] == 1 and g["to_torch"] == 2
    assert g["slice"] == 4 * ROPE_APPS + 2 and g["neg"] == 2 * ROPE_APPS and g["concat"] == ROPE_APPS
    assert g["mul"] == 2 * ROPE_APPS and s["unaligned_slices"] == 4 * ROPE_APPS
    assert g["sdpa"] == SDPA_CALLS and s["sdpa_with_program_config"] == 0
    assert g["linear"] == RESIDUAL_LINEARS + PLAIN_LINEARS + OTHER_LINEARS
    assert g["add"] == ROPE_APPS + RESIDUAL_LINEARS + 2 * 10
    assert g["conv2d"] == 2 * 21 and g["conv_transpose2d"] == 2 * 2 and s["conv_host_weight_calls"] == 46
    assert g["relu"] == 2 * 15 and g["to_layout"] == 2 * 18 + 1 and s["conv2d_with_fused_relu"] == 0
    assert s["total_ttnn_calls"] == 2395 + 2                                    # +2: dec_norm on device (tap fix 2026-09-14)


def _check_fused_trace_shape(d):
    """Common assertions of every traced fused configuration: one ``rotary_embedding_llama``
    per q/k, exact DPT cleanups, cached conv weights, the whole graph in one trace with nothing
    crossing host<->device inside the capture; steady state = copy-in + execute_trace + 2
    readbacks."""
    c = d["cfg"]
    assert c["enabled"] and c["rope"] == "llama" and c["trace"]
    assert not d["violations"] and d["caches_empty"] and d["conv_host_weight_calls_in_capture"] == 0
    assert d["first"]["begin_trace_capture"] == 1 and d["first"]["end_trace_capture"] == 1
    assert d["first"]["to_device"] == 1 and d["first"]["execute_trace"] == 1
    assert d["second"] == {"copy_host_to_device_tensor": 1, "execute_trace": 1, "from_torch": 1, "to_torch": 2}
    g, s = d["graph"], d["graph_stats"]
    assert g["rotary_embedding_llama"] == ROPE_APPS and "rotary_embedding" not in g
    assert "neg" not in g and "concat" not in g and "mul" not in g and g["slice"] == 2 and s["unaligned_slices"] == 0
    assert "from_torch" not in g and "to_device" not in g and "to_torch" not in g
    assert g["sdpa"] == SDPA_CALLS and "minimal_matmul" not in g
    assert g["conv2d"] == 42 and g["conv_transpose2d"] == 4 and s["conv_host_weight_calls"] == 0
    assert g["relu"] == 2 * 7 and s["conv2d_with_fused_relu"] == 2 * 8          # 7 resconv pre-activations/head stay
    assert g["to_layout"] == 2 * 10 + 1                                          # 8 tap conversions/head gone
    return g, s


def test_graph_default_is_traced_dit_sdpa():
    """Nothing set (the served default since 2026-09-13): the traced single-kernel-RoPE graph
    with the 120 residual linears as ``dit_minimal_matmul_addcmul_fused`` and the explicit
    SDPA program config on all 72 calls -- 946 ttnn calls per pair (944 before the
    2026-09-14 tap fix: the last DPT tap is now the dec_norm'ed block-11 output, +2 layer_norm)."""
    d = _run_graph()
    c = d["cfg"]
    assert c["mm"] == "dit" and c["sdpa_chunks"] == [128, 256]
    g, s = _check_fused_trace_shape(d)
    assert g["dit_minimal_matmul_addcmul_fused"] == RESIDUAL_LINEARS
    assert g["linear"] == PLAIN_LINEARS + OTHER_LINEARS and "gelu" not in g
    assert s["sdpa_with_program_config"] == SDPA_CALLS
    assert g["add"] == 2 * 10
    assert s["total_ttnn_calls"] == 1066 - RESIDUAL_LINEARS
    assert _run_graph(TT_FUSED="1")["graph"] == g                             # TT_FUSED=1 == unset


def test_graph_fused_plain_linear_default_sdpa():
    """``MAST3R_FUSED_MM=linear MAST3R_SDPA_CHUNKS=default``: the pre-validation fused graph
    (plain ``ttnn.linear`` + ``add``, ttnn's own SDPA chunks), 1 064 calls -- the A/B
    reference for the two device-gated sub-knobs."""
    d = _run_graph(TT_FUSED="1", MAST3R_FUSED_MM="linear", MAST3R_SDPA_CHUNKS="default")
    c = d["cfg"]
    assert c["mm"] == "linear" and c["sdpa_chunks"] is None
    g, s = _check_fused_trace_shape(d)
    assert g["linear"] == RESIDUAL_LINEARS + PLAIN_LINEARS + OTHER_LINEARS and "dit_minimal_matmul_addcmul_fused" not in g
    assert s["sdpa_with_program_config"] == 0
    assert g["add"] == RESIDUAL_LINEARS + 2 * 10
    assert s["total_ttnn_calls"] == 1066


def test_graph_fused_all_sub_knobs():
    """``MAST3R_FUSED_MM=minimal`` + ``MAST3R_SDPA_CHUNKS=128,128`` + ``MAST3R_ROPE=legacy``:
    every fused branch runs, shapes pass the ops' validate rules mirrored in fake_ttnn."""
    d = _run_graph(TT_FUSED="1", MAST3R_FUSED_MM="minimal", MAST3R_SDPA_CHUNKS="128,128",
                   MAST3R_ROPE="legacy", MAST3R_ROPE_LUT="bf16")
    assert not d["violations"] and d["caches_empty"]
    g, s = d["graph"], d["graph_stats"]
    assert g["rotary_embedding"] == ROPE_APPS and "rotary_embedding_llama" not in g
    assert g["dit_minimal_matmul_addcmul_fused"] == RESIDUAL_LINEARS
    assert g["minimal_matmul"] == PLAIN_LINEARS and g["gelu"] == ENC_BLOCKS + DEC_BLOCKS
    assert g["linear"] == OTHER_LINEARS and g["add"] == 2 * 10
    assert g["sdpa"] == SDPA_CALLS and s["sdpa_with_program_config"] == SDPA_CALLS
    assert s["total_ttnn_calls"] == 1066 - RESIDUAL_LINEARS + (ENC_BLOCKS + DEC_BLOCKS)


def test_graph_fused_eager_dit():
    """``MAST3R_TRACE=0``: the same fused graph runs eagerly (A/B path), ``dit`` fuses only the
    residual linears and keeps ``ttnn.linear`` for qkv/fc1/cq/ckv."""
    d = _run_graph(TT_FUSED="1", MAST3R_TRACE="0", MAST3R_FUSED_MM="dit")
    assert not d["violations"] and "begin_trace_capture" not in d["first"]
    g = d["graph"]
    assert g["to_device"] == 1 and g["to_torch"] == 2 and g["rotary_embedding_llama"] == ROPE_APPS
    assert g["dit_minimal_matmul_addcmul_fused"] == RESIDUAL_LINEARS and "minimal_matmul" not in g
    assert g["linear"] == PLAIN_LINEARS + OTHER_LINEARS and "gelu" not in g


def test_capture_failure_leaves_no_persistent_buffer():
    """A graph error INSIDE the trace capture must not leak ``TtDust3r._persistent_in`` (the
    next call re-enters ``capture``), must leave the device out of capture mode and release
    the half-built trace; the retry then captures normally with exactly one persistent
    input allocation (review finding, fake-ttnn)."""
    d = _run_graph("--fail-capture", TT_FUSED="1")
    assert d["cfg"]["trace"]
    f, a = d["failed"], d["after_failure"]
    assert a["raised"] == "synthetic capture failure"
    assert f["to_device"] == 1 and f["begin_trace_capture"] == 1 and f["end_trace_capture"] == 1
    assert f["release_trace"] == 1 and f["deallocate"] >= 1 + 2       # persistent input + the 2 warm outputs
    assert not a["capturing"] and not a["persistent_in"] and a["in_shape"] is None
    assert not a["trace_captured"] and not a["outs"]
    r, b = d["retry"], d["after_retry"]
    assert r["to_device"] == 1 and r["begin_trace_capture"] == 1 and r["end_trace_capture"] == 1
    assert "release_trace" not in r and b["trace_captured"] and b["persistent_in"] and not b["capturing"]
    assert not b["violations"] and d["caches_empty"] and d["released"] >= 3     # input + 2 outputs


# ------------------------------------------------- knob plumbing in the device module

def test_open_device_kwargs_select_execution_mode():
    """The scripts that run the accuracy gates (harness, eval, server) open the device through
    ``FusedConfig.open_device_kwargs``: legacy and ``MAST3R_TRACE=0`` keep the plain open,
    the traced fused path adds the explicit trace region (this tree's default 0 would select
    the dynamic trace-allocation mode instead)."""
    base = dict(device_id=0, l1_small_size=32 * 1024)
    assert fz.FusedConfig().open_device_kwargs(**base) == base
    assert fz.FusedConfig.from_env({"TT_FUSED": "0"}).open_device_kwargs(**base) == base
    assert fz.FusedConfig.from_env({"MAST3R_TRACE": "0"}).open_device_kwargs(**base) == base
    on = fz.FusedConfig.from_env({"TT_FUSED": "1"}).open_device_kwargs(**base)
    assert on == {**base, "trace_region_size": fz.DEFAULT_TRACE_REGION_SIZE}
    assert fz.FusedConfig.from_env({}).open_device_kwargs(**base) == on          # unset = traced default
    custom = fz.FusedConfig.from_env({"TT_FUSED": "1", "MAST3R_TRACE_REGION": "1073741824"}).open_device_kwargs()
    assert custom == {"trace_region_size": 1 << 30}


def test_harness_and_eval_open_the_device_in_the_reported_mode():
    """``test_mast3r.py`` exposes the on-device DPT head as its own layer (the ``dpt_head``
    layer is the torch bf16 host reference and runs no ttnn op) and, like ``eval_mast3r.py``
    and the server, opens the device via ``open_device_kwargs`` and releases the port's
    device caches before ``close_device``. Source-level checks: the scripts import ttnn only
    inside ``main`` and are never executed here."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("mast3r_harness", os.path.join(_CODE_ROOT, "test_mast3r.py"))
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)                         # torch + reference imports only
    for layer in ("dpt_head_device", "dpt_head_device_2", "full_encoder", "full_decoder", "end_to_end"):
        assert layer in harness.LAYER_DISPATCH, layer

    scripts = ["test_mast3r.py", "eval_mast3r.py", os.path.join("models", "server", "app.py")]
    for script in scripts:
        if script.startswith("models") and not os.path.exists(os.path.join(_CODE_ROOT, script)):
            continue  # the FastAPI serving app ships only in the HF package (changh95/mast3r-p150), not here
        src = open(os.path.join(_CODE_ROOT, script)).read()
        assert "open_device_kwargs(" in src, script
        assert "release_device_caches()" in src and "close_device(" in src, script
        assert src.index("release_device_caches()") < src.rindex("close_device("), script
    harness_src = open(os.path.join(_CODE_ROOT, "test_mast3r.py")).read()
    assert "dpt_head_device as tt_dpt_head_device" in harness_src


def test_ttnn_dust3r_selects_legacy_when_knob_off(monkeypatch):
    ttnn = pytest.importorskip("ttnn")  # import only; no tensor is created
    del ttnn
    from models.demos.mast3r.tt import ttnn_dust3r as port

    for fn in (port.encoder_block_device_pre, port.decoder_block_device_pre, port.full_encoder,
               port.full_decoder, port.dpt_head_device):
        sig = inspect.signature(fn)
        assert "cfg" in sig.parameters and sig.parameters["cfg"].default is None, fn.__name__
    assert inspect.signature(port.dust3r_forward).parameters.keys() == {"img1", "img2", "state", "device"}
    assert callable(port.release_device_caches) and callable(port.get_model)

    monkeypatch.setenv("TT_FUSED", "0")
    port.reset_fused_config()
    cfg = port.fused_config()
    assert cfg == fz.FusedConfig() and not cfg.enabled
    assert port.fused_config() is cfg                                      # read once

    monkeypatch.delenv("TT_FUSED", raising=False)
    assert port.fused_config() is cfg                                      # still the first read
    port.reset_fused_config()
    on = port.fused_config()                                               # unset = fused default
    assert on.enabled and on.rope == "llama" and on.trace and on.mm == "dit" and on.sdpa_chunks == (128, 256)
    port.reset_fused_config()
