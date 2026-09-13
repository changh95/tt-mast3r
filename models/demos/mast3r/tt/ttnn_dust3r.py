"""TT-Metal (ttnn) implementation of DUSt3R layers.

Each function takes a device + torch weights/inputs and returns a torch tensor
on host so the test harness can compute PCC against the reference.

Two execution paths share this module:

* **Legacy (``TT_FUSED=0``; the default until the 2026-09-13 device validation)** -- the
  original eager graph: every call is a separate ttnn op, the image is uploaded inside
  ``patch_embed_device``, the DPT convs receive host weight tensors, RoPE is a 10-call
  slice/neg/concat/mul/add chain per q/k. ~2.4k ttnn calls per pair, 234 ms best-of-25 on
  the p150a. Nothing on this path changes when the knob is off; every fused branch below is
  guarded by an explicit ``cfg`` check.
* **Fused (default: ``TT_FUSED`` unset or ``1``, read ONCE by :func:`fused_config` at model
  build)** -- the same graph with the host-implementable "megakernel" levers of
  ``reports/megakernel/mast3r-p150.md`` (see :class:`models.demos.mast3r.tt.fused.FusedConfig`
  for every sub-knob), 944 ttnn calls per pair in one metal trace, 73 ms best-of-25 on the
  p150a (``DEVICE_VALIDATION.md`` "Results"):

  - 2-D RoPE as ONE kernel per q/k (``ttnn.experimental.rotary_embedding_llama`` prefill,
    or the legacy ``rotary_embedding`` op) after a per-head channel permutation that is
    folded into the q/k rows of the qkv / projq / projk|projv weights and biases at
    preload (exact row gather) and into the ``[1, 1, N, 64]`` cos/sin tables;
  - ``dit_minimal_matmul_addcmul_fused`` for the 120 residual linears (proj / cproj / fc2;
    ``MAST3R_FUSED_MM=dit``, the default) -- ``minimal`` additionally routes qkv / fc1 / cq /
    ckv through ``minimal_matmul`` (measured slower and less accurate here; knob only);
  - ``SDPAProgramConfig(q_chunk=128, k_chunk=256)`` on all 72 attention calls per pair (24
    encoder + 48 decoder; ``MAST3R_SDPA_CHUNKS``, default ``128,256``);
  - prepared conv2d / conv_transpose2d weights cached on device after the first call,
    TILE reshape of the four DPT taps, relu fused into the resconv conv1 / head2 convs;
  - :class:`TtDust3r`: persistent ``[2, 3, 512, 512]`` bf16 ROW_MAJOR input buffer,
    ``copy_host_to_device_tensor`` per call, the WHOLE device graph (patch embed ->
    encoder -> decoder -> both DPT heads) captured once into a metal trace after an eager
    warm run, ``execute_trace(blocking=False)`` + two ``to_torch`` readbacks per pair.

  ``dust3r_forward`` keeps its signature and dispatches to the wrapper when the knob is
  on. ``full_encoder`` / ``full_decoder`` / ``dpt_head_device`` (the per-layer harness
  entry points) honour the knob eagerly, so ``test_mast3r.py --layer full_encoder`` A/Bs
  the fused ops without the trace. Device validation plan: ``DEVICE_VALIDATION.md``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import ttnn

from models.demos.mast3r.tt.fused import (
    FusedConfig,
    permute_kv_rows,
    permute_q_rows,
    permute_qkv_rows,
    residual_scale_ones,
    rope_lut_permuted,
    rope_trans_mat,
)


def _t2d(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


# ---------- TT_FUSED knob (read once) and fused-path device constants ----------

_FUSED_CFG: Optional[FusedConfig] = None


def fused_config() -> FusedConfig:
    """The ``TT_FUSED`` knob, parsed from the environment ONCE (first use == model build)
    and cached for the process lifetime. Default (unset) is the legacy configuration."""
    global _FUSED_CFG
    if _FUSED_CFG is None:
        _FUSED_CFG = FusedConfig.from_env()
    return _FUSED_CFG


def reset_fused_config() -> None:
    """Forget the cached knob (tests only; a live model must not change configuration)."""
    global _FUSED_CFG
    _FUSED_CFG = None


_FUSED_DEVICE_CACHE: dict = {}   # (id(device), cfg) -> device constants of the fused path


def _fused_consts(device, cfg: FusedConfig) -> dict:
    """Per-device constants of the fused path, built lazily on first use (i.e. during the
    eager warm run, before any trace capture) and released by ``release_device_caches``:

    * ``trans_mat`` ``[1, 1, 32, 32]`` bf16 TILE and the RoPE compute config (HiFi4 +
      fp32 dest accumulation -- allowed for head_dim <= 128 -- so the three products are
      rounded once at pack) for ``rotary_embedding_llama``;
    * ``MinimalMatmulConfig`` blocks validated on this p150a in the rf-detr pass:
      (8, 4, 4, 2x2) for plain matmuls, (4, 4, 4, 2x2) for the fused residual op (the
      default 8x8x8 blocks need ~1.1 MB of L1 CBs per core), HiFi2 without fp32
      accumulation (the fp32/HiFi4 variants lowered rf-detr's task metric);
    * ``ones`` vectors ``[1, 1, N]`` in DRAM (the fused op needs the scale vector in the
      SAME buffer type as the residual; every residual here is DRAM interleaved);
    * ``SDPAProgramConfig(grid, q_chunk, k_chunk, exp_approx_mode=False)``.
    """
    key = (id(device), cfg)
    c = _FUSED_DEVICE_CACHE.get(key)
    if c is not None:
        return c
    c = {"ones": {}}
    arch = device.arch()
    grid = device.compute_with_storage_grid_size()
    if cfg.rope == "llama":
        c["trans_mat"] = ttnn.from_torch(rope_trans_mat(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        c["rope_ck"] = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=False,
        )
    if cfg.fused_residual_mm:
        c["mm_ck"] = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
            fp32_dest_acc_en=False, packer_l1_acc=True,
        )
        c["mm_cfg"] = ttnn.MinimalMatmulConfig(
            M_block_size=8, K_block_size=4, N_block_size=4, subblock_h=2, subblock_w=2,
            compute_with_storage_grid_size=grid,
        )
        c["dit_cfg"] = ttnn.MinimalMatmulConfig(
            M_block_size=4, K_block_size=4, N_block_size=4, subblock_h=2, subblock_w=2,
            compute_with_storage_grid_size=grid,
        )
    if cfg.sdpa_chunks is not None:
        q_chunk, k_chunk = cfg.sdpa_chunks
        c["sdpa_pc"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
    _FUSED_DEVICE_CACHE[key] = c
    return c


def _ones_vec(c: dict, N: int, device):
    """DRAM ``ones[1, 1, N]`` bf16 TILE for the fused residual op (created on first use =
    eager warm run, never inside a trace capture)."""
    t = c["ones"].get(N)
    if t is None:
        t = ttnn.from_torch(
            residual_scale_ones(N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        c["ones"][N] = t
    return t


def _mm(h, w, b, device, cfg: Optional[FusedConfig], activation: Optional[str] = None):
    """``h @ w + b`` (+ optional exact GELU) for qkv / fc1 / cq / ckv.

    Legacy: ``ttnn.linear`` (with ``activation="gelu"`` fused == ``UnaryOpType::GELU``,
    accurate variant). ``MAST3R_FUSED_MM=minimal``: ``ttnn.experimental.minimal_matmul``
    then a separate exact ``ttnn.gelu`` (fused GELU inside minimal_matmul was slower in the
    rf-detr pass: the SFPU serialises with math).
    """
    if cfg is not None and cfg.minimal_mm:
        c = _fused_consts(device, cfg)
        out = ttnn.experimental.minimal_matmul(
            h, w, bias_tensor=b, config=c["mm_cfg"], compute_kernel_config=c["mm_ck"]
        )
        return ttnn.gelu(out) if activation == "gelu" else out
    if activation is not None:
        return ttnn.linear(h, w, bias=b, activation=activation)
    return ttnn.linear(h, w, bias=b)


def _mm_residual(h, w, b, residual, device, cfg: Optional[FusedConfig]):
    """``residual + (h @ w + b)`` for proj / cproj / fc2.

    Legacy: ``ttnn.linear`` then ``ttnn.add(residual, y)`` (2 programs). ``MAST3R_FUSED_MM
    in (dit, minimal)``: one ``dit_minimal_matmul_addcmul_fused(h, w, 1.0, residual, ones)``
    == ``residual + 1.0 * (h @ w + b) * ones``; ``residual`` must be ``[..., M, N]`` == the
    matmul output shape (``[2, 1024, 1024]`` / ``[1, 1024, 768]`` here) and ``ones``
    ``[1, 1, N]`` in the residual's buffer type (DRAM).
    """
    if cfg is not None and cfg.fused_residual_mm:
        c = _fused_consts(device, cfg)
        N = int(w.shape[-1])
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            h, w, 1.0, residual, _ones_vec(c, N, device), bias_tensor=b,
            config=c["dit_cfg"], compute_kernel_config=c["mm_ck"],
        )
    y = ttnn.linear(h, w, bias=b)
    return ttnn.add(residual, y)


def _sdpa(q, k, v, device, cfg: Optional[FusedConfig]):
    """FlashAttention-2 kernel. ``[B, H, S, 64]`` TILE bf16 interleaved, no mask (the kernel
    handles padding itself; an explicit mask was numerically wrong on Blackhole). Legacy:
    ttnn's default 32/32 chunks. ``MAST3R_SDPA_CHUNKS=q,k``: explicit program config
    (fewer online-softmax rescale steps; rf-detr measured 0.30 -> 0.13 ms per layer AND
    higher accuracy with larger chunks)."""
    if cfg is not None and cfg.sdpa_chunks is not None:
        c = _fused_consts(device, cfg)
        return ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, program_config=c["sdpa_pc"])
    return ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)


_PATCH_EMBED_CACHE: dict = {}


def patch_embed(img: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device):
    """Host-returning patch_embed (used by per-layer tests).

    Wraps the on-device version and downloads to host.
    """
    tt_out = patch_embed_device(img, weight, bias, device)
    out_torch = ttnn.to_torch(tt_out).reshape(1, -1, weight.shape[0])
    return out_torch


def patch_embed_device(img: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device, tt_img=None):
    """Patch embedding fully on device: upload img + ttnn im2col (reshape+permute)
    + ttnn.linear matmul. img: (B, 3, H, W) host → ttnn (B, N, 1024) tile.

    ``tt_img`` (fused path): the image already on device as a bf16 ROW_MAJOR ``(B, 3, H, W)``
    tensor -- the trace's persistent input buffer -- so no host->device transfer happens
    inside the graph; ``img`` is then ignored.
    """
    if tt_img is None:
        B, C, H, W = img.shape
    else:
        B, C, H, W = (int(s) for s in tt_img.shape)
    p = weight.shape[2]
    hp, wp = H // p, W // p
    N = hp * wp

    cache_key = (id(weight), id(device))
    cached = _PATCH_EMBED_CACHE.get(cache_key)
    if cached is None:
        w_flat = weight.reshape(weight.shape[0], -1).t().contiguous()  # (C*p*p, E)
        tt_w = ttnn.from_torch(w_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(bias.reshape(1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _PATCH_EMBED_CACHE[cache_key] = (tt_w, tt_b)
    else:
        tt_w, tt_b = cached

    # Upload img directly; do im2col entirely on device via reshape + permute.
    if tt_img is None:
        img_bf16 = img.to(torch.bfloat16) if img.dtype != torch.bfloat16 else img
        tt_img = ttnn.from_torch(img_bf16, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # (B, C, H, W) → (B, C, hp, p, wp, p) → permute → (B, hp, wp, C, p, p) → (B, N, C*p*p)
    tt_img = ttnn.reshape(tt_img, (B, C, hp, p, wp, p))
    tt_img = ttnn.permute(tt_img, (0, 2, 4, 1, 3, 5))
    tt_patches = ttnn.reshape(tt_img, (B, N, C * p * p))
    tt_patches = ttnn.to_layout(tt_patches, ttnn.TILE_LAYOUT)
    tt_out = ttnn.linear(tt_patches, tt_w, bias=tt_b)
    return tt_out


# ---------- RoPE (host-side, identical to reference) ----------

_ROPE_CACHE: dict = {}
_POS_MAX_CACHE: dict = {}
_ROPE_LUT_CACHE: dict = {}
_DEVICE_ROPE_CACHE: dict = {}


def _rope_cos_sin(D: int, max_pos: int, base: float, dtype):
    key = (D, max_pos, base, dtype)
    c = _ROPE_CACHE.get(key)
    if c is None:
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        seq = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", seq, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype)
        c = (emb.cos(), emb.sin())
        _ROPE_CACHE[key] = c
    return c


def _pos_max_plus_one(pos: torch.Tensor) -> int:
    # pos is reused across all encoder/decoder blocks every inference; the
    # `.item()` host sync was repeating ~96 times per call. Memoise on object
    # id (positions are constructed once per dust3r_forward).
    key = id(pos)
    v = _POS_MAX_CACHE.get(key)
    if v is None:
        v = int(pos.max().item()) + 1
        _POS_MAX_CACHE[key] = v
    return v


def _rope_lut_host(pos: torch.Tensor, D: int, base: float, dtype):
    """Pre-indexed combined cos/sin LUTs of shape (B, 1, N, 2D).

    First D channels are vertical-pos cos/sin, last D are horizontal-pos.
    Indexing cos[pos[..., 0]] etc. happens once per inference instead of
    every RoPE call (~96× per inference for encoder + decoder).
    """
    key = (id(pos), D, base, dtype)
    lut = _ROPE_LUT_CACHE.get(key)
    if lut is not None:
        return lut
    max_pos = _pos_max_plus_one(pos)
    cos, sin = _rope_cos_sin(D, max_pos, base, dtype)
    cy = cos[pos[..., 0]][:, None, :, :]
    sy = sin[pos[..., 0]][:, None, :, :]
    cx = cos[pos[..., 1]][:, None, :, :]
    sx = sin[pos[..., 1]][:, None, :, :]
    cos_full = torch.cat((cy, cx), dim=-1).contiguous()  # (B, 1, N, 2D)
    sin_full = torch.cat((sy, sx), dim=-1).contiguous()
    lut = (cos_full, sin_full)
    _ROPE_LUT_CACHE[key] = lut
    return lut


def _rope_apply(tokens: torch.Tensor, pos: torch.Tensor, base: float = 100.0) -> torch.Tensor:
    """Apply DUSt3R 2D RoPE100 on host. tokens: (B, H, N, Dh), pos: (B, N, 2).

    Single multiply-add over the full Dh, with rotate-half applied independently
    to the y-half (first Dh//2) and x-half (last Dh//2) of each head.
    """
    Dh = tokens.shape[-1]
    D = Dh // 2
    cos_full, sin_full = _rope_lut_host(pos, D, base, tokens.dtype)

    # rotate_half_split: split into 4 quarters [a, b, c, d] (each Dh//4) → [-b, a, -d, c].
    Q = D // 2
    a = tokens[..., :Q]
    b = tokens[..., Q:2 * Q]
    c = tokens[..., 2 * Q:3 * Q]
    d = tokens[..., 3 * Q:]
    rotated = torch.cat((-b, a, -d, c), dim=-1)
    return tokens * cos_full + rotated * sin_full


def _rope_lut_device(pos: torch.Tensor, dh: int, device, base: float = 100.0):
    """cos_full / sin_full of shape (B, 1, N, dh) uploaded to device — cached
    per (pos object id, dh) so we pay one upload per inference.
    """
    key = (id(pos), dh, base, id(device))
    cached = _DEVICE_ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    cos_full, sin_full = _rope_lut_host(pos, dh // 2, base, torch.bfloat16)
    tt_cos = ttnn.from_torch(cos_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_sin = ttnn.from_torch(sin_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _DEVICE_ROPE_CACHE[key] = (tt_cos, tt_sin)
    return tt_cos, tt_sin


def _rope_device(tt_t, tt_cos, tt_sin, B: int, H: int, N: int, dh: int):
    """Apply DUSt3R 2D RoPE on device. tt_t: (B, H, N, dh) — split y/x halves,
    rotate each half independently via the [-b, a, -d, c] pattern.
    """
    Q = dh // 4
    a = ttnn.slice(tt_t, [0, 0, 0, 0],         [B, H, N, Q])
    b = ttnn.slice(tt_t, [0, 0, 0, Q],         [B, H, N, 2 * Q])
    c = ttnn.slice(tt_t, [0, 0, 0, 2 * Q],     [B, H, N, 3 * Q])
    d = ttnn.slice(tt_t, [0, 0, 0, 3 * Q],     [B, H, N, 4 * Q])
    rotated = ttnn.concat([ttnn.neg(b), a, ttnn.neg(d), c], dim=-1)
    return ttnn.add(ttnn.mul(tt_t, tt_cos), ttnn.mul(rotated, tt_sin))


def _rope_lut_device_fused(pos: torch.Tensor, dh: int, device, cfg: FusedConfig, base: float = 100.0):
    """cos/sin tables for the single-kernel RoPE ops, ``[1, 1, N, dh]`` bf16 TILE (batch and
    heads are broadcast by both kernels; ``shape[0] == shape[1] == 1`` is what they
    validate), channels in the permuted order of ``cfg.rope`` -- see ``fused.py``.
    ``MAST3R_ROPE_LUT=fp32`` (default) computes cos/sin of fp32 angles like the reference;
    ``bf16`` reproduces the legacy port's angle-rounded tables. Cached per
    ``(id(pos), dh, device, mode)`` like the legacy LUT; uploaded once, never inside a trace.
    """
    key = ("fused", id(pos), dh, base, id(device), cfg.rope, cfg.rope_lut)
    cached = _DEVICE_ROPE_CACHE.get(key)
    if cached is not None:
        return cached
    angle_dtype = torch.float32 if cfg.rope_lut == "fp32" else torch.bfloat16
    cos, sin = rope_lut_permuted(pos[0], dh, cfg.rope, base, angle_dtype, torch.bfloat16)
    tt_cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _DEVICE_ROPE_CACHE[key] = (tt_cos, tt_sin)
    return tt_cos, tt_sin


def _apply_rope(tt_t, pos: torch.Tensor, B: int, H: int, N: int, dh: int, device, cfg: Optional[FusedConfig]):
    """DUSt3R 2-D RoPE on a device tensor ``[B, H, N, dh]`` (TILE bf16 interleaved).

    Legacy / ``MAST3R_ROPE=slices``: the 10-call slice/neg/concat/mul/add chain (~19 device
    programs: the 16-column slices and the 4-way concat fall back to row-major).
    ``llama``: ``rotary_embedding_llama(x, cos, sin, trans_mat, is_decode_mode=False)`` --
    one program; requires the q/k channels in ``P_il`` order (folded into the weights).
    ``legacy``: ``rotary_embedding(x, cos, sin)`` -- one program; ``P_qs`` order; needs
    ``dh % 64 == 0`` (64 ok) and ``cos.shape[-2] >= N``.
    """
    if cfg is not None and cfg.rope == "llama":
        tt_cos, tt_sin = _rope_lut_device_fused(pos, dh, device, cfg)
        c = _fused_consts(device, cfg)
        return ttnn.experimental.rotary_embedding_llama(
            tt_t, tt_cos, tt_sin, c["trans_mat"], is_decode_mode=False, compute_kernel_config=c["rope_ck"]
        )
    if cfg is not None and cfg.rope == "legacy":
        tt_cos, tt_sin = _rope_lut_device_fused(pos, dh, device, cfg)
        return ttnn.experimental.rotary_embedding(tt_t, tt_cos, tt_sin)
    tt_cos, tt_sin = _rope_lut_device(pos, dh, device)
    return _rope_device(tt_t, tt_cos, tt_sin, B, H, N, dh)


# ---------- Encoder block (device-resident) ----------

def _preload_enc_block_weights(state: dict, i: int, device, cfg: Optional[FusedConfig] = None) -> dict:
    """Upload weights for one encoder block.

    Linear weights use BFLOAT16 (was BFLOAT8_B) — bf8's tile-scaled quantisation
    drops head1 pointmap PCC to 0.98 on real CO3D images; bf16 weights keep
    PCC > 0.99 at ~2× weight DRAM traffic but the matmul is already compute-
    bound, not bandwidth-bound, so wall clock is barely affected.

    Fused RoPE (``cfg.fused_rope``): the q and k output rows of ``attn.qkv`` (weight and
    bias) are permuted per head by the kernel's channel permutation before upload -- a
    pure row gather (bit-exact, ``torch.equal`` in the host tests); v rows untouched.
    """
    def mat8(t):
        return ttnn.from_torch(t.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    qkv_w = state[f"enc_blocks.{i}.attn.qkv.weight"]
    qkv_b = state[f"enc_blocks.{i}.attn.qkv.bias"]
    if cfg is not None and cfg.fused_rope:
        heads = 16
        qkv_w, qkv_b = permute_qkv_rows(qkv_w, qkv_b, heads, cfg.rope)
    tt = {}
    tt["g1"] = _t2d(state[f"enc_blocks.{i}.norm1.weight"].reshape(1, 1, -1), device)
    tt["b1"] = _t2d(state[f"enc_blocks.{i}.norm1.bias"].reshape(1, 1, -1), device)
    tt["g2"] = _t2d(state[f"enc_blocks.{i}.norm2.weight"].reshape(1, 1, -1), device)
    tt["b2"] = _t2d(state[f"enc_blocks.{i}.norm2.bias"].reshape(1, 1, -1), device)
    tt["qkv_w"] = mat8(qkv_w)
    tt["qkv_b"] = _t2d(qkv_b.reshape(1, 1, -1), device)
    tt["pw"] = mat8(state[f"enc_blocks.{i}.attn.proj.weight"])
    tt["pb"] = _t2d(state[f"enc_blocks.{i}.attn.proj.bias"].reshape(1, 1, -1), device)
    tt["w1"] = mat8(state[f"enc_blocks.{i}.mlp.fc1.weight"])
    tt["b1f"] = _t2d(state[f"enc_blocks.{i}.mlp.fc1.bias"].reshape(1, 1, -1), device)
    tt["w2"] = mat8(state[f"enc_blocks.{i}.mlp.fc2.weight"])
    tt["b2f"] = _t2d(state[f"enc_blocks.{i}.mlp.fc2.bias"].reshape(1, 1, -1), device)
    return tt


def encoder_block_device_pre(
    tt_x,
    pos: torch.Tensor,
    tt_w: dict,
    device,
    heads: int = 16,
    cfg: Optional[FusedConfig] = None,
):
    """Encoder block with pre-uploaded device weights (no per-call upload).

    ``cfg=None`` (or a disabled config) is the legacy 31-call block. With ``TT_FUSED=1``
    the RoPE chains become one op each (weights must have been preloaded with the same
    ``cfg``), and the matmul / SDPA sub-knobs select the fused kernels -- see ``_apply_rope``,
    ``_mm``, ``_mm_residual``, ``_sdpa``.
    """
    shape = tt_x.shape
    B, N, D = int(shape[0]), int(shape[1]), int(shape[2])
    dh = D // heads

    tt_n1 = ttnn.layer_norm(tt_x, weight=tt_w["g1"], bias=tt_w["b1"], epsilon=1e-6)
    tt_qkv = _mm(tt_n1, tt_w["qkv_w"], tt_w["qkv_b"], device, cfg)

    # Single fused op replaces 3 slices + reshape + permute + concat — v stays on device.
    tt_q, tt_k, tt_v = ttnn.transformer.split_query_key_value_and_split_heads(
        tt_qkv, num_heads=heads, transpose_key=False
    )
    # On-device 2D RoPE — eliminates the host roundtrip for q/k.
    tt_q = _apply_rope(tt_q, pos, B, heads, N, dh, device, cfg)
    tt_k = _apply_rope(tt_k, pos, B, heads, N, dh, device, cfg)
    tt_ctx = _sdpa(tt_q, tt_k, tt_v, device, cfg)
    tt_ctx = ttnn.transformer.concatenate_heads(tt_ctx)  # (B, N, D)
    tt_x = _mm_residual(tt_ctx, tt_w["pw"], tt_w["pb"], tt_x, device, cfg)

    tt_n2 = ttnn.layer_norm(tt_x, weight=tt_w["g2"], bias=tt_w["b2"], epsilon=1e-6)
    tt_h = _mm(tt_n2, tt_w["w1"], tt_w["b1f"], device, cfg, activation="gelu")
    tt_x = _mm_residual(tt_h, tt_w["w2"], tt_w["b2f"], tt_x, device, cfg)
    return tt_x


def encoder_block_device(
    tt_x,  # ttnn tensor on device, (B, N, 1024)
    pos: torch.Tensor,  # (B, N, 2) host
    weights: dict,
    device,
    heads: int = 16,
):
    """Encoder block that keeps x on device end-to-end.

    Only the attention q/k get briefly round-tripped to host for RoPE
    (ttnn.experimental.rotary_embedding doesn't directly match DUSt3R's 2D
     RoPE100 — TODO implement device RoPE).
    """
    # Read shape from x before work (tt_x shape == torch shape).
    shape = tt_x.shape
    B, N, D = int(shape[0]), int(shape[1]), int(shape[2])
    dh = D // heads

    # --- norm1 + qkv on device, keep residual ---
    g1 = _t2d(weights["norm1.weight"].reshape(1, 1, -1), device)
    b1 = _t2d(weights["norm1.bias"].reshape(1, 1, -1), device)
    tt_n1 = ttnn.layer_norm(tt_x, weight=g1, bias=b1, epsilon=1e-6)

    qkv_w = _t2d(weights["attn.qkv.weight"].t().contiguous(), device)
    qkv_b = _t2d(weights["attn.qkv.bias"].reshape(1, 1, -1), device)
    tt_qkv = ttnn.linear(tt_n1, qkv_w, bias=qkv_b)

    # Round-trip qkv once for RoPE + attention (keeps attn on device for the bmm).
    qkv_host = ttnn.to_torch(tt_qkv).reshape(B, N, 3, heads, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv_host[0], qkv_host[1], qkv_host[2]
    q = _rope_apply(q, pos)
    k = _rope_apply(k, pos)

    tt_q = _t2d(q.reshape(B * heads, N, dh), device)
    tt_k = _t2d(k.transpose(-2, -1).reshape(B * heads, dh, N).contiguous(), device)
    tt_v = _t2d(v.reshape(B * heads, N, dh), device)
    tt_scores = ttnn.matmul(tt_q, tt_k)
    tt_scores = ttnn.multiply(tt_scores, 1.0 / math.sqrt(dh))
    tt_attn = ttnn.softmax(tt_scores, dim=-1)
    tt_ctx = ttnn.matmul(tt_attn, tt_v)  # (B*H, N, dh)

    # Permute heads/tokens back to (B, N, D) on host then re-upload for proj.
    ctx_host = ttnn.to_torch(tt_ctx).reshape(B, heads, N, dh).transpose(1, 2).reshape(B, N, D)
    tt_ctx2 = _t2d(ctx_host, device)
    pw = _t2d(weights["attn.proj.weight"].t().contiguous(), device)
    pb = _t2d(weights["attn.proj.bias"].reshape(1, 1, -1), device)
    tt_proj = ttnn.linear(tt_ctx2, pw, bias=pb)

    tt_x = ttnn.add(tt_x, tt_proj)

    # --- norm2 + mlp fully on device ---
    g2 = _t2d(weights["norm2.weight"].reshape(1, 1, -1), device)
    b2 = _t2d(weights["norm2.bias"].reshape(1, 1, -1), device)
    tt_n2 = ttnn.layer_norm(tt_x, weight=g2, bias=b2, epsilon=1e-6)

    w1 = _t2d(weights["mlp.fc1.weight"].t().contiguous(), device)
    b1f = _t2d(weights["mlp.fc1.bias"].reshape(1, 1, -1), device)
    tt_h = ttnn.linear(tt_n2, w1, bias=b1f)
    tt_h = ttnn.gelu(tt_h)

    w2 = _t2d(weights["mlp.fc2.weight"].t().contiguous(), device)
    b2f = _t2d(weights["mlp.fc2.bias"].reshape(1, 1, -1), device)
    tt_m = ttnn.linear(tt_h, w2, bias=b2f)

    tt_x = ttnn.add(tt_x, tt_m)
    return tt_x


def encoder_block(
    x: torch.Tensor,
    pos: torch.Tensor,
    weights: dict,
    device,
    heads: int = 16,
) -> torch.Tensor:
    """Single ViT-L encoder block on TT.

    x: (B, N, 1024) torch      pos: (B, N, 2) torch
    weights: dict with keys norm1/2.{weight,bias}, attn.qkv/proj.{weight,bias},
             mlp.fc1/fc2.{weight,bias}
    returns (B, N, 1024) torch
    """
    B, N, D = x.shape
    dh = D // heads

    tt_x = _t2d(x, device)

    # --- norm1 + qkv ---
    g1 = _t2d(weights["norm1.weight"].reshape(1, 1, -1), device)
    b1 = _t2d(weights["norm1.bias"].reshape(1, 1, -1), device)
    tt_n1 = ttnn.layer_norm(tt_x, weight=g1, bias=b1, epsilon=1e-6)

    qkv_w = _t2d(weights["attn.qkv.weight"].t().contiguous(), device)
    qkv_b = _t2d(weights["attn.qkv.bias"].reshape(1, 1, -1), device)
    tt_qkv = ttnn.matmul(tt_n1, qkv_w)
    tt_qkv = ttnn.add(tt_qkv, qkv_b)

    # Pull qkv to host to apply RoPE and attention (simple + correct).
    qkv_host = ttnn.to_torch(tt_qkv)  # (B, N, 3D)
    qkv_host = qkv_host.reshape(B, N, 3, heads, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv_host[0], qkv_host[1], qkv_host[2]  # (B, H, N, dh)
    q = _rope_apply(q, pos)
    k = _rope_apply(k, pos)

    # Attention on device via bmm+softmax.
    tt_q = _t2d(q.reshape(B * heads, N, dh), device)
    tt_k = _t2d(k.transpose(-2, -1).reshape(B * heads, dh, N).contiguous(), device)
    tt_v = _t2d(v.reshape(B * heads, N, dh), device)
    tt_scores = ttnn.matmul(tt_q, tt_k)
    tt_scores = ttnn.multiply(tt_scores, 1.0 / math.sqrt(dh))
    tt_attn = ttnn.softmax(tt_scores, dim=-1)
    tt_ctx = ttnn.matmul(tt_attn, tt_v)  # (B*H, N, dh)
    ctx_host = ttnn.to_torch(tt_ctx).reshape(B, heads, N, dh).transpose(1, 2).reshape(B, N, D)

    tt_ctx2 = _t2d(ctx_host, device)
    pw = _t2d(weights["attn.proj.weight"].t().contiguous(), device)
    pb = _t2d(weights["attn.proj.bias"].reshape(1, 1, -1), device)
    tt_proj = ttnn.matmul(tt_ctx2, pw)
    tt_proj = ttnn.add(tt_proj, pb)

    tt_x = ttnn.add(tt_x, tt_proj)

    # --- norm2 + mlp ---
    g2 = _t2d(weights["norm2.weight"].reshape(1, 1, -1), device)
    b2 = _t2d(weights["norm2.bias"].reshape(1, 1, -1), device)
    tt_n2 = ttnn.layer_norm(tt_x, weight=g2, bias=b2, epsilon=1e-6)

    w1 = _t2d(weights["mlp.fc1.weight"].t().contiguous(), device)
    b1f = _t2d(weights["mlp.fc1.bias"].reshape(1, 1, -1), device)
    tt_h = ttnn.matmul(tt_n2, w1)
    tt_h = ttnn.add(tt_h, b1f)
    tt_h = ttnn.gelu(tt_h)

    w2 = _t2d(weights["mlp.fc2.weight"].t().contiguous(), device)
    b2f = _t2d(weights["mlp.fc2.bias"].reshape(1, 1, -1), device)
    tt_m = ttnn.matmul(tt_h, w2)
    tt_m = ttnn.add(tt_m, b2f)

    tt_out = ttnn.add(tt_x, tt_m)
    return ttnn.to_torch(tt_out)


def full_encoder(
    img: torch.Tensor,
    state: dict,
    device,
    depth: int = 24,
    return_device: bool = False,
    cfg: Optional[FusedConfig] = None,
    tt_img=None,
):
    """Full encoder: patch_embed -> 24 encoder blocks -> enc_norm.

    img: (B, 3, H, W) torch. Returns (B, N, 1024) — host torch by default,
    device tensor when ``return_device=True`` (used by dust3r_forward to feed
    the decoder without a round-trip).

    ``cfg``: fused-path configuration; ``None`` means the process-wide ``fused_config()``
    (legacy when ``TT_FUSED`` is unset). ``tt_img``: the image already on device
    (persistent trace input, see ``patch_embed_device``).
    """
    if cfg is None:
        cfg = fused_config()
    # Patch embed on device, output stays device-resident.
    if tt_img is None:
        B, C, H, W = img.shape
    else:
        B, C, H, W = (int(s) for s in tt_img.shape)
    hp, wp = H // 16, W // 16
    tt_x = patch_embed_device(img, state["patch_embed.proj.weight"], state["patch_embed.proj.bias"], device,
                              tt_img=tt_img)
    # Positions matching the reference (row-major y then x). Cached: the RoPE LUT
    # caches below are keyed on id(pos), so a fresh tensor per call re-derived the
    # encoder LUT on host and re-uploaded it under a new key every forward
    # (unbounded cache growth in a long-running server).
    pos = _make_positions_cached(B, hp, wp)

    # Pre-upload all encoder block weights once (cached on module as a simple memo).
    # The key includes cfg: fused RoPE permutes the q/k rows at upload time.
    cache_key = (id(state), cfg)
    if not hasattr(full_encoder, "_cache") or full_encoder._cache_key != cache_key:
        full_encoder._cache = [_preload_enc_block_weights(state, i, device, cfg) for i in range(depth)]
        full_encoder._enc_norm_g = _t2d(state["enc_norm.weight"].reshape(1, 1, -1), device)
        full_encoder._enc_norm_b = _t2d(state["enc_norm.bias"].reshape(1, 1, -1), device)
        full_encoder._cache_key = cache_key

    for i in range(depth):
        tt_x = encoder_block_device_pre(tt_x, pos, full_encoder._cache[i], device, cfg=cfg)

    tt_x = ttnn.layer_norm(tt_x, weight=full_encoder._enc_norm_g, bias=full_encoder._enc_norm_b, epsilon=1e-6)
    if return_device:
        return tt_x
    return ttnn.to_torch(tt_x)


# ---------- Decoder block ----------

def _ttnn_linear(t_host: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device) -> torch.Tensor:
    """Simple device-side linear: host in/out (for use as building block)."""
    tt_in = _t2d(t_host, device)
    tt_w = _t2d(weight.t().contiguous(), device)
    tt_b = _t2d(bias.reshape(1, 1, -1), device)
    tt_out = ttnn.matmul(tt_in, tt_w)
    tt_out = ttnn.add(tt_out, tt_b)
    return ttnn.to_torch(tt_out)


def _ttnn_layer_norm(t_host: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, device) -> torch.Tensor:
    tt_in = _t2d(t_host, device)
    g = _t2d(weight.reshape(1, 1, -1), device)
    b = _t2d(bias.reshape(1, 1, -1), device)
    return ttnn.to_torch(ttnn.layer_norm(tt_in, weight=g, bias=b, epsilon=1e-6))


def _ttnn_attn_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, device) -> torch.Tensor:
    """Attention on device: (B*H, N, dh) x (B*H, dh, M) -> softmax -> @v."""
    dh = q.shape[-1]
    tt_q = _t2d(q, device)
    tt_k = _t2d(k, device)
    tt_v = _t2d(v, device)
    scores = ttnn.matmul(tt_q, tt_k)
    scores = ttnn.multiply(scores, 1.0 / math.sqrt(dh))
    attn = ttnn.softmax(scores, dim=-1)
    ctx = ttnn.matmul(attn, tt_v)
    return ttnn.to_torch(ctx)


def decoder_block(
    x: torch.Tensor,
    y: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    weights: dict,
    device,
    heads: int = 12,
) -> torch.Tensor:
    """One DUSt3R decoder block on TT.

    x, y: (B, N, 768) torch           pos_x, pos_y: (B, N, 2) torch
    weights: keys norm1/2/3/norm_y.{w,b}, attn.qkv/proj.{w,b},
             cross_attn.projq/projk/projv/proj.{w,b}, mlp.fc1/fc2.{w,b}
    """
    B, N, D = x.shape
    dh = D // heads

    # --- self-attention over x ---
    x_n = _ttnn_layer_norm(x, weights["norm1.weight"], weights["norm1.bias"], device)
    qkv = _ttnn_linear(x_n, weights["attn.qkv.weight"], weights["attn.qkv.bias"], device)
    qkv = qkv.reshape(B, N, 3, heads, dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = _rope_apply(q, pos_x)
    k = _rope_apply(k, pos_x)
    ctx = _ttnn_attn_core(
        q.reshape(B * heads, N, dh),
        k.transpose(-2, -1).reshape(B * heads, dh, N).contiguous(),
        v.reshape(B * heads, N, dh),
        device,
    )
    ctx = ctx.reshape(B, heads, N, dh).transpose(1, 2).reshape(B, N, D)
    proj = _ttnn_linear(ctx, weights["attn.proj.weight"], weights["attn.proj.bias"], device)
    x = x + proj

    # --- cross-attention: q from x, k/v from y ---
    x_n2 = _ttnn_layer_norm(x, weights["norm2.weight"], weights["norm2.bias"], device)
    y_n = _ttnn_layer_norm(y, weights["norm_y.weight"], weights["norm_y.bias"], device)
    q = _ttnn_linear(x_n2, weights["cross_attn.projq.weight"], weights["cross_attn.projq.bias"], device)
    k = _ttnn_linear(y_n, weights["cross_attn.projk.weight"], weights["cross_attn.projk.bias"], device)
    v = _ttnn_linear(y_n, weights["cross_attn.projv.weight"], weights["cross_attn.projv.bias"], device)
    q = q.reshape(B, N, heads, dh).transpose(1, 2)
    k = k.reshape(B, y.shape[1], heads, dh).transpose(1, 2)
    v = v.reshape(B, y.shape[1], heads, dh).transpose(1, 2)
    q = _rope_apply(q, pos_x)
    k = _rope_apply(k, pos_y)
    ctx = _ttnn_attn_core(
        q.reshape(B * heads, N, dh),
        k.transpose(-2, -1).reshape(B * heads, dh, y.shape[1]).contiguous(),
        v.reshape(B * heads, y.shape[1], dh),
        device,
    )
    ctx = ctx.reshape(B, heads, N, dh).transpose(1, 2).reshape(B, N, D)
    proj = _ttnn_linear(ctx, weights["cross_attn.proj.weight"], weights["cross_attn.proj.bias"], device)
    x = x + proj

    # --- MLP ---
    x_n3 = _ttnn_layer_norm(x, weights["norm3.weight"], weights["norm3.bias"], device)
    h = _ttnn_linear(x_n3, weights["mlp.fc1.weight"], weights["mlp.fc1.bias"], device)
    tt_h = _t2d(h, device)
    tt_h = ttnn.gelu(tt_h)
    h = ttnn.to_torch(tt_h)
    m = _ttnn_linear(h, weights["mlp.fc2.weight"], weights["mlp.fc2.bias"], device)
    return x + m


def _preload_dec_block_weights(state: dict, idx: int, branch: int, device,
                               cfg: Optional[FusedConfig] = None, heads: int = 12) -> dict:
    """Upload one decoder block's weights. Fused RoPE (``cfg.fused_rope``): the q/k rows
    of ``attn.qkv``, all rows of ``cross_attn.projq`` and the k half of the fused
    ``projk|projv`` weight/bias are permuted per head (exact row gather) before upload."""
    prefix = "dec_blocks." if branch == 1 else "dec_blocks2."
    def w(k):
        return state[f"{prefix}{idx}.{k}"]
    def vec(t):
        return _t2d(t.reshape(1, 1, -1), device)
    def mat(t):
        return ttnn.from_torch(t.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Fuse cross-attn K and V projections into one linear so we can use
    # split_query_key_value_and_split_heads(qc, kv_input_tensor=kvc, ...).
    ck_w_t = w("cross_attn.projk.weight")
    cv_w_t = w("cross_attn.projv.weight")
    ckv_w_torch = torch.cat([ck_w_t, cv_w_t], dim=0)            # (2*D, D) (out, in)
    ckv_b_torch = torch.cat([w("cross_attn.projk.bias"), w("cross_attn.projv.bias")], dim=0)
    qkv_w_torch, qkv_b_torch = w("attn.qkv.weight"), w("attn.qkv.bias")
    cq_w_torch, cq_b_torch = w("cross_attn.projq.weight"), w("cross_attn.projq.bias")
    if cfg is not None and cfg.fused_rope:
        qkv_w_torch, qkv_b_torch = permute_qkv_rows(qkv_w_torch, qkv_b_torch, heads, cfg.rope)
        cq_w_torch, cq_b_torch = permute_q_rows(cq_w_torch, cq_b_torch, heads, cfg.rope)
        ckv_w_torch, ckv_b_torch = permute_kv_rows(ckv_w_torch, ckv_b_torch, heads, cfg.rope)
    return {
        "g1": vec(w("norm1.weight")), "b1v": vec(w("norm1.bias")),
        "g2": vec(w("norm2.weight")), "b2v": vec(w("norm2.bias")),
        "g3": vec(w("norm3.weight")), "b3v": vec(w("norm3.bias")),
        "gy": vec(w("norm_y.weight")), "byv": vec(w("norm_y.bias")),
        "qkv_w": mat(qkv_w_torch), "qkv_b": vec(qkv_b_torch),
        "proj_w": mat(w("attn.proj.weight")), "proj_b": vec(w("attn.proj.bias")),
        "cq_w": mat(cq_w_torch), "cq_b": vec(cq_b_torch),
        "ckv_w": mat(ckv_w_torch), "ckv_b": vec(ckv_b_torch),
        "cp_w": mat(w("cross_attn.proj.weight")), "cp_b": vec(w("cross_attn.proj.bias")),
        "fc1_w": mat(w("mlp.fc1.weight")), "fc1_b": vec(w("mlp.fc1.bias")),
        "fc2_w": mat(w("mlp.fc2.weight")), "fc2_b": vec(w("mlp.fc2.bias")),
    }


def decoder_block_device_pre(
    tt_x,  # device (B, N, 768)
    tt_y,  # device (B, N, 768)
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    tt_w: dict,
    device,
    heads: int = 12,
    cfg: Optional[FusedConfig] = None,
):
    """One decoder block (self-attn, cross-attn on ``tt_y``, MLP). ``cfg=None`` = legacy
    60-call block; with ``TT_FUSED=1`` the four RoPE chains are one op each (weights
    preloaded with the same ``cfg``) and the matmul / SDPA sub-knobs apply."""
    shape = tt_x.shape
    B, N, D = int(shape[0]), int(shape[1]), int(shape[2])
    dh = D // heads

    # --- self-attention (SDPA, v stays on device) ---
    tt_n1 = ttnn.layer_norm(tt_x, weight=tt_w["g1"], bias=tt_w["b1v"], epsilon=1e-6)
    tt_qkv = _mm(tt_n1, tt_w["qkv_w"], tt_w["qkv_b"], device, cfg)
    tt_q, tt_k, tt_v = ttnn.transformer.split_query_key_value_and_split_heads(
        tt_qkv, num_heads=heads, transpose_key=False
    )
    tt_q = _apply_rope(tt_q, pos_x, B, heads, N, dh, device, cfg)
    tt_k = _apply_rope(tt_k, pos_x, B, heads, N, dh, device, cfg)
    tt_ctx = _sdpa(tt_q, tt_k, tt_v, device, cfg)
    tt_ctx = ttnn.transformer.concatenate_heads(tt_ctx)  # 1 op vs permute+reshape
    tt_x = _mm_residual(tt_ctx, tt_w["proj_w"], tt_w["proj_b"], tt_x, device, cfg)

    # --- cross-attention (SDPA) ---
    # Fused K+V linear + split_qkv reduces from 3 linears + 5 reshape/permute ops
    # to 2 linears + 1 split.
    tt_n2 = ttnn.layer_norm(tt_x, weight=tt_w["g2"], bias=tt_w["b2v"], epsilon=1e-6)
    tt_yn = ttnn.layer_norm(tt_y, weight=tt_w["gy"], bias=tt_w["byv"], epsilon=1e-6)
    tt_qc = _mm(tt_n2, tt_w["cq_w"], tt_w["cq_b"], device, cfg)
    tt_kvc = _mm(tt_yn, tt_w["ckv_w"], tt_w["ckv_b"], device, cfg)  # (B, M, 2*D)
    tt_q, tt_k, tt_v = ttnn.transformer.split_query_key_value_and_split_heads(
        tt_qc, kv_input_tensor=tt_kvc, num_heads=heads, transpose_key=False
    )
    M = int(tt_y.shape[1])
    tt_q = _apply_rope(tt_q, pos_x, B, heads, N, dh, device, cfg)
    tt_k = _apply_rope(tt_k, pos_y, B, heads, M, dh, device, cfg)
    tt_ctx = _sdpa(tt_q, tt_k, tt_v, device, cfg)
    tt_ctx = ttnn.transformer.concatenate_heads(tt_ctx)
    tt_x = _mm_residual(tt_ctx, tt_w["cp_w"], tt_w["cp_b"], tt_x, device, cfg)

    # --- MLP ---
    tt_n3 = ttnn.layer_norm(tt_x, weight=tt_w["g3"], bias=tt_w["b3v"], epsilon=1e-6)
    tt_h = _mm(tt_n3, tt_w["fc1_w"], tt_w["fc1_b"], device, cfg, activation="gelu")
    tt_x = _mm_residual(tt_h, tt_w["fc2_w"], tt_w["fc2_b"], tt_x, device, cfg)
    return tt_x


def _dec_block_weights(state: dict, idx: int, branch: int) -> dict:
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
    return {k: state[f"{prefix}{idx}.{k}"] for k in keys}


def full_decoder(
    feat1: torch.Tensor,
    feat2: torch.Tensor,
    pos: torch.Tensor,
    state: dict,
    device,
    depth: int = 12,
    compute_norm: bool = True,
    cfg: Optional[FusedConfig] = None,
):
    """Dual-branch DUSt3R decoder on TT.

    feat1, feat2: (B, N, 1024) encoder outputs.  pos: (B, N, 2).
    Returns (out1, out2) each (B, N, 768).

    When ``compute_norm=False`` (the dust3r_forward path), the final dec_norm
    + downloads are skipped — DPT only consumes tap-layer features.

    ``cfg``: fused-path configuration (``None`` = process-wide ``fused_config()``).
    """
    if cfg is None:
        cfg = fused_config()
    # Pre-upload decoder weights once (memoized on module; key includes cfg because the
    # fused RoPE permutes q/k rows at upload).
    cache_key = ("dec", id(state), cfg)
    if getattr(full_decoder, "_cache_key", None) != cache_key:
        full_decoder._w1 = [_preload_dec_block_weights(state, i, 1, device, cfg) for i in range(depth)]
        full_decoder._w2 = [_preload_dec_block_weights(state, i, 2, device, cfg) for i in range(depth)]
        full_decoder._emb_w = _t2d(state["decoder_embed.weight"].t().contiguous(), device)
        full_decoder._emb_b = _t2d(state["decoder_embed.bias"].reshape(1, 1, -1), device)
        full_decoder._dnorm_g = _t2d(state["dec_norm.weight"].reshape(1, 1, -1), device)
        full_decoder._dnorm_b = _t2d(state["dec_norm.bias"].reshape(1, 1, -1), device)
        full_decoder._cache_key = cache_key

    # feat1/feat2 may already be device tensors (from full_encoder return_device=True)
    # — accept either form so we can avoid the encoder→decoder round-trip.
    tt_in1 = feat1 if isinstance(feat1, ttnn.Tensor) else _t2d(feat1, device)
    tt_in2 = feat2 if isinstance(feat2, ttnn.Tensor) else _t2d(feat2, device)
    tt_f1 = ttnn.linear(tt_in1, full_decoder._emb_w, bias=full_decoder._emb_b)
    tt_f2 = ttnn.linear(tt_in2, full_decoder._emb_w, bias=full_decoder._emb_b)

    # Hold tap tensors on device through the loop, batch all downloads at the
    # end so we pay one sync wave instead of three mid-loop syncs per branch.
    dev_taps1: list = []
    dev_taps2: list = []
    tap_layers = (0, 6, 11)
    for i in range(depth):
        nf1 = decoder_block_device_pre(tt_f1, tt_f2, pos, pos, full_decoder._w1[i], device, cfg=cfg)
        nf2 = decoder_block_device_pre(tt_f2, tt_f1, pos, pos, full_decoder._w2[i], device, cfg=cfg)
        tt_f1, tt_f2 = nf1, nf2
        if i in tap_layers:
            dev_taps1.append(tt_f1)
            dev_taps2.append(tt_f2)

    if compute_norm:
        tt_out1 = ttnn.layer_norm(tt_f1, weight=full_decoder._dnorm_g, bias=full_decoder._dnorm_b, epsilon=1e-6)
        tt_out2 = ttnn.layer_norm(tt_f2, weight=full_decoder._dnorm_g, bias=full_decoder._dnorm_b, epsilon=1e-6)
        out1 = ttnn.to_torch(tt_out1)
        out2 = ttnn.to_torch(tt_out2)
    else:
        out1 = out2 = None
    # Return tap tensors as device tensors (caller may download or feed to dpt_head_device).
    return out1, out2, dev_taps1, dev_taps2


_POS_TENSOR_CACHE: dict = {}


def _make_positions_cached(B: int, hp: int, wp: int) -> torch.Tensor:
    """Cache the positions tensor across inferences so the device RoPE LUT
    cache (keyed on id(pos)) survives — eliminates per-inference host work for
    the cos/sin pos-indexing.
    """
    key = (B, hp, wp)
    pos = _POS_TENSOR_CACHE.get(key)
    if pos is None:
        ys = torch.arange(hp)
        xs = torch.arange(wp)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack((gy, gx), dim=-1).reshape(hp * wp, 2).unsqueeze(0).expand(B, -1, -1).contiguous()
        _POS_TENSOR_CACHE[key] = pos
    return pos


def dust3r_forward(img1: torch.Tensor, img2: torch.Tensor, state: dict, device):
    """Full end-to-end DUSt3R forward on TT — DPT runs on device too.

    Encoder is run once at B=2 (both views stacked) — the dispatch/queue
    setup cost halves vs two sequential forwards; compute stays the same.

    ``TT_FUSED=1`` (read once, ``fused_config()``): dispatches to the process-wide
    :class:`TtDust3r` wrapper (``get_model``) -- same signature, same ``(1, 4, H, W)``
    float32 outputs, traced device graph. Otherwise the legacy eager graph below runs
    unchanged.
    """
    if fused_config().enabled:
        return get_model(state, device)(img1, img2)
    B, C, H, W = img1.shape
    hp, wp = H // 16, W // 16
    img_batched = torch.cat([img1, img2], dim=0)   # (2, 3, H, W)
    tt_enc_batched = full_encoder(img_batched, state, device, return_device=True)
    # Split B=2 encoder output → two (1, N, 1024) device tensors.
    N = hp * wp
    # ttnn.slice on the batch dim.
    tt_enc1 = ttnn.slice(tt_enc_batched, [0, 0, 0], [1, N, 1024])
    tt_enc2 = ttnn.slice(tt_enc_batched, [1, 0, 0], [2, N, 1024])
    pos = _make_positions_cached(1, hp, wp)

    _, _, dev_taps1, dev_taps2 = full_decoder(tt_enc1, tt_enc2, pos, state, device, compute_norm=False)
    feats1 = [tt_enc1, dev_taps1[0], dev_taps1[1], dev_taps1[2]]
    feats2 = [tt_enc2, dev_taps2[0], dev_taps2[1], dev_taps2[2]]

    tt_out1 = dpt_head_device(feats1, (hp, wp), state, 1, device)
    tt_out2 = dpt_head_device(feats2, (hp, wp), state, 2, device)
    # Final downloads — single sync wave.
    Hh, Wh = H, W
    out1 = ttnn.to_torch(tt_out1).reshape(B, Hh, Wh, 4).permute(0, 3, 1, 2).contiguous().float()
    out2 = ttnn.to_torch(tt_out2).reshape(B, Hh, Wh, 4).permute(0, 3, 1, 2).contiguous().float()
    return out1, out2


# ---------- Fused path: traced whole-graph wrapper ----------

class TtDust3r:
    """DUSt3R device graph behind ``TT_FUSED=1``: persistent input buffer + one metal trace.

    Per call (``__call__(img1, img2)`` -> the same two ``(1, 4, H, W)`` float32 tensors as
    ``dust3r_forward``):

    1. host: ``cat([img1, img2])`` -> bf16 -> host ttnn ROW_MAJOR tensor ``[2, 3, H, W]``
       (the legacy cast; 3.1 MB for 512x512);
    2. ``copy_host_to_device_tensor`` into the persistent device buffer the trace reads;
    3. ``execute_trace(blocking=False)`` -- patch embed (device im2col + linear), 24
       encoder blocks at B=2, batch split, decoder embed, 12 x 2 decoder blocks, two DPT
       heads -- one submission, no host dispatch per op;
    4. two ``to_torch`` readbacks of the ``[1, 1, H*W, 4]`` head outputs (the only syncs).

    Capture (:meth:`capture`, on the first call or explicitly from the server warm-up):
    ``ttnn.to_device`` allocates the persistent input; an **eager warm run** of the same
    graph JIT-compiles every kernel, fills the program cache, uploads all weights / LUTs /
    ``ones`` / ``trans_mat`` and lets the first ``conv2d`` / ``conv_transpose2d`` calls
    store their device-prepared weights (all of that mutates host caches and must happen
    OUTSIDE the capture); then ``begin_trace_capture`` -> graph -> ``end_trace_capture``.
    The trace is shape-locked to the captured input shape. ``MAST3R_TRACE=0`` runs the
    same fused graph eagerly (A/B). ``ttnn.open_device`` needs ``trace_region_size``
    (``MAST3R_TRACE_REGION``, default 512 MiB; ~1.2k programs -- unmeasured).
    """

    def __init__(self, state: dict, device, cfg: Optional[FusedConfig] = None):
        self.cfg = fused_config() if cfg is None else cfg
        self.state = state
        self.device = device
        self._trace_id = None
        self._persistent_in = None   # device bf16 ROW_MAJOR [2, 3, H, W]
        self._outs = None            # (tt_out1, tt_out2) written by the trace
        self._in_shape = None        # torch.Size of img1 the trace was captured for

    @property
    def trace_captured(self) -> bool:
        return self._trace_id is not None

    # -------------------------------------------------------------- pieces
    @staticmethod
    def _host_input(img1: torch.Tensor, img2: torch.Tensor):
        """Host ttnn tensor ``[2, 3, H, W]`` bf16 ROW_MAJOR (not yet on device)."""
        img = torch.cat([img1, img2], dim=0)
        img = img.to(torch.bfloat16) if img.dtype != torch.bfloat16 else img
        return ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _graph(self, tt_img):
        """The whole device graph, device tensor in -> two device tensors out."""
        _, _, H, W = (int(s) for s in tt_img.shape)
        hp, wp = H // 16, W // 16
        N = hp * wp
        tt_enc = full_encoder(None, self.state, self.device, return_device=True, cfg=self.cfg, tt_img=tt_img)
        tt_enc1 = ttnn.slice(tt_enc, [0, 0, 0], [1, N, 1024])
        tt_enc2 = ttnn.slice(tt_enc, [1, 0, 0], [2, N, 1024])
        pos = _make_positions_cached(1, hp, wp)
        _, _, taps1, taps2 = full_decoder(tt_enc1, tt_enc2, pos, self.state, self.device,
                                          compute_norm=False, cfg=self.cfg)
        out1 = dpt_head_device([tt_enc1, *taps1], (hp, wp), self.state, 1, self.device, cfg=self.cfg)
        out2 = dpt_head_device([tt_enc2, *taps2], (hp, wp), self.state, 2, self.device, cfg=self.cfg)
        return out1, out2

    @staticmethod
    def _read(outs, B: int, H: int, W: int):
        return tuple(
            ttnn.to_torch(o).reshape(B, H, W, 4).permute(0, 3, 1, 2).contiguous().float() for o in outs
        )

    # --------------------------------------------------------------- trace
    def capture(self, img1: torch.Tensor, img2: torch.Tensor) -> None:
        """Allocate the persistent input, warm the graph eagerly, capture the trace."""
        if self._trace_id is not None:
            return
        device = self.device
        host_in = self._host_input(img1, img2)
        self._persistent_in = ttnn.to_device(host_in, device)
        self._in_shape = tuple(img1.shape)
        tid = None
        try:
            outs = self._graph(self._persistent_in)      # eager warm run (JIT, caches, prepared weights)
            ttnn.synchronize_device(device)
            for o in outs:
                ttnn.deallocate(o)

            tid = ttnn.begin_trace_capture(device, cq_id=0)
            outs = self._graph(self._persistent_in)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
        except Exception:
            # Leave nothing behind: a failed warm run / capture must not leak the persistent
            # input (the next __call__ re-enters capture and would allocate a second one) nor
            # keep the device in capture mode.
            if tid is not None:
                for fn in (lambda: ttnn.end_trace_capture(device, tid, cq_id=0),
                           lambda: ttnn.release_trace(device, tid)):
                    try:
                        fn()
                    except Exception:
                        pass
            _deallocate_tensors(self._persistent_in)
            self._persistent_in = self._in_shape = None
            raise
        self._outs = outs
        self._trace_id = tid

    def _run_trace(self, host_in, B: int, H: int, W: int):
        ttnn.copy_host_to_device_tensor(host_in, self._persistent_in, cq_id=0)
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
        return self._read(self._outs, B, H, W)

    def _run_eager(self, host_in, B: int, H: int, W: int):
        tt_img = ttnn.to_device(host_in, self.device)
        return self._read(self._graph(tt_img), B, H, W)

    # ---------------------------------------------------------------- call
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor):
        if img1.shape != img2.shape or img1.dim() != 4 or img1.shape[0] != 1:
            raise ValueError(f"TtDust3r expects two (1, 3, H, W) views, got {tuple(img1.shape)} / {tuple(img2.shape)}")
        B, _, H, W = img1.shape
        if not self.cfg.trace:
            return self._run_eager(self._host_input(img1, img2), B, H, W)
        if self._trace_id is None:
            self.capture(img1, img2)
        elif tuple(img1.shape) != self._in_shape:
            raise ValueError(f"trace captured for input {self._in_shape}, got {tuple(img1.shape)}")
        return self._run_trace(self._host_input(img1, img2), B, H, W)

    def release(self) -> int:
        """Release the trace and the wrapper's device buffers (before ``close_device``)."""
        n = 0
        if self._trace_id is not None:
            try:
                ttnn.release_trace(self.device, self._trace_id)
            except Exception:
                pass
            self._trace_id = None
        n += _deallocate_tensors([self._persistent_in, list(self._outs or ())])
        self._persistent_in = self._outs = self._in_shape = None
        return n


_MODEL_CACHE: dict = {}   # (id(state), id(device)) -> TtDust3r


def get_model(state: dict, device, cfg: Optional[FusedConfig] = None) -> TtDust3r:
    """The process-wide :class:`TtDust3r` for ``(state, device)`` (built on first use;
    released by ``release_device_caches``). ``trace_captured`` tells the server whether
    the warm-up already captured the trace."""
    key = (id(state), id(device))
    m = _MODEL_CACHE.get(key)
    if m is None:
        m = TtDust3r(state, device, cfg)
        _MODEL_CACHE[key] = m
    return m


_DPT_HEAD_CACHE: dict = {}


def dpt_head(feats_list, hw, state: dict, branch: int, device):
    """Host wrapper kept for per-layer pytest harness."""
    from models.demos.mast3r.reference.torch_dust3r import load_dpt_head
    key = (id(state), branch, "bf16cl", hw)
    head = _DPT_HEAD_CACHE.get(key)
    if head is None:
        head = (
            load_dpt_head(state, branch=branch)
            .to(torch.bfloat16)
            .to(memory_format=torch.channels_last)
            .eval()
        )
        _DPT_HEAD_CACHE[key] = head
    feats_bf16 = [f.to(torch.bfloat16) for f in feats_list]
    with torch.no_grad():
        return head(feats_bf16, hw).float()


# ---------- On-device DPT ----------

_DPT_DEVICE_CACHE: dict = {}


def _preload_dpt_weights(state: dict, branch: int, device) -> dict:
    """Upload all DPT weights as ttnn tensors (per branch). Cached on module.

    1×1 convs are stored as (in, out) tile tensors on device for use as matmul
    via ttnn.linear (avoids fragile conv2d shard alignment for tiny spatial dims).
    Other convs keep their (out, in, kH, kW) host tensors for ttnn.conv2d.
    """
    prefix = f"downstream_head{branch}.dpt."

    def _w(k):
        return state[prefix + k]

    def _wt(t):
        # weight: (out, in, kH, kW). ttnn.conv2d accepts torch tensor in this exact shape.
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16)

    def _bt(t):
        return ttnn.from_torch(t.reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16)

    def _w_lin(t):
        # 1x1 conv weight (out, in, 1, 1) → linear matmul weight (in, out) on device.
        out_c, in_c = t.shape[0], t.shape[1]
        w_t = t.reshape(out_c, in_c).t().contiguous()
        return ttnn.from_torch(w_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _b_lin(t):
        # bias as a (1, 1, out_c) tile tensor for ttnn.linear bias arg.
        return ttnn.from_torch(t.reshape(1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    w = {}
    # act_postprocess: 1x1 projections as linear weights, spatial up/down as conv2d.
    w["ap0_proj_w"] = _w_lin(_w("act_postprocess.0.0.weight"))
    w["ap0_proj_b"] = _b_lin(_w("act_postprocess.0.0.bias"))
    w["ap0_up_w"] = _wt(_w("act_postprocess.0.1.weight"))
    w["ap0_up_b"] = _bt(_w("act_postprocess.0.1.bias"))
    w["ap1_proj_w"] = _w_lin(_w("act_postprocess.1.0.weight"))
    w["ap1_proj_b"] = _b_lin(_w("act_postprocess.1.0.bias"))
    w["ap1_up_w"] = _wt(_w("act_postprocess.1.1.weight"))
    w["ap1_up_b"] = _bt(_w("act_postprocess.1.1.bias"))
    w["ap2_proj_w"] = _w_lin(_w("act_postprocess.2.0.weight"))
    w["ap2_proj_b"] = _b_lin(_w("act_postprocess.2.0.bias"))
    w["ap3_proj_w"] = _w_lin(_w("act_postprocess.3.0.weight"))
    w["ap3_proj_b"] = _b_lin(_w("act_postprocess.3.0.bias"))
    w["ap3_down_w"] = _wt(_w("act_postprocess.3.1.weight"))
    w["ap3_down_b"] = _bt(_w("act_postprocess.3.1.bias"))
    # layer_rn (no bias)
    for i in (1, 2, 3, 4):
        w[f"l{i}_rn_w"] = _wt(_w(f"scratch.layer{i}_rn.weight"))
    # refinenets
    for r in (1, 2, 3, 4):
        for cu in (1, 2):
            for k in (1, 2):
                w[f"r{r}_u{cu}_c{k}_w"] = _wt(_w(f"scratch.refinenet{r}.resConfUnit{cu}.conv{k}.weight"))
                w[f"r{r}_u{cu}_c{k}_b"] = _bt(_w(f"scratch.refinenet{r}.resConfUnit{cu}.conv{k}.bias"))
        w[f"r{r}_out_w"] = _w_lin(_w(f"scratch.refinenet{r}.out_conv.weight"))
        w[f"r{r}_out_b"] = _b_lin(_w(f"scratch.refinenet{r}.out_conv.bias"))
    # head
    w["head0_w"] = _wt(_w("head.0.weight"))
    w["head0_b"] = _bt(_w("head.0.bias"))
    w["head2_w"] = _wt(_w("head.2.weight"))
    w["head2_b"] = _bt(_w("head.2.bias"))
    w["head4_w"] = _w_lin(_w("head.4.weight"))
    w["head4_b"] = _b_lin(_w("head.4.bias"))
    return w


_DPT_KCFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


_RELU_CONV_CFG = None


def _relu_conv_config():
    """``Conv2dConfig(activation=relu)``: relu applied to the accumulator before the bf16
    pack instead of a separate ``ttnn.relu`` program. Equal up to the sign of zero (relu
    commutes with round-to-nearest; host test ``test_relu_commutes_with_bf16_rounding``)."""
    global _RELU_CONV_CFG
    if _RELU_CONV_CFG is None:
        _RELU_CONV_CFG = ttnn.Conv2dConfig(activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU))
    return _RELU_CONV_CFG


def _store_prepared(cache, pw, pb):
    """Fused path: keep the op's device-prepared weight (and bias) in the weight dict so the
    next call passes device tensors -- ``conv2d.cpp`` then validates them with
    ``is_valid_device_conv_weights`` instead of re-preparing + re-uploading host tensors on
    EVERY call (23 conv + 2 convT weights per head per forward), and nothing crosses
    host->device inside the trace."""
    w, wk, bk = cache
    w[wk] = pw
    if bk is not None and pb is not None:
        w[bk] = pb


def _conv2d(tt_x, w_t, b_t, in_c, out_c, B, H, W, k=3, stride=1, padding=1, device=None,
            relu=False, cache=None):
    """ttnn.conv2d wrapper with high-fidelity compute config (DPT precision matters).

    ``relu``: fuse relu into the conv (``Conv2dConfig(activation=relu)``). ``cache``:
    ``(weights dict, weight key, bias key | None)`` -> ask the op for its prepared weights
    and store them for the next call (see ``_store_prepared``). Both default off = legacy.
    """
    kwargs = {}
    if relu:
        kwargs["conv_config"] = _relu_conv_config()
    if cache is not None:
        out, (pw, pb) = ttnn.conv2d(
            input_tensor=tt_x, weight_tensor=w_t, bias_tensor=b_t,
            device=device, in_channels=in_c, out_channels=out_c,
            batch_size=B, input_height=H, input_width=W,
            kernel_size=(k, k), stride=(stride, stride), padding=(padding, padding),
            compute_config=_DPT_KCFG, return_weights_and_bias=True, **kwargs,
        )
        _store_prepared(cache, pw, pb)
        return out
    return ttnn.conv2d(
        input_tensor=tt_x, weight_tensor=w_t, bias_tensor=b_t,
        device=device, in_channels=in_c, out_channels=out_c,
        batch_size=B, input_height=H, input_width=W,
        kernel_size=(k, k), stride=(stride, stride), padding=(padding, padding),
        compute_config=_DPT_KCFG, **kwargs,
    )


def _linear_1x1(tt_x, w_lin, b_lin, B, H, W, in_c, out_c):
    """1×1 conv via ttnn.linear (matmul). Input/output in conv2d-flat (1, 1, N, C) form, TILE layout."""
    if tt_x.layout != ttnn.TILE_LAYOUT:
        tt_x = ttnn.to_layout(tt_x, ttnn.TILE_LAYOUT)
    return ttnn.linear(tt_x, w_lin, bias=b_lin)


def _conv_t2d(tt_x, w_t, b_t, in_c, out_c, B, H, W, k, stride, padding=0, device=None, cache=None):
    """ttnn.conv_transpose2d wrapper; ``cache`` as in ``_conv2d`` (the op re-transforms and
    re-uploads host weights on every call otherwise, ``conv_transpose2d.cpp``)."""
    if cache is not None:
        out, (pw, pb) = ttnn.conv_transpose2d(
            input_tensor=tt_x, weight_tensor=w_t, bias_tensor=b_t,
            device=device, in_channels=in_c, out_channels=out_c,
            batch_size=B, input_height=H, input_width=W,
            kernel_size=(k, k), stride=(stride, stride), padding=(padding, padding),
            return_weights_and_bias=True,
        )
        _store_prepared(cache, pw, pb)
        return out
    return ttnn.conv_transpose2d(
        input_tensor=tt_x, weight_tensor=w_t, bias_tensor=b_t,
        device=device, in_channels=in_c, out_channels=out_c,
        batch_size=B, input_height=H, input_width=W,
        kernel_size=(k, k), stride=(stride, stride), padding=(padding, padding),
    )


def _tokens_to_nhwc(tt_t, B, N, C, tile=False):
    """Tokens (B, N, C) (TILE) → NHWC flat (1, 1, N, C) for the 1x1 projections.

    Legacy: to_layout(ROW_MAJOR) + reshape (and ``_linear_1x1`` converts straight back to
    TILE). ``tile=True`` (fused DPT cleanup): ``ttnn.reshape`` in TILE -- a 0-cost view
    since the last two dims do not change; drops 2 layout programs per tap, bit-identical.
    """
    if tile:
        return ttnn.reshape(tt_t, (1, 1, N, C))
    if tt_t.layout != ttnn.ROW_MAJOR_LAYOUT:
        tt_t = ttnn.to_layout(tt_t, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(tt_t, (1, 1, N, C))


def _resconv(tt_x, w, prefix, ch, B, H, W, device, cfg: Optional[FusedConfig] = None):
    """ResConvUnit on device: relu + 3x3 conv + relu + 3x3 conv + add residual.

    Fused DPT cleanup: the middle relu is folded into conv1; prepared conv weights are
    cached when ``cfg.conv_weight_cache``. The leading relu is a pre-activation of the
    residual input and cannot be fused."""
    fuse = cfg is not None and cfg.dpt_fuse
    use_cache = cfg is not None and cfg.conv_weight_cache
    c1w, c1b, c2w, c2b = f"{prefix}_c1_w", f"{prefix}_c1_b", f"{prefix}_c2_w", f"{prefix}_c2_b"
    tt_relu = ttnn.relu(tt_x)
    tt_c1 = _conv2d(tt_relu, w[c1w], w[c1b], ch, ch, B, H, W, k=3, padding=1, device=device,
                    relu=fuse, cache=(w, c1w, c1b) if use_cache else None)
    if not fuse:
        tt_c1 = ttnn.relu(tt_c1)
    tt_c2 = _conv2d(tt_c1, w[c2w], w[c2b], ch, ch, B, H, W, k=3, padding=1, device=device,
                    cache=(w, c2w, c2b) if use_cache else None)
    return ttnn.add(tt_x, tt_c2)


def _flat_to_nhwc(tt_x, B, H, W, C):
    """Conv2d output (1, 1, B*H*W, C) tile → upsample input (B, H, W, C) row-major."""
    if tt_x.layout != ttnn.ROW_MAJOR_LAYOUT:
        tt_x = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(tt_x, (B, H, W, C))


def _nhwc_to_flat(tt_x, B, H, W, C):
    """Upsample output (B, H, W, C) row-major sharded → conv2d input (1, 1, B*H*W, C)
    interleaved tile. Goes through DRAM-interleaved to avoid shard tile-alignment errors.
    """
    if tt_x.is_sharded():
        tt_x = ttnn.sharded_to_interleaved(tt_x, ttnn.DRAM_MEMORY_CONFIG)
    tt_x = ttnn.reshape(tt_x, (1, 1, B * H * W, C))
    if tt_x.layout != ttnn.TILE_LAYOUT:
        tt_x = ttnn.to_layout(tt_x, ttnn.TILE_LAYOUT)
    return tt_x


def _ffb(tt_x, w, r: int, ch, B, H, W, device, skip=None, cfg: Optional[FusedConfig] = None):
    """FeatureFusionBlock: optional skip branch's resConvUnit1, then resConvUnit2,
    then bilinear upsample (×2), then 1×1 out_conv. Input/output in conv2d-flat form."""
    if skip is not None:
        tt_skip_proc = _resconv(skip, w, f"r{r}_u1", ch, B, H, W, device, cfg=cfg)
        tt_x = ttnn.add(tt_x, tt_skip_proc)
    tt_x = _resconv(tt_x, w, f"r{r}_u2", ch, B, H, W, device, cfg=cfg)
    # Upsample requires NHWC 4D shape.
    tt_x = _flat_to_nhwc(tt_x, B, H, W, ch)
    tt_x = ttnn.upsample(tt_x, scale_factor=2, mode="bilinear")
    H2, W2 = H * 2, W * 2
    tt_x = _nhwc_to_flat(tt_x, B, H2, W2, ch)
    tt_x = _linear_1x1(tt_x, w[f"r{r}_out_w"], w[f"r{r}_out_b"], B, H2, W2, ch, ch)
    return tt_x


def dpt_head_device(tt_feats_list, hw, state: dict, branch: int, device, cfg: Optional[FusedConfig] = None):
    """Full DPT on TT device. Inputs are 4 device tensors (B, N, D) tile.
    Returns a device tensor in NHWC flat shape (1, 1, B*H_img*W_img, 4).

    ``cfg`` (``None`` = process-wide ``fused_config()``): with ``TT_FUSED=1`` the four taps
    are reshaped in TILE (no RM round trip), relu is fused into the resconv conv1 and head2
    convs (``MAST3R_DPT_FUSE``), and every conv2d / conv_transpose2d keeps its
    device-prepared weights after the first call (required for the trace). Legacy: 99 calls
    per head, host conv weights re-prepared on every call.
    """
    if cfg is None:
        cfg = fused_config()
    tile = cfg.dpt_fuse
    use_cache = cfg.conv_weight_cache

    def cache(wk, bk):
        return (w, wk, bk) if use_cache else None

    H, W = hw
    B = 1
    N = H * W

    cache_key = ("dpt", id(state), branch, id(device))
    w = _DPT_DEVICE_CACHE.get(cache_key)
    if w is None:
        w = _preload_dpt_weights(state, branch, device)
        _DPT_DEVICE_CACHE[cache_key] = w

    # Reshape inputs to NHWC flat for conv2d.
    f0 = _tokens_to_nhwc(tt_feats_list[0], B, N, 1024, tile)  # enc tap
    f1 = _tokens_to_nhwc(tt_feats_list[1], B, N, 768, tile)
    f2 = _tokens_to_nhwc(tt_feats_list[2], B, N, 768, tile)
    f3 = _tokens_to_nhwc(tt_feats_list[3], B, N, 768, tile)

    # ap0: 1024 -> 96 1x1 (linear), then ConvTranspose 4x4 stride 4 (96 -> 96).
    l1 = _linear_1x1(f0, w["ap0_proj_w"], w["ap0_proj_b"], B, H, W, 1024, 96)
    l1 = _conv_t2d(l1, w["ap0_up_w"], w["ap0_up_b"], 96, 96, B, H, W, k=4, stride=4, padding=0, device=device,
                   cache=cache("ap0_up_w", "ap0_up_b"))
    H1, W1 = H * 4, W * 4

    # ap1: 768 -> 192 1x1, then ConvTranspose 2x2 stride 2 (192 -> 192).
    l2 = _linear_1x1(f1, w["ap1_proj_w"], w["ap1_proj_b"], B, H, W, 768, 192)
    l2 = _conv_t2d(l2, w["ap1_up_w"], w["ap1_up_b"], 192, 192, B, H, W, k=2, stride=2, padding=0, device=device,
                   cache=cache("ap1_up_w", "ap1_up_b"))
    H2, W2 = H * 2, W * 2

    # ap2: 768 -> 384 1x1.
    l3 = _linear_1x1(f2, w["ap2_proj_w"], w["ap2_proj_b"], B, H, W, 768, 384)

    # ap3: 768 -> 768 1x1, then 768 -> 768 3x3 stride 2 padding 1.
    l4 = _linear_1x1(f3, w["ap3_proj_w"], w["ap3_proj_b"], B, H, W, 768, 768)
    l4 = _conv2d(l4, w["ap3_down_w"], w["ap3_down_b"], 768, 768, B, H, W, k=3, stride=2, padding=1, device=device,
                 cache=cache("ap3_down_w", "ap3_down_b"))
    H4, W4 = H // 2, W // 2

    # layer_rn: scale to 256 channels, no bias.
    l1 = _conv2d(l1, w["l1_rn_w"], None, 96, 256, B, H1, W1, k=3, padding=1, device=device, cache=cache("l1_rn_w", None))
    l2 = _conv2d(l2, w["l2_rn_w"], None, 192, 256, B, H2, W2, k=3, padding=1, device=device, cache=cache("l2_rn_w", None))
    l3 = _conv2d(l3, w["l3_rn_w"], None, 384, 256, B, H, W, k=3, padding=1, device=device, cache=cache("l3_rn_w", None))
    l4 = _conv2d(l4, w["l4_rn_w"], None, 768, 256, B, H4, W4, k=3, padding=1, device=device, cache=cache("l4_rn_w", None))

    # Refinenets (top-down).
    p4 = _ffb(l4, w, 4, 256, B, H4, W4, device, skip=None, cfg=cfg)   # 16->32
    p3 = _ffb(p4, w, 3, 256, B, H, W, device, skip=l3, cfg=cfg)       # 32->64
    p2 = _ffb(p3, w, 2, 256, B, H2, W2, device, skip=l2, cfg=cfg)     # 64->128
    p1 = _ffb(p2, w, 1, 256, B, H1, W1, device, skip=l1, cfg=cfg)     # 128->256

    # head
    p1_h, p1_w = H1 * 2, W1 * 2
    x = _conv2d(p1, w["head0_w"], w["head0_b"], 256, 128, B, p1_h, p1_w, k=3, padding=1, device=device,
                cache=cache("head0_w", "head0_b"))
    x = _flat_to_nhwc(x, B, p1_h, p1_w, 128)
    x = ttnn.upsample(x, scale_factor=2, mode="bilinear")  # 256->512
    Hh, Wh = p1_h * 2, p1_w * 2
    x = _nhwc_to_flat(x, B, Hh, Wh, 128)
    x = _conv2d(x, w["head2_w"], w["head2_b"], 128, 128, B, Hh, Wh, k=3, padding=1, device=device,
                relu=tile, cache=cache("head2_w", "head2_b"))
    if not tile:
        x = ttnn.relu(x)
    x = _linear_1x1(x, w["head4_w"], w["head4_b"], B, Hh, Wh, 128, 4)
    return x  # NHWC flat (1, 1, B*Hh*Wh, 4)


# ---------- teardown ----------

def _deallocate_tensors(obj) -> int:
    """Best-effort ttnn.deallocate of every device-resident tensor reachable from
    ``obj`` (dict / list / tuple / ttnn.Tensor). Host tensors are left to the GC."""
    n = 0
    if isinstance(obj, ttnn.Tensor):
        try:
            if ttnn.is_tensor_storage_on_device(obj):
                ttnn.deallocate(obj)
                n += 1
        except Exception:
            pass
    elif isinstance(obj, dict):
        for v in obj.values():
            n += _deallocate_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            n += _deallocate_tensors(v)
    return n


def release_device_caches() -> int:
    """Drop every module-level reference to device-resident ttnn tensors.

    The id()-keyed caches above (patch_embed weights, RoPE LUTs, encoder / decoder /
    DPT weights) otherwise keep device buffers alive past ``ttnn.close_device``, and
    ttnn then frees them against a closed device at interpreter exit -- typically an
    error or a crash during shutdown. Call this BEFORE ``ttnn.close_device``.

    Returns the number of device tensors explicitly deallocated. Every cache is left
    empty, so a later forward on a (re)opened device re-uploads from ``state``.

    Fused path: the :class:`TtDust3r` wrappers are released FIRST (``release_trace``, then
    the persistent input / output buffers), then the fused constants (``trans_mat``,
    ``ones``) and the permuted-weight caches.
    """
    n = 0
    for model in list(_MODEL_CACHE.values()):
        n += model.release()
    _MODEL_CACHE.clear()
    for cache in (_PATCH_EMBED_CACHE, _DEVICE_ROPE_CACHE, _DPT_DEVICE_CACHE, _FUSED_DEVICE_CACHE):
        n += _deallocate_tensors(cache)
        cache.clear()
    for cache in (_ROPE_CACHE, _POS_MAX_CACHE, _ROPE_LUT_CACHE, _POS_TENSOR_CACHE, _DPT_HEAD_CACHE):
        cache.clear()
    for fn, attrs in (
        (full_encoder, ("_cache", "_enc_norm_g", "_enc_norm_b", "_cache_key")),
        (full_decoder, ("_w1", "_w2", "_emb_w", "_emb_b", "_dnorm_g", "_dnorm_b", "_cache_key")),
    ):
        for a in attrs:
            if hasattr(fn, a):
                n += _deallocate_tensors(getattr(fn, a))
                delattr(fn, a)
    return n
