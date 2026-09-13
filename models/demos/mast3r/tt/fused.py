"""Host-side (torch-only) pieces of the ``TT_FUSED=1`` path of the DUSt3R port.

This module deliberately does NOT import ttnn: the host tests
(``models/tests/test_fused_host.py``) import it on a machine without a device, and
``models/server/app.py`` may import it at module import time (build-time ``verify``
runs with no device). Everything that touches ttnn lives in ``ttnn_dust3r.py``.

Contents
--------
* :class:`FusedConfig` -- the ``TT_FUSED`` knob and its sub-knobs, parsed ONCE from the
  environment at model-build time (``ttnn_dust3r.fused_config()``). Since the device
  validation of 2026-09-13 (``DEVICE_VALIDATION.md`` "Results") the fused path is the
  DEFAULT: ``TT_FUSED`` unset or ``1`` => fused; ``TT_FUSED=0`` => every field holds its
  legacy value and the device code takes exactly the legacy branches (bit-for-bit the
  pre-2026-09-13 behaviour, the graph the CO3Dv2 numbers of the card were measured with).
* 2-D RoPE reformulation for the tree's single-kernel rotary ops. DUSt3R's RoPE2D rotates
  the y-half (channels ``0..dh/2``) and the x-half (``dh/2..dh``) of every head
  independently, each with ``rotate_half`` = ``[a, b] -> [-b, a]`` on 16-wide quarters.
  Two tree kernels compute ``x * cos + rot(x) * sin`` in one program with a *different*
  fixed ``rot``; each equals DUSt3R's rotation after a fixed per-head channel permutation
  ``P`` applied identically to q and k (``q . k^T`` is invariant under a shared permutation
  of the channel axis). ``P`` is folded into the rows of the q/k projection weights and
  biases (a pure row gather: bit-exact) and into the cos/sin lookup tables (a gather):

  - ``llama`` -> ``ttnn.experimental.rotary_embedding_llama(x, cos, sin, trans_mat,
    is_decode_mode=False)``: ``rot(x) = x @ T`` per 32-wide tile with
    ``T[2i, 2i+1] = 1, T[2i+1, 2i] = -1``, i.e. ``rot[2i] = -x[2i+1], rot[2i+1] = x[2i]``
    (tree ground truth ``apply_rotary_emb_qk_real``). Permutation ``P_il``: interleave the
    two 16-wide quarters of each 32-wide half: ``[a0, b0, a1, b1, ...]``.
  - ``legacy`` -> ``ttnn.experimental.rotary_embedding(x, cos, sin)``: ``rot(x) =
    [-x[dh/2:], x[:dh/2]]`` (half swap over the full head). Permutation ``P_qs``: quarter
    swap ``[a, c, b, d]``.

  Proof of both identities (fp64) and of the weight fold (``torch.equal``) is in the
  host tests.
* Small torch helpers the host tests use to pin down other exact claims (the on-device
  im2col, the ones-vector residual formulation, relu/rounding commutation).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

ROPE_MODES = ("slices", "llama", "legacy")
ROPE_LUT_MODES = ("fp32", "bf16")
MM_MODES = ("linear", "dit", "minimal")
DEFAULT_MM_MODE = "dit"                        # device-validated 2026-09-13 (7-pair real-image A/B)
DEFAULT_SDPA_CHUNKS = (128, 256)               # idem; "default" / "ttnn" / "0" selects ttnn's 32/32
SDPA_DEFAULT_WORDS = ("default", "ttnn", "0", "none", "off")
DEFAULT_TRACE_REGION_SIZE = 512 * 1024 * 1024  # bytes; measured: the 944-op trace fits (rf-detr used 90 MB for ~260)

_TRUE = ("1", "true", "yes", "on")


def _truthy(v: Optional[str], default: bool) -> bool:
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in _TRUE


@dataclass(frozen=True)
class FusedConfig:
    """All ``TT_FUSED`` behaviour switches. The default *instance* (``FusedConfig()``) is the
    legacy path; the default *environment* (nothing set) selects the fused path since the
    2026-09-13 device validation -- see :meth:`from_env`.

    Environment (read once, see :meth:`from_env`):

    ``TT_FUSED``            master knob, default ``1`` (unset = fused). ``0`` / ``false`` = the
                            legacy eager graph; everything below is then ignored.
    ``MAST3R_ROPE``         ``llama`` (default) | ``legacy`` | ``slices``. Which RoPE kernel:
                            the llama rotary op (1 program), the legacy rotary op (1
                            program), or the port's original 10-call slice/neg/concat chain
                            (A/B reference inside the fused graph).
    ``MAST3R_ROPE_LUT``     ``fp32`` (default) | ``bf16``. Angle precision of the cos/sin
                            tables. The reference computes cos/sin of fp32 angles; the legacy
                            port rounds the ANGLE to bf16 first (``_rope_cos_sin``), an error
                            of up to ~0.06 rad in the highest-frequency channels. ``bf16``
                            reproduces the legacy tables for A/B attribution.
    ``MAST3R_FUSED_MM``     ``dit`` (default) | ``linear`` | ``minimal``. ``dit`` fuses the 5
                            residual linears per block (proj/fc2, cproj) into
                            ``dit_minimal_matmul_addcmul_fused`` (device-validated: -6.7 ms,
                            real-image xyz PCC >= the ``linear`` variant); ``linear`` = plain
                            ``ttnn.linear`` + ``add``; ``minimal`` additionally runs
                            qkv/fc1/cq/ckv through ``minimal_matmul`` (measured slower and
                            less accurate than ``dit`` on this p150a -- kept as a knob only).
    ``MAST3R_SDPA_CHUNKS``  ``"q,k"``; default ``128,256`` (device-validated: -14 ms, accuracy
                            unchanged). ``default`` / ``ttnn`` / ``0`` keeps ttnn's 32/32
                            program config (the legacy choice).
    ``MAST3R_TRACE``        ``1`` (default) | ``0``. Capture the whole device graph into one
                            metal trace (persistent input buffer, replay per call).
    ``MAST3R_TRACE_REGION`` trace region bytes for ``ttnn.open_device`` (default 512 MiB).
    ``MAST3R_DPT_FUSE``     ``1`` (default) | ``0``. Exact DPT cleanups: TILE reshape of the
                            four taps, relu fused into the resconv conv1 / head2 convs.

    Prepared conv2d / conv_transpose2d weights are cached on device whenever ``TT_FUSED=1``
    (no sub-knob: the trace needs it, and it is bit-identical).
    """

    enabled: bool = False
    rope: str = "slices"
    rope_lut: str = "bf16"
    mm: str = "linear"
    sdpa_chunks: Optional[Tuple[int, int]] = None
    trace: bool = False
    trace_region_size: int = DEFAULT_TRACE_REGION_SIZE
    dpt_fuse: bool = False
    conv_weight_cache: bool = False

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "FusedConfig":
        env = os.environ if env is None else env
        if not _truthy(env.get("TT_FUSED"), True):
            return cls()
        rope = (env.get("MAST3R_ROPE") or "llama").strip().lower()
        if rope not in ROPE_MODES:
            raise ValueError(f"MAST3R_ROPE must be one of {ROPE_MODES}, got {rope!r}")
        rope_lut = (env.get("MAST3R_ROPE_LUT") or "fp32").strip().lower()
        if rope_lut not in ROPE_LUT_MODES:
            raise ValueError(f"MAST3R_ROPE_LUT must be one of {ROPE_LUT_MODES}, got {rope_lut!r}")
        mm = (env.get("MAST3R_FUSED_MM") or DEFAULT_MM_MODE).strip().lower()
        if mm not in MM_MODES:
            raise ValueError(f"MAST3R_FUSED_MM must be one of {MM_MODES}, got {mm!r}")
        sdpa_chunks = parse_sdpa_chunks(env.get("MAST3R_SDPA_CHUNKS"), default=DEFAULT_SDPA_CHUNKS)
        trace = _truthy(env.get("MAST3R_TRACE"), True)
        region = int(env.get("MAST3R_TRACE_REGION") or DEFAULT_TRACE_REGION_SIZE)
        if region <= 0:
            raise ValueError(f"MAST3R_TRACE_REGION must be > 0, got {region}")
        dpt_fuse = _truthy(env.get("MAST3R_DPT_FUSE"), True)
        return cls(
            enabled=True, rope=rope, rope_lut=rope_lut, mm=mm, sdpa_chunks=sdpa_chunks,
            trace=trace, trace_region_size=region, dpt_fuse=dpt_fuse, conv_weight_cache=True,
        )

    @property
    def fused_rope(self) -> bool:
        """True when q/k weights, biases and LUTs must be channel-permuted."""
        return self.rope in ("llama", "legacy")

    @property
    def fused_residual_mm(self) -> bool:
        return self.mm in ("dit", "minimal")

    @property
    def minimal_mm(self) -> bool:
        return self.mm == "minimal"

    def summary(self) -> dict:
        return asdict(self)

    def open_device_kwargs(self, **base) -> dict:
        """Keyword arguments for ``ttnn.open_device`` that select the execution mode this
        configuration needs: ``base`` (e.g. ``l1_small_size``) plus
        ``trace_region_size=self.trace_region_size`` iff the traced fused path is on.

        Without it the tree's ``DEFAULT_TRACE_REGION_SIZE`` (0 in gbp-tt) puts the device in
        the *dynamic* trace-allocation mode (``mesh_device.cpp`` ``begin_mesh_trace``:
        "trace_region_size is 0 (dynamic allocation mode)"), a third mode nobody has
        exercised for this port. The harness (``test_mast3r.py``), ``eval_mast3r.py`` and the
        server therefore all open the device through this helper, so every gate runs either
        eager (``MAST3R_TRACE=0`` / ``TT_FUSED=0``) or traced with the explicit region.
        """
        kw = dict(base)
        if self.enabled and self.trace:
            kw["trace_region_size"] = self.trace_region_size
        return kw


def parse_sdpa_chunks(value: Optional[str],
                      default: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    """``"128,128"`` -> ``(128, 128)``; unset / empty -> ``default``; one of
    :data:`SDPA_DEFAULT_WORDS` (``default`` / ``ttnn`` / ``0`` ...) -> None = ttnn's own 32/32
    program config. Chunks must be multiples of 32."""
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in SDPA_DEFAULT_WORDS:
        return None
    parts = [p.strip() for p in value.replace("x", ",").split(",")]
    if len(parts) != 2:
        raise ValueError(f"MAST3R_SDPA_CHUNKS must be 'q,k', got {value!r}")
    q, k = int(parts[0]), int(parts[1])
    if q <= 0 or k <= 0 or q % 32 or k % 32:
        raise ValueError(f"MAST3R_SDPA_CHUNKS chunks must be positive multiples of 32, got {value!r}")
    return q, k


# ---------------------------------------------------------------- RoPE reformulation

def rope_channel_perm(dh: int, mode: str) -> torch.Tensor:
    """Per-head channel permutation ``P`` (LongTensor ``[dh]``) with
    ``x_new[..., c] = x_ref[..., P[c]]`` for the given kernel ``mode``.

    ``llama``: within each 32-wide half interleave the two 16-wide quarters
    ``[a0, b0, a1, b1, ...]`` (pairs ``(2i, 2i+1)`` are what ``x @ T`` rotates).
    ``legacy``: quarter swap ``[a, c, b, d]`` (the kernel swaps the two 32-wide halves).
    """
    if dh % 4:
        raise ValueError(f"head_dim must be a multiple of 4, got {dh}")
    half, q = dh // 2, dh // 4
    if mode == "llama":
        idx = []
        for h0 in (0, half):
            for j in range(q):
                idx += [h0 + j, h0 + q + j]
    elif mode == "legacy":
        idx = list(range(0, q)) + list(range(2 * q, 3 * q)) + list(range(q, 2 * q)) + list(range(3 * q, 4 * q))
    else:
        raise ValueError(f"no channel permutation for rope mode {mode!r}")
    return torch.tensor(idx, dtype=torch.long)


def expand_head_perm(perm: torch.Tensor, heads: int) -> torch.Tensor:
    """``[dh]`` per-head permutation -> ``[heads * dh]`` over the concatenated heads."""
    dh = perm.numel()
    return torch.cat([perm + h * dh for h in range(heads)])


def permute_out_rows(weight: torch.Tensor, bias: Optional[torch.Tensor], idx: torch.Tensor):
    """Row gather of an ``nn.Linear`` weight ``(out, in)`` and bias ``(out,)``: exact."""
    w = weight[idx].contiguous()
    b = None if bias is None else bias[idx].contiguous()
    return w, b


def permute_qkv_rows(qkv_w: torch.Tensor, qkv_b: torch.Tensor, heads: int, mode: str):
    """Fused ``qkv`` linear ``(3D, D)``: permute the q rows ``[0:D]`` and k rows ``[D:2D]``
    per head; v rows untouched."""
    D3 = qkv_w.shape[0]
    D = D3 // 3
    dh = D // heads
    p = expand_head_perm(rope_channel_perm(dh, mode), heads)
    idx = torch.cat([p, p + D, torch.arange(2 * D, 3 * D)])
    return permute_out_rows(qkv_w, qkv_b, idx)


def permute_q_rows(w: torch.Tensor, b: torch.Tensor, heads: int, mode: str):
    """``projq`` linear ``(D, D)``: permute every row block per head."""
    D = w.shape[0]
    idx = expand_head_perm(rope_channel_perm(D // heads, mode), heads)
    return permute_out_rows(w, b, idx)


def permute_kv_rows(kv_w: torch.Tensor, kv_b: torch.Tensor, heads: int, mode: str):
    """Fused ``projk|projv`` linear ``(2D, D)``: permute the k rows ``[0:D]`` only."""
    D = kv_w.shape[0] // 2
    p = expand_head_perm(rope_channel_perm(D // heads, mode), heads)
    idx = torch.cat([p, torch.arange(D, 2 * D)])
    return permute_out_rows(kv_w, kv_b, idx)


def rope_cos_sin_tables(D: int, max_pos: int, base: float = 100.0,
                        angle_dtype: torch.dtype = torch.float32):
    """1-D tables ``cos, sin`` of shape ``[max_pos, D]`` for a ``D``-channel half.

    Reference (``torch_dust3r.RoPE2D.get_cos_sin``): fp32 angles, ``emb = cat(freqs, freqs)``,
    cos/sin in fp32 (``angle_dtype=float32``). The legacy port casts the ANGLE to bf16 before
    cos/sin (``angle_dtype=bfloat16``); pass that to reproduce its tables.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    if angle_dtype != torch.float32:
        emb = emb.to(angle_dtype)
    return emb.cos().float(), emb.sin().float()


def rope_lut_reference(pos: torch.Tensor, dh: int, base: float = 100.0,
                       angle_dtype: torch.dtype = torch.float32):
    """Un-permuted per-token tables ``cos_full, sin_full`` of shape ``[N, dh]`` (fp32):
    channels ``0..dh/2`` from ``pos[:, 0]`` (y), ``dh/2..dh`` from ``pos[:, 1]`` (x) -- the
    layout ``RoPE2D`` multiplies the tokens with. ``pos``: ``[N, 2]`` long."""
    if pos.dim() != 2 or pos.shape[-1] != 2:
        raise ValueError(f"pos must be [N, 2], got {tuple(pos.shape)}")
    D = dh // 2
    max_pos = int(pos.max().item()) + 1
    cos, sin = rope_cos_sin_tables(D, max_pos, base, angle_dtype)
    cos_full = torch.cat((cos[pos[:, 0]], cos[pos[:, 1]]), dim=-1)
    sin_full = torch.cat((sin[pos[:, 0]], sin[pos[:, 1]]), dim=-1)
    return cos_full, sin_full


def rope_lut_permuted(pos: torch.Tensor, dh: int, mode: str, base: float = 100.0,
                      angle_dtype: torch.dtype = torch.float32,
                      out_dtype: torch.dtype = torch.bfloat16):
    """cos/sin tables in the permuted channel order the fused kernels consume:
    ``[1, 1, N, dh]`` (``shape[0] == 1`` and ``shape[1] == 1`` are what
    ``rotary_embedding_llama`` prefill and the legacy ``rotary_embedding`` validate;
    heads and batch are broadcast by the kernels)."""
    cos_full, sin_full = rope_lut_reference(pos, dh, base, angle_dtype)
    p = rope_channel_perm(dh, mode)
    cos = cos_full[:, p].to(out_dtype).reshape(1, 1, -1, dh).contiguous()
    sin = sin_full[:, p].to(out_dtype).reshape(1, 1, -1, dh).contiguous()
    return cos, sin


def rope_trans_mat(tile: int = 32) -> torch.Tensor:
    """``[1, 1, 32, 32]`` transformation matrix of ``rotary_embedding_llama``:
    ``T[2i, 2i+1] = 1, T[2i+1, 2i] = -1`` so that ``(x @ T)[2i] = -x[2i+1]`` and
    ``(x @ T)[2i+1] = x[2i]`` (same construction as the tree's
    ``get_rot_transformation_mat``). Entries are 0 / +-1: exact in bf16."""
    T = torch.zeros(1, 1, tile, tile)
    T[..., torch.arange(0, tile, 2), torch.arange(1, tile, 2)] = 1.0
    T[..., torch.arange(1, tile, 2), torch.arange(0, tile, 2)] = -1.0
    return T


# --------------------------------------------------- host emulation of the device math

def rope2d_reference(tokens: torch.Tensor, pos: torch.Tensor, base: float = 100.0) -> torch.Tensor:
    """DUSt3R ``RoPE2D`` exactly as ``torch_dust3r.py`` writes it (self-contained copy so the
    tests can run in any dtype incl. fp64). ``tokens``: ``[B, H, N, dh]``; ``pos``: ``[B, N, 2]``."""
    D = tokens.shape[-1] // 2
    max_pos = int(pos.max().item()) + 1
    cos, sin = rope_cos_sin_tables(D, max_pos, base)
    cos, sin = cos.to(tokens.dtype), sin.to(tokens.dtype)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply1d(x, p1):
        c = cos[p1][:, None, :, :]
        s = sin[p1][:, None, :, :]
        return x * c + rotate_half(x) * s

    y, x = tokens[..., :D], tokens[..., D:]
    return torch.cat((apply1d(y, pos[..., 0]), apply1d(x, pos[..., 1])), dim=-1)


def rope_llama_emulated(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                        trans_mat: torch.Tensor, tile: int = 32) -> torch.Tensor:
    """What ``rotary_embedding_llama`` computes: ``x * cos + (x @ T) * sin`` with the matmul
    applied per 32-wide tile of the head dimension. ``x``: ``[B, H, N, dh]``; ``cos/sin``:
    ``[1, 1, N, dh]`` (broadcast over B and H); ``trans_mat``: ``[1, 1, 32, 32]``."""
    T = trans_mat.reshape(tile, tile).to(x.dtype)
    dh = x.shape[-1]
    parts = [x[..., i:i + tile] @ T for i in range(0, dh, tile)]
    rotated = torch.cat(parts, dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


def rope_legacy_emulated(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """What the legacy ``rotary_embedding`` op computes: half swap over the full head,
    ``rot(x) = [-x[dh/2:], x[:dh/2]]``, then ``x * cos + rot(x) * sin``."""
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


# ------------------------------------------------------------- other exact reformulations

def patch_im2col(img: torch.Tensor, p: int) -> torch.Tensor:
    """The port's on-device im2col as torch ops: ``(B, C, H, W) -> (B, N, C*p*p)`` via
    ``reshape(B, C, hp, p, wp, p) -> permute(0, 2, 4, 1, 3, 5) -> reshape``; feature order
    ``(c, kh, kw)`` like ``conv.weight.reshape(E, -1)``."""
    B, C, H, W = img.shape
    hp, wp = H // p, W // p
    x = img.reshape(B, C, hp, p, wp, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, hp * wp, C * p * p)


def residual_scale_ones(N: int) -> torch.Tensor:
    """The ``addcmul_input_tensor2`` of ``dit_minimal_matmul_addcmul_fused`` for a plain
    residual add: ``ones[1, 1, N]`` so ``residual + 1.0 * (h @ W + b) * ones == residual + (h @ W + b)``."""
    return torch.ones(1, 1, N)
