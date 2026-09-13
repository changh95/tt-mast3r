# SPDX-License-Identifier: Apache-2.0
"""A shape-propagating stand-in for ``ttnn`` so the DUSt3R device graph can be *executed
on the host without a device* (no kernels, no numerics): every op returns a
:class:`FakeTensor` carrying shape / layout / placement, and every call is recorded.

Used by ``mock_graph_run.py`` (installed as ``sys.modules["ttnn"]`` BEFORE importing
``ttnn_dust3r``) to check what the real ``import ttnn`` cannot: that every fused branch
runs, which ops the legacy vs fused graphs issue and how often, that no host->device
transfer and no host conv weight is seen inside the trace capture, and that the constant
tables handed to the fused ops have the shapes the ops validate. It knows nothing about
the real kernels' numerics -- the device pass remains the only proof of those.
"""
from __future__ import annotations

import math
from collections import Counter

import torch

TILE_LAYOUT = "TILE"
ROW_MAJOR_LAYOUT = "ROW_MAJOR"
bfloat16 = "bf16"
DRAM_MEMORY_CONFIG = "DRAM"
L1_MEMORY_CONFIG = "L1"

CALLS: Counter = Counter()
LOG: list = []            # (op, info) in call order
CAPTURING = [False]
VIOLATIONS: list = []
_NEXT_TRACE = [1]


def reset():
    CALLS.clear()
    LOG.clear()
    VIOLATIONS.clear()
    CAPTURING[0] = False


def _rec(op, **info):
    CALLS[op] += 1
    LOG.append((op, CAPTURING[0], info))


def _host_transfer(op):
    if CAPTURING[0]:
        VIOLATIONS.append(f"host->device transfer inside trace capture: {op}")


class FakeTensor:
    def __init__(self, shape, layout=TILE_LAYOUT, on_device=True, sharded=False, memcfg=DRAM_MEMORY_CONFIG):
        self.shape = tuple(int(s) for s in shape)
        self.layout = layout
        self.on_device = on_device
        self.sharded = sharded
        self.memcfg = memcfg

    def is_sharded(self):
        return self.sharded

    def memory_config(self):
        return self

    @property
    def buffer_type(self):
        return self.memcfg

    def __repr__(self):
        return f"FakeTensor{self.shape}/{self.layout}/{'dev' if self.on_device else 'host'}"


Tensor = FakeTensor


class MathFidelity:
    LoFi, HiFi2, HiFi3, HiFi4 = "LoFi", "HiFi2", "HiFi3", "HiFi4"


class UnaryOpType:
    RELU, GELU = "RELU", "GELU"


class UnaryWithParam:
    def __init__(self, op, *params):
        self.op, self.params = op, params


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def WormholeComputeKernelConfig(**kw):
    return _Cfg(kind="compute", **kw)


def init_device_compute_kernel_config(arch, **kw):
    return _Cfg(kind="compute", arch=arch, **kw)


def MinimalMatmulConfig(**kw):
    return _Cfg(kind="mm", **kw)


def SDPAProgramConfig(**kw):
    return _Cfg(kind="sdpa", **kw)


def Conv2dConfig(**kw):
    return _Cfg(kind="conv", **kw)


class FakeDevice:
    def arch(self):
        return "BLACKHOLE"

    def compute_with_storage_grid_size(self):
        return (13, 10)

    def enable_program_cache(self):
        pass


# ------------------------------------------------------------------ tensors

def from_torch(t, dtype=None, layout=ROW_MAJOR_LAYOUT, device=None, memory_config=None):
    _rec("from_torch", device=device is not None, layout=layout, shape=tuple(t.shape))
    if device is not None:
        _host_transfer("from_torch(device=)")
    return FakeTensor(t.shape, layout, on_device=device is not None, memcfg=memory_config or DRAM_MEMORY_CONFIG)


def to_device(t, device, memory_config=None):
    _rec("to_device")
    _host_transfer("to_device")
    return FakeTensor(t.shape, t.layout, on_device=True)


def copy_host_to_device_tensor(host, dev, cq_id=None):
    _rec("copy_host_to_device_tensor")
    _host_transfer("copy_host_to_device_tensor")
    assert host.shape == dev.shape and host.layout == dev.layout, (host, dev)


def to_torch(t):
    _rec("to_torch", shape=t.shape)
    if CAPTURING[0]:
        VIOLATIONS.append("device->host readback inside trace capture")
    return torch.zeros(t.shape, dtype=torch.bfloat16)


def deallocate(t):
    _rec("deallocate")


def is_tensor_storage_on_device(t):
    return t.on_device


def synchronize_device(device):
    _rec("synchronize_device")


def reshape(t, shape, memory_config=None):
    _rec("reshape", layout=t.layout)
    shape = tuple(int(s) for s in shape)
    assert math.prod(shape) == math.prod(t.shape), (t.shape, shape)
    return FakeTensor(shape, t.layout, t.on_device, t.sharded, t.memcfg)


def permute(t, dims):
    _rec("permute")
    return FakeTensor([t.shape[d] for d in dims], t.layout, t.on_device)


def to_layout(t, layout):
    _rec("to_layout", to=layout)
    return FakeTensor(t.shape, layout, t.on_device, t.sharded, t.memcfg)


def sharded_to_interleaved(t, memcfg):
    _rec("sharded_to_interleaved")
    return FakeTensor(t.shape, t.layout, t.on_device, False, memcfg)


def _same(op):
    def f(t, *a, **k):
        _rec(op)
        return FakeTensor(t.shape, t.layout, t.on_device, t.sharded, t.memcfg)
    return f


neg, relu, gelu = _same("neg"), _same("relu"), _same("gelu")


def add(a, b, **k):
    _rec("add")
    return FakeTensor(a.shape, a.layout, a.on_device, a.sharded, a.memcfg)


def mul(a, b, **k):
    _rec("mul")
    return FakeTensor(a.shape, a.layout, a.on_device)


def slice(t, start, end, **k):  # noqa: A001 - mirrors ttnn's name
    _rec("slice", aligned=all(s % 32 == 0 for s in start[-1:]) and all(e % 32 == 0 for e in end[-1:]))
    return FakeTensor([e - s for s, e in zip(start, end)], t.layout, t.on_device)


def concat(ts, dim, **k):
    _rec("concat", n=len(ts))
    shape = list(ts[0].shape)
    shape[dim] = sum(t.shape[dim] for t in ts)
    return FakeTensor(shape, ts[0].layout, ts[0].on_device)


def linear(a, w, bias=None, activation=None, **k):
    _rec("linear", activation=activation)
    assert a.layout == TILE_LAYOUT and w.layout == TILE_LAYOUT, (a, w)
    assert a.shape[-1] == w.shape[-2], (a.shape, w.shape)
    if bias is not None:
        assert bias.shape[-1] == w.shape[-1]
    return FakeTensor(a.shape[:-1] + (w.shape[-1],), TILE_LAYOUT, True)


def layer_norm(x, weight=None, bias=None, epsilon=1e-5, **k):
    _rec("layer_norm")
    return FakeTensor(x.shape, x.layout, x.on_device)


def upsample(x, scale_factor=2, mode="bilinear", **k):
    _rec("upsample", mode=mode)
    assert x.layout == ROW_MAJOR_LAYOUT and len(x.shape) == 4, x
    B, H, W, C = x.shape
    return FakeTensor((B, H * scale_factor, W * scale_factor, C), ROW_MAJOR_LAYOUT, True, sharded=True, memcfg=L1_MEMORY_CONFIG)


def _conv_common(op, input_tensor, weight_tensor, bias_tensor, out_channels, batch_size, Ho, Wo,
                 return_weights_and_bias, conv_config):
    host_w = not weight_tensor.on_device
    _rec(op, host_weight=host_w, relu=bool(conv_config is not None and getattr(conv_config, "activation", None) is not None),
         out_channels=out_channels)
    if host_w and CAPTURING[0]:
        VIOLATIONS.append(f"{op} with HOST weight tensor inside trace capture")
    if bias_tensor is not None and not bias_tensor.on_device and CAPTURING[0]:
        VIOLATIONS.append(f"{op} with HOST bias tensor inside trace capture")
    out = FakeTensor((1, 1, batch_size * Ho * Wo, out_channels), TILE_LAYOUT, True)
    if return_weights_and_bias:
        pw = weight_tensor if weight_tensor.on_device else FakeTensor((1, 1, 32, max(32, out_channels)), TILE_LAYOUT, True)
        pb = None
        if bias_tensor is not None:
            pb = bias_tensor if bias_tensor.on_device else FakeTensor((1, 1, 32, max(32, out_channels)), TILE_LAYOUT, True)
        return out, (pw, pb)
    return out


def conv2d(*, input_tensor, weight_tensor, bias_tensor=None, device, in_channels, out_channels, batch_size,
           input_height, input_width, kernel_size, stride=(1, 1), padding=(0, 0), compute_config=None,
           conv_config=None, return_weights_and_bias=False, **k):
    assert input_tensor.shape[-1] == in_channels, (input_tensor.shape, in_channels)
    assert input_tensor.shape[-2] == batch_size * input_height * input_width, (input_tensor.shape, input_height, input_width)
    Ho = (input_height + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    Wo = (input_width + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    return _conv_common("conv2d", input_tensor, weight_tensor, bias_tensor, out_channels, batch_size, Ho, Wo,
                        return_weights_and_bias, conv_config)


def conv_transpose2d(*, input_tensor, weight_tensor, bias_tensor=None, device, in_channels, out_channels, batch_size,
                     input_height, input_width, kernel_size, stride=(1, 1), padding=(0, 0),
                     return_weights_and_bias=False, **k):
    assert input_tensor.shape[-1] == in_channels, (input_tensor.shape, in_channels)
    Ho = (input_height - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
    Wo = (input_width - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
    return _conv_common("conv_transpose2d", input_tensor, weight_tensor, bias_tensor, out_channels, batch_size, Ho, Wo,
                        return_weights_and_bias, None)


# ---------------------------------------------------------------- transformer

class transformer:
    @staticmethod
    def split_query_key_value_and_split_heads(qkv, kv_input_tensor=None, num_heads=1, transpose_key=False, **k):
        _rec("split_qkv_heads", cross=kv_input_tensor is not None)
        B, N, W = qkv.shape
        if kv_input_tensor is None:
            D = W // 3
            dh = D // num_heads
            shp = (B, num_heads, N, dh)
            return FakeTensor(shp), FakeTensor(shp), FakeTensor(shp)
        D = W
        dh = D // num_heads
        _, M, W2 = kv_input_tensor.shape
        assert W2 == 2 * D, (qkv.shape, kv_input_tensor.shape)
        return FakeTensor((B, num_heads, N, dh)), FakeTensor((B, num_heads, M, dh)), FakeTensor((B, num_heads, M, dh))

    @staticmethod
    def scaled_dot_product_attention(q, k, v, is_causal=True, program_config=None, **kw):
        _rec("sdpa", program_config=program_config is not None, causal=is_causal)
        assert "attn_mask" not in kw or kw["attn_mask"] is None, "BRIEF: no explicit SDPA mask"
        assert q.shape[-1] == k.shape[-1] == v.shape[-1] and q.shape[:2] == k.shape[:2] == v.shape[:2], (q, k, v)
        if program_config is not None:
            assert program_config.q_chunk_size % 32 == 0 and program_config.k_chunk_size % 32 == 0
        return FakeTensor(q.shape)

    @staticmethod
    def concatenate_heads(x, **k):
        _rec("concatenate_heads")
        B, H, S, dh = x.shape
        return FakeTensor((B, S, H * dh))


class experimental:
    @staticmethod
    def rotary_embedding_llama(x, cos, sin, trans_mat, is_decode_mode=True, compute_kernel_config=None, **k):
        _rec("rotary_embedding_llama", decode=is_decode_mode, fp32acc=getattr(compute_kernel_config, "fp32_dest_acc_en", None))
        # the op's prefill validate (rotary_embedding_llama_device_operation.cpp)
        assert not is_decode_mode
        assert all(t.layout == TILE_LAYOUT and t.on_device for t in (x, cos, sin, trans_mat)), (x, cos, sin, trans_mat)
        head_dim = x.shape[-1]
        assert head_dim <= 128 or not compute_kernel_config.fp32_dest_acc_en
        assert cos.shape == sin.shape and cos.shape[0] == 1 and cos.shape[-1] == head_dim, (cos.shape, x.shape)
        assert cos.shape[1] in (1, x.shape[1]), (cos.shape, x.shape)
        assert trans_mat.shape == (1, 1, 32, 32), trans_mat.shape
        assert cos.shape[-2] >= x.shape[-2], "cos tiles must cover the sequence (else zero-filled output)"
        return FakeTensor(x.shape)

    @staticmethod
    def rotary_embedding(x, cos, sin, token_index=None, **k):
        _rec("rotary_embedding")
        X = x.shape[-1]
        assert X == 32 or X % 64 == 0, X
        assert cos.shape[0] == 1 and cos.shape[1] == 1 and cos.shape[-1] == X and cos.shape[-2] >= x.shape[-2], (cos.shape, x.shape)
        assert cos.shape == sin.shape
        return FakeTensor(x.shape)

    @staticmethod
    def _validate_minimal_matmul(a, w, bias_tensor, config):
        """minimal_matmul_device_operation.cpp validate, shape/layout part."""
        assert a.layout == TILE_LAYOUT and w.layout == TILE_LAYOUT
        assert all(s == 1 for s in w.shape[:-2]) and a.shape[-1] == w.shape[-2], (a.shape, w.shape)
        N = w.shape[-1]
        if bias_tensor is not None:
            assert bias_tensor.shape[-1] == N and all(s == 1 for s in bias_tensor.shape[:-1]) and N % 32 == 0
        if config is not None:
            assert config.M_block_size % config.subblock_h == 0 and config.N_block_size % config.subblock_w == 0
            rows = math.prod(a.shape[:-1]) // 32
            assert rows % config.M_block_size == 0 and (N // 32) % config.N_block_size == 0, (a.shape, N, config.__dict__)
        return FakeTensor(a.shape[:-1] + (N,))

    @staticmethod
    def minimal_matmul(a, w, bias_tensor=None, *, config=None, compute_kernel_config=None, **k):
        _rec("minimal_matmul")
        return experimental._validate_minimal_matmul(a, w, bias_tensor, config)

    @staticmethod
    def dit_minimal_matmul_addcmul_fused(a, w, scalar, residual, scale_vec, bias_tensor=None, *, config=None,
                                         compute_kernel_config=None, **k):
        _rec("dit_minimal_matmul_addcmul_fused")
        out = experimental._validate_minimal_matmul(a, w, bias_tensor, config)
        M, N = a.shape[-2], w.shape[-1]
        assert residual.shape[-2] == M and residual.shape[-1] == N and residual.shape == out.shape, (residual.shape, out.shape)
        assert scale_vec.shape[-1] == N and scale_vec.shape[-2] in (1, M), scale_vec.shape
        assert scale_vec.layout == TILE_LAYOUT and residual.layout == TILE_LAYOUT
        assert scale_vec.memcfg == residual.memcfg, "scale vector must share the residual's buffer type"
        assert scalar == 1.0
        return FakeTensor(residual.shape, TILE_LAYOUT, True, memcfg=residual.memcfg)


# --------------------------------------------------------------------- trace

def begin_trace_capture(device, cq_id=None):
    _rec("begin_trace_capture")
    assert not CAPTURING[0]
    CAPTURING[0] = True
    tid = _NEXT_TRACE[0]
    _NEXT_TRACE[0] += 1
    return tid


def end_trace_capture(device, trace_id, cq_id=None):
    _rec("end_trace_capture")
    assert CAPTURING[0]
    CAPTURING[0] = False


def execute_trace(device, trace_id, cq_id=None, blocking=True):
    _rec("execute_trace", blocking=blocking)


def release_trace(device, trace_id):
    _rec("release_trace")
