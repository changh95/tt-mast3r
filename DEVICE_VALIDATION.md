# DEVICE_VALIDATION -- mast3r-p150 `TT_FUSED=1` path (branch `opt/mast3r-p150-megakernel`)

> **Status 2026-09-13: device-validated, fused path is the DEFAULT since commit e2d5e5b**
> (`TT_FUSED=0` = legacy). Sections 0-4 below are the pre-hardware plan as written (they still
> say "unset = legacy"); the outcome is in [Results (device, 2026-09-13)](#results-device-2026-09-13).

Everything on this branch was written and checked **without a device** (BRIEF §0 of
`reports/megakernel/BRIEF.md`, 2026-09-13). This file is the plan for the hardware pass:
what to run, in which order, what must come out, and what nobody has verified yet.
Speed numbers below are **estimates** (arithmetic in `reports/megakernel/mast3r-p150.md`
§5) unless marked *measured*; the only measured numbers are the legacy path's.

## 0. What the branch adds (all behind `TT_FUSED=1`; unset = legacy graph, bit-for-bit)

| lever | knob (default under `TT_FUSED=1`) | exactness class | host proof |
|---|---|---|---|
| 2-D RoPE as ONE kernel per q/k: `ttnn.experimental.rotary_embedding_llama(x, cos, sin, trans_mat, is_decode_mode=False)` after a per-head channel permutation `P_il` folded into the q/k rows of `attn.qkv`, `cross_attn.projq`, fused `projk|projv` weights + biases and into the `[1,1,1024,64]` cos/sin tables; `trans_mat [1,1,32,32]` | `MAST3R_ROPE=llama` (fallbacks `legacy` = `ttnn.experimental.rotary_embedding` with quarter-swap `P_qs`; `slices` = the old 10-call chain inside the fused graph) | weight/LUT permutation **bit-identical** (`torch.equal`); kernel math bf16-rounding class (one rounding at pack with fp32 dest acc vs three today) | `test_fused_host.py`: scores equal to fp64 round-off (`< 1e-12`), permuted per-token output `torch.equal`, fold `torch.equal`, LUT shape/dtype/pair structure, `T @ T = -I` |
| cos/sin tables from fp32 angles (reference) instead of the legacy bf16-rounded angle (up to 0.06 rad error in the top channels) | `MAST3R_ROPE_LUT=fp32` (`bf16` reproduces the legacy tables exactly, for A/B attribution) | precision-affecting, expected **up** | `test_lut_bf16_angle_reproduces_legacy_port_tables` |
| Whole-graph metal trace: persistent `[2,3,512,512]` bf16 ROW_MAJOR input, `copy_host_to_device_tensor`, eager warm run, `begin/end_trace_capture`, `execute_trace(blocking=False)`, two `to_torch` (`TtDust3r`, `dust3r_forward` dispatches to it) | `MAST3R_TRACE=1`, `MAST3R_TRACE_REGION=536870912` | bit-identical (replays the same programs) | fake-ttnn graph run: 1,064 ops captured, **0** host<->device transfers and **0** host conv weights inside the capture; steady state = copy-in + `execute_trace` + 2 readbacks |
| Device-prepared conv2d / conv_transpose2d weights + biases cached after the first call (`return_weights_and_bias=True`) | always with `TT_FUSED=1` | bit-identical | 46 host weight preparations per pair -> 0 (fake-ttnn count) |
| DPT exact cleanups: TILE `ttnn.reshape` of the 4 taps (drops 8 `to_layout` per head), `Conv2dConfig(activation=relu)` on resconv conv1 and head2 (drops 8 `ttnn.relu` per head) | `MAST3R_DPT_FUSE=1` | bit-identical (relu commutes with round-to-nearest; sign of zero aside) | `test_relu_commutes_with_bf16_rounding`; op counts 37 -> 21 `to_layout`, 30 -> 14 `relu` per pair |
| `dit_minimal_matmul_addcmul_fused(h, W, 1.0, residual, ones)` for the 120 residual linears (enc proj/fc2, dec proj/cproj/fc2); `minimal_matmul` for the 144 qkv/fc1/cq/ckv with a separate exact `ttnn.gelu`; `MinimalMatmulConfig` (4,4,4,2x2) / (8,4,4,2x2), HiFi2, no fp32 acc, DRAM `ones[1,1,N]` | `MAST3R_FUSED_MM=linear` (**off**; `dit` / `minimal` for the A/B) | bf16-rounding class (different blocking) -> gate on the model metric | `test_ones_vector_residual_formulation`, `test_minimal_matmul_block_divisibility`; fake-ttnn: 120 dit + 144 minimal + 48 gelu, shapes pass the mirrored validate |
| `SDPAProgramConfig(grid, q_chunk, k_chunk, exp_approx_mode=False)` on all 72 SDPA calls (24 encoder + 48 decoder; the evaluation's "96" over-counted) | `MAST3R_SDPA_CHUNKS` unset (**off**; try `128,128`, then `128,256`) | precision-affecting (fewer online-softmax rescales; accuracy went UP in rf-detr) -> gate | `test_sdpa_work_items` (256 / 96 work items) |

Op budget per pair (fake-ttnn counts, `ttnn.*` python calls, not device programs):
legacy **2,395** (578 slices, 288 neg, 144 concat, 288 mul, 284 add, 46 host conv preps)
-> `TT_FUSED=1` **1,064** in one trace (144 `rotary_embedding_llama`, 2 slices)
-> with `MAST3R_FUSED_MM=minimal` **992**.

Host tests (torch-only, the tree's python; run from `code/`):

```bash
TREE=/home/deepgadget/experiments/gbp-tt/tt-metal
cd /home/deepgadget/experiments/tt-models/models/mast3r-p150/code
env -u TT_FUSED PYTHONPATH=.:$TREE:$TREE/ttnn:$TREE/tools TT_METAL_HOME=$TREE \
  $TREE/python_env/bin/python -m pytest -q -p no:cacheprovider models/tests/test_fused_host.py
# 30 passed in ~8 s (2026-09-13; includes 5 subprocess runs of the graph on models/tests/fake_ttnn.py,
#   one of them the capture-failure probe, plus the open_device_kwargs / harness-layer checks)
```

## 1. Environment (same as SERVING.md "Run on the host")

```bash
TREE=/home/deepgadget/experiments/gbp-tt/tt-metal      # v0.78.0-dev20260820, python 3.10, torch 2.11.0
REPO=/home/deepgadget/experiments/tt-models/models/mast3r-p150
export PYTHONPATH=$REPO/code:$TREE:$TREE/ttnn:$TREE/tools:$HTTP   # $HTTP = staged fastapi/uvicorn (SERVING.md)
export TT_METAL_HOME=$TREE
export MAST3R_WEIGHTS_DIR=<dir with model.safetensors>             # or HF_MODEL + TT_WEIGHTS_REVISION
PY=$TREE/python_env/bin/python
```

### Execution modes and how each script selects one

The port has exactly two execution modes, chosen by the environment at model build (read
once by `ttnn_dust3r.fused_config()`):

| mode | environment | device open | what runs |
|---|---|---|---|
| **eager** | `TT_FUSED` unset (legacy graph) **or** `TT_FUSED=1 MAST3R_TRACE=0` (fused ops, no trace) | `ttnn.open_device(device_id, l1_small_size=32 KiB)` -- ttnn's default trace region (0 in this tree) | one ttnn call per op, host dispatch |
| **traced** | `TT_FUSED=1` (default `MAST3R_TRACE=1`) | `ttnn.open_device(device_id, l1_small_size=32 KiB, trace_region_size=MAST3R_TRACE_REGION)` (default 512 MiB) | `TtDust3r`: eager warm run, `begin/end_trace_capture`, `execute_trace` per call |

`test_mast3r.py`, `eval_mast3r.py` and the server (`models.server.app`) all build their
`open_device` call through `FusedConfig.open_device_kwargs(...)`, print / log the resolved
configuration (`# port path: fused {...}` / `legacy`) and call `release_device_caches()` before
`close_device` (trace + persistent buffers + weight caches are freed against an open device).
So every command below runs in the mode its environment says: **no `MAST3R_TRACE=0` means
traced**, and there is no third configuration. The tree would otherwise accept a trace capture
on a device opened without a region -- `ttnn.device.DEFAULT_TRACE_REGION_SIZE == 0` selects a
*dynamic* trace-allocation mode (`tt_metal/distributed/mesh_device.cpp` `begin_mesh_trace`:
"trace_region_size is 0 (dynamic allocation mode)"); no script of this port opens the device
that way any more, and the mode is listed as unverified in §4.

Per-layer harness entries and what they exercise (all eager; the trace exists only in
`end_to_end` / the server):

| `--layer` | port function | fused levers reached with `TT_FUSED=1` |
|---|---|---|
| `full_encoder` | `full_encoder` | 48 x `rotary_embedding_llama` at `[2,16,1024,64]`, fused matmuls / SDPA chunks if enabled |
| `full_decoder` | `full_decoder` | 96 x `rotary_embedding_llama` at `[1,12,1024,64]`, same sub-knobs |
| `dpt_head_device` / `dpt_head_device_2` | `dpt_head_device` (the function `dust3r_forward` / `TtDust3r` call) | TILE tap reshape, `Conv2dConfig(activation=relu)`, cached prepared conv weights (`MAST3R_DPT_FUSE=0` isolates the first two) |
| `dpt_head` / `dpt_head_2` | `dpt_head` = **torch bf16 host reference**, runs no ttnn op | nothing -- do not use it to gate the DPT levers |
| `end_to_end` | `dust3r_forward` -> `TtDust3r` (traced) or the eager fused graph (`MAST3R_TRACE=0`) | everything |

## 2. Reference numbers to beat / not to lose (legacy path, *measured*)

| metric | value | source |
|---|---:|---|
| `test_mast3r.py --layer end_to_end` PCC (synthetic pair) | 0.9962 (gate >= 0.99) | card, results.tsv |
| same, latency best-of-25 | 228-230 ms | results.tsv |
| served `timing_ms.forward` (smoke test) | 249-276 ms | publish record |
| served end-to-end `total` (npz) | 430-445 ms (169 ms is host npz encode) | card |
| CO3Dv2 12 pairs xyz PCC head1 / head2, mean / min (`eval_mast3r.py`) | 0.9934 / 0.9777 and 0.9962 / 0.9901 | co3d_eval_results.md |
| PairViewer pose AUC@30, est. / known focal (`eval_mast3r.py`) | 38.1 / 40.4 (torch ref 35.0 / 35.4) | card, media/pose_accuracy.md |
| confidence channels PCC | 0.85-0.96 (fragile; must not fall further) | card caveats |

Gates for every fused configuration: e2e PCC >= 0.99, CO3D head1/head2 xyz PCC mean/min not
below the legacy values by more than 0.002 (dit / minimal / SDPA chunks are the levers that
can move them), pose AUC@30 >= 38.1 / 40.4, smoke test PASS with `--require-pose`.

## 3. Order of operations (A/B first, then combine)

### Step 0 -- there is no tree-side sanity test for `rotary_embedding_llama` on Blackhole

The tree's nightly file
`tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py` is a
guaranteed no-op on the p150a: **all five** of its test functions
(`test_rotary_embedding_llama`, `..._with_program_cache`, `..._direct_cos_padding_tail`,
`..._per_head`, `..._prefill_cos_sin_and_trans_mat_sharding`; decorators at lines 423, 512,
695, 716, 900) carry `@skip_for_blackhole("Requires eth connected devices to run, only single
chip BH available. See #12349")`. Do not spend device time on it. The `per_head` case
`batch2_1024` (`x[2, heads, 1024, dh]`, cos/sin stacked pairs, `trans_mat` from
`get_rot_transformation_mat`) is our exact convention on paper only. **Step 2a with
`--layer full_encoder` is therefore the first real device check of `rotary_embedding_llama`
in this tree** (48 calls at `[2,16,1024,64]`); treat a validate error / hang there as the
kernel's, not the port's, and fall back to `MAST3R_ROPE=legacy` as described under 2a.

### Step 1 -- legacy baseline on this tree (TT_FUSED unset)

```bash
cd $REPO/code && env -u TT_FUSED $PY test_mast3r.py --layer end_to_end --runs 25 --device-id 0
```

Expect PCC 0.9962, ~228-230 ms. This confirms the branch did not change the default path
(the code paths are guarded; the fake-ttnn test pins the op multiset at 2,395 calls).

### Step 2 -- `TT_FUSED=1` default (single-kernel RoPE + DPT cleanups + cached conv weights + trace)

2a. **Eager** (`MAST3R_TRACE=0`, plain device open) fused ops first, per stage, to localise
any failure. Each layer below runs a device function of the port; the legacy comparison is
the same command with `TT_FUSED` unset:

```bash
TT_FUSED=1 MAST3R_TRACE=0 $PY test_mast3r.py --layer full_encoder --runs 5       # 48 rotary_embedding_llama at [2,16,1024,64] (first device check of the op, see step 0)
TT_FUSED=1 MAST3R_TRACE=0 $PY test_mast3r.py --layer full_decoder --runs 5       # 96 at [1,12,1024,64]
TT_FUSED=1 MAST3R_TRACE=0 $PY test_mast3r.py --layer dpt_head_device --runs 5    # dpt_head_device: TILE tap reshape + relu fusion + cached prepared weights (eager)
TT_FUSED=1 MAST3R_TRACE=0 $PY test_mast3r.py --layer dpt_head_device_2 --runs 5  # branch 2 (head2 weights)
TT_FUSED=1 MAST3R_TRACE=0 MAST3R_DPT_FUSE=0 $PY test_mast3r.py --layer dpt_head_device --runs 5   # weight cache only -> bisects the DPT levers
TT_FUSED=1 MAST3R_TRACE=0 $PY test_mast3r.py --layer end_to_end --runs 10        # whole fused graph, eager
```

(`--layer dpt_head` is the torch bf16 host reference of the DPT head and runs no ttnn op; it
does not gate anything on this branch. `dpt_head_device` compares the on-device head against
the fp32 torch head on synthetic taps; its legacy PCC is not in the card, so record the
`TT_FUSED`-unset number from the same command as its baseline.)

Expected: each PCC >= its legacy value (RoPE now has one rounding and fp32 angles; the
DPT changes are exact). If `rotary_embedding_llama` fails to validate or hangs at these
shapes: `MAST3R_ROPE=legacy` (the `rotary_embedding` op, `P_qs` permutation, same host
proof) and report which; `MAST3R_ROPE=slices` keeps everything else fused. If the DPT
stage fails: the `MAST3R_DPT_FUSE=0` run isolates the TILE-reshape/relu fusion from the
weight cache. If a conv complains about its prepared weights on the second call
(`is_valid_device_conv_weights` warning "pulling back to host"), report it -- that path
would also break the trace.

2b. **Traced** (the served configuration). The harness now opens the device with the trace
region whenever `TT_FUSED=1` and `MAST3R_TRACE` is not `0` (`FusedConfig.open_device_kwargs`),
so the traced e2e is one command; the first `dust3r_forward` (the harness's warm call) runs
the eager warm pass + capture, the timed calls replay:

```bash
TT_FUSED=1 $PY test_mast3r.py --layer end_to_end --runs 25        # traced; log line "# port path: fused {... 'trace': True ...}"
```

Expected: PCC as in 2a (trace is exact); latency **estimate** ~120 ms (250 - 90 RoPE - 35
trace/dispatch - 1.4 DPT; range 100-150). Failure modes and knobs: `end_trace_capture`
errors about the trace buffer -> `MAST3R_TRACE_REGION=1073741824`; an op refusing to be
captured (`conv_transpose2d`, bilinear `upsample`, `sharded_to_interleaved`, 6-D
row-major `permute` were never traced on this box) -> the error names the op (a failed
capture releases the persistent input and the half-built trace, so the process can fall
back to `MAST3R_TRACE=0` without a leak); the eager path (2a) remains the fallback for the
device pass.

2c. Server contract:

```bash
cd /tmp && TT_FUSED=1 MAST3R_WARMUP=1 $PY -m uvicorn --host 127.0.0.1 --port 20000 --lifespan on models.server.app:app
# log must show "Port path: fused {...}", "Warmup forward 1/1", "Metal trace captured", then READY
$PY $REPO/code/models/server/smoke_test.py --url http://127.0.0.1:20000 --require-pose
curl -s localhost:20000/info | python3 -c "import json,sys; print(json.load(sys.stdin)['fused'])"
```

Also start once with `MAST3R_WARMUP=0` (the app must force one capturing forward before
READY) and check `timing_ms.forward` / `forward_sym` (the symmetric pose forward replays the
same trace with the swapped pair) against 249-276 ms.

2d. Accuracy gate for step 2 -- **traced** (the served configuration; `eval_mast3r.py` opens the
device with the trace region under `TT_FUSED=1` and its warm-up pair captures the trace; each
CO3D pair is a `[2,3,512,512]` replay, the symmetric forward replays the same trace with the
swapped pair):

```bash
env -u TT_FUSED $PY eval_mast3r.py --co3d-root <root> --category <cat> --pairs 6            # legacy baseline (table §2)
TT_FUSED=1 $PY eval_mast3r.py --co3d-root <root> --category <cat> --pairs 6                 # traced fused path (gate)
TT_FUSED=1 MAST3R_ROPE_LUT=bf16 $PY eval_mast3r.py --co3d-root <root> --category <cat> --pairs 6   # attribution: angle precision vs kernel
TT_FUSED=1 MAST3R_TRACE=0 $PY eval_mast3r.py --co3d-root <root> --category <cat> --pairs 6         # eager control only if the traced numbers differ from 2a
```

The traced and eager fused runs must agree bit for bit on the same pair (the trace replays
the same programs); if they do not, the trace is the bug, not the fusion.

### Step 3 -- fused matmuls (`MAST3R_FUSED_MM=dit`, then `minimal`) -- traced

```bash
TT_FUSED=1 MAST3R_FUSED_MM=dit     $PY test_mast3r.py --layer end_to_end --runs 25   # 120 residual linears fused (traced)
TT_FUSED=1 MAST3R_FUSED_MM=minimal $PY test_mast3r.py --layer end_to_end --runs 25   # + 144 minimal_matmul, exact gelu separate (traced)
TT_FUSED=1 MAST3R_FUSED_MM=<winner> $PY eval_mast3r.py --co3d-root <root> --category <cat> --pairs 6   # gate (traced)
```

Add `MAST3R_TRACE=0` to any of these to run the same sub-knob eagerly (e.g. to get a clean
L1 CB allocation error message outside the capture). Gate with `eval_mast3r.py` (rf-detr
lesson: PCC can rise while the task metric falls). Watch for L1 CB allocation errors at the
encoder shapes (`[2048,4096]x[4096,1024]`, `[2048,1024]x[1024,4096]` are much larger than
rf-detr's); if they appear, the block configs in `ttnn_dust3r._fused_consts` (`mm_cfg`
8/4/4, `dit_cfg` 4/4/4) are the knobs. Estimate: -4 ms from the removed adds, 0..-20 ms
from the kernel itself (unmeasured).

### Step 4 -- SDPA chunks (`MAST3R_SDPA_CHUNKS=128,128`, then `128,256`) -- traced

```bash
TT_FUSED=1 MAST3R_FUSED_MM=<best of step 3> MAST3R_SDPA_CHUNKS=128,128 $PY test_mast3r.py --layer end_to_end --runs 25
TT_FUSED=1 MAST3R_FUSED_MM=<best of step 3> MAST3R_SDPA_CHUNKS=128,128 $PY eval_mast3r.py --co3d-root <root> --category <cat> --pairs 6
```

Gate with `eval_mast3r.py`. Estimate -15..-30 ms (scaled from rf-detr's 0.30 -> 0.13 ms per
layer). No mask is passed anywhere (BRIEF §1B).

### Step 5 -- publish decision

Only after steps 2-4 pass their gates: set `serve.env.TT_FUSED: "1"` (+ the winning
sub-knobs) in `tt-model.yaml`, re-run `tt-model package` + `serve` + `smoke_test.py`, update
the card's speed line (and `/info`'s `reference_numbers`). Until then the manifest ships the
legacy path.

## 4. Not verified without hardware (honest list)

- **Any timing.** The 35 us launch floor and per-op numbers are borrowed from the rf-detr
  pass; program counts per op were estimated from the tree's dispatch code.
- `rotary_embedding_llama` at `[2,16,1024,64]` / `[1,12,1024,64]` on Blackhole in this tree:
  validate passes on paper; L1/CB fit and the `use_reload_impl` path
  (`num_rows_per_core = ceil(32 seq tiles / cores) * heads = 16 > 8`) unverified. The
  `x @ T` sign convention was cross-checked against the tree's ground truth
  (`apply_rotary_emb_qk_real`) and `tensor_utils.get_rot_transformation_mat` on the host
  only. Compute config passed: HiFi4, fp32 dest acc (allowed for head_dim <= 128),
  `packer_l1_acc=False`, vs the op's own default HiFi4 without fp32 acc.
- The legacy `rotary_embedding` fallback at batch 2 / 16 heads (validate allows it).
- Trace capture of `conv_transpose2d`, bilinear `upsample` (auto-shard + halo),
  `sharded_to_interleaved`, 6-D row-major `permute` / `reshape`; the trace region size for
  ~1.1k programs (rf-detr: 90 MB for ~260); DRAM headroom with 1.1 GB weights + ~80 MB
  prepared conv weights + 512 MiB trace region + 3 MB input + 34 MB outputs.
- The tree's **dynamic trace-allocation mode** (`trace_region_size=0`, which is
  `ttnn.device.DEFAULT_TRACE_REGION_SIZE` in gbp-tt; `mesh_device.cpp` `begin_mesh_trace` /
  `end_mesh_trace` track a DRAM high-water mark instead of using a reserved region). None of
  the port's scripts opens the device that way when tracing any more
  (`FusedConfig.open_device_kwargs`), so it is deliberately **not** part of any gate; whether
  it would even work for a 1.1k-program trace next to 1.2 GB of weights is unknown. If a
  future script opens the device without the region under `TT_FUSED=1`, it runs in this
  mode, not eagerly.
- `test_mast3r.py --layer dpt_head_device` (new on this branch) has no legacy number in the
  card; its `TT_FUSED`-unset PCC on the synthetic taps is the baseline to record in step 2a.
- Teardown: `release_device_caches()` before `close_device` in the harness / eval scripts is
  new; that the trace release + buffer deallocation ordering is clean on the real device
  (the server already did this) is unverified.
- That `conv2d` accepts our prepared weights on every later call without the
  "pulling back to host" path (`is_valid_device_conv_weights` reads true for
  TILE `[1,1,K,Cout>=out_channels]`, dtype unchecked when `weights_dtype` is unset -- it
  did for rf-detr's projector; `conv_transpose2d` has no such check and uses them as-is).
- `Conv2dConfig(activation=relu)` together with the DPT compute config (HiFi4 + fp32 dest
  acc + packer_l1_acc) on Blackhole; sign-of-zero differences are possible and harmless.
- `minimal_matmul` / `dit_minimal_matmul_addcmul_fused` L1 CB fit and speed at K = 4096 and
  N = 4096 (encoder fc1/fc2); whether `ttnn.linear` is already near the kernel's speed at
  these sizes; that a separate accurate `ttnn.gelu` matches `linear(activation="gelu")`
  (both map to `UnaryOpType::GELU` accurate; fp32-vs-bf16 input rounding differs).
- SDPA chunk choice on the 110-core grid (256 / 96 work items at 128/128).
- The accuracy effect of the fp32-angle LUT (expected up; the legacy tables carried up to
  0.06 rad angle error) -- attribute with `MAST3R_ROPE_LUT=bf16`.
- The fake-ttnn graph run checks control flow, op multiset, shapes/layouts against the ops'
  validate rules as transcribed, and host<->device hygiene inside the capture -- NOT
  kernel numerics or kwargs that the fake accepts more loosely than nanobind would.
- Readback un-padding of the `[1,1,262144,4]` TILE outputs (16.8 MB per head) and the
  host `np.savez_compressed` cost (169 ms) were left alone (evaluation §3 L7).

## Results (device, 2026-09-13)

Hardware pass on the p150a (tree `gbp-tt` v0.78.0-dev20260820 for the host runs, the shipped
image `tt-model/mast3r-p150:06f41b4a2319` for the final gate and the served A/B). Evidence:
`/home/deepgadget/experiments/tt-models/logs/megakernel-validate/mast3r/` (every log, probe and
script named below), one row per experiment in `reports/megakernel/VALIDATION.md`. Branch
`opt/mast3r-p150-megakernel`; nothing merged, pushed or re-packaged.

### What ran, in order

| step | command (host python_env unless noted) | result | log |
|---|---|---|---|
| 1 legacy regression | `env -u TT_FUSED test_mast3r.py --layer end_to_end --runs 25` (HEAD 9a047cd) | PCC **0.9968** (head1 0.9804 / head2 0.9963), best-of-25 **234.18 ms** -- card 0.9962 / 228-230 ms: unchanged within noise. New-layer baselines: `dpt_head_device` 1.0000 / 45.51 ms, `dpt_head_device_2` 1.0000 / 46.10, `full_encoder` 0.9996 / 55.86, `full_decoder` 1.0000 / 85.38 | `s1_legacy_e2e.log` |
| 2 host tests | `pytest models/tests/test_fused_host.py` | 30 passed (before the default flip), 32 passed after | `s2_host_tests.log`, `s2b_host_tests_after_flip.log` |
| 2a fused eager, per stage | `TT_FUSED=1 MAST3R_TRACE=0`, fresh process per layer | **no validate / L1-CB / layout / dtype error anywhere** (first device run of `rotary_embedding_llama` at `[2,16,1024,64]` and `[1,12,1024,64]`; the `use_reload_impl` path, prepared conv weights on every later call, `Conv2dConfig(activation=relu)` with the DPT compute config -- all fine, no "pulling back to host"). `full_encoder` 0.9996 / **21.05 ms** (-34.8), `full_decoder` 1.0000 / **29.55** (-55.8), `dpt_head_device` 1.0000 / **16.60** (-28.9; `MAST3R_DPT_FUSE=0` 18.27 -> the weight cache is -27 ms, TILE reshape + relu fusion -1.7 ms), `dpt_head_device_2` 1.0000 / 17.77, e2e eager PCC **0.9979** (0.9911 / 0.9976) / **93.88 ms** | `s3_fused_chain.log`, `s3_{enc,dec,dpt1,dpt2,dpt1_nofuse,e2e_eager}.log` |
| 2b fused traced e2e | `TT_FUSED=1 test_mast3r.py --layer end_to_end --runs 25` | capture OK on the first try (1 064-op graph incl. `conv_transpose2d`, bilinear `upsample`, 6-D RM `permute`, `sharded_to_interleaved`; 512 MiB region; JIT cache 458/458), PCC 0.9979 (= eager), **93.68 ms** | `s3_e2e_traced.log` |
| 2d accuracy (CO3D substitute, see below) | `probe_realpair.py` legacy / traced / eager; `compare_pt.py` | **eager == traced: `torch.equal` on all four outputs**; `TT_FUSED=1 MAST3R_ROPE=slices` (trace + DPT cleanups + conv cache, old RoPE chain) **== legacy bit for bit** -> the three "bit-identical" levers are bit-identical on device | `s3_realpair_*.log`, `s3b_realpair_slices.log`, `*.pt` |
| attribution | `MAST3R_ROPE_LUT=bf16`, `MAST3R_ROPE=legacy` | llama + fp32 LUT (default) e2e 0.9979 / 93.5 ms; llama + bf16 LUT 0.9966 (head1 0.9768) / 93.2; legacy rotary op + fp32 LUT 0.9967 (head1 0.9694) / 96.8 -> default kept | `s3b_e2e_{lut_bf16,rope_legacy}.log` |
| 3 fused matmuls (traced) | `MAST3R_FUSED_MM=dit` / `minimal` | dit: e2e 0.9976 / **88.71 ms** (-5.0), no L1 CB error at the encoder shapes; minimal: 0.9970 / 90.89 (slower than dit, real-pair depth error 2.5x legacy) -> **minimal dropped (knob only)** | `s3b_e2e_{dit,minimal}.log` |
| 4 SDPA chunks (traced) | `MAST3R_SDPA_CHUNKS=128,128` / `128,256` | 0.9969 / **80.91 ms**; 0.9974 / **79.31 ms** (-14.4) -> 128,256 kept | `s3b_e2e_sdpa128{,x256}.log` |
| 3+4 combined | `MAST3R_FUSED_MM=dit MAST3R_SDPA_CHUNKS=128,256` | e2e PCC **0.9970** (head1 0.9791 / head2 0.9966) / **72.96 ms** best-of-25 (**3.2x** vs 234.18) | `s3b_e2e_dit_sdpa.log` |
| 7-pair real-image A/B | `probe_multipair.py` legacy / default / +sdpa / +dit / +dit+sdpa | table below | `s3c_chain.log`, `s3c_multipair_*.log` |
| 4a/4b served A/B, shipped image | `serve.sh` (flags = `tt-model serve --print`, code bind-mounted), `serve_probe.py`, `smoke_test.py --require-pose` | table below | `s4_serve_{legacy,fused}.log`, `s4_probe_*.log`, `s4_smoke_*.log` |
| 5 final gate, shipped image (+pytest) | `s5_gate.sh final` (mast3r-dev:latest): host tests, e2e default, e2e `TT_FUSED=0` | host tests **32 passed**; code default (fused: llama, fp32, dit, (128,256), trace): PCC **0.9970** (0.9791 / 0.9966) / **72.92 ms**; `TT_FUSED=0`: PCC 0.9968 (0.9804 / 0.9963) / 236.21 ms -- legacy unchanged in the image too | `s5_gate_final.log` |
| 6 served, final code | no `TT_FUSED` in env (code default) / `TT_FUSED=0` (knob) | default: READY 5.5 s (warm-up 2383 ms, kernels cached), forward **73.68 / 73.05 / 82.39 ms**, total 241.29, smoke PASS, npz + pose byte-identical to 4b; `TT_FUSED=0`: "Port path: legacy (TT_FUSED=0)", forward 236.94 / 234.67 / 240.35, total 404.65, smoke PASS, npz + pose byte-identical to 4a; `MAST3R_WARMUP=0` (plan 2c): "Trace not captured yet ... running one capturing forward" -> "Metal trace captured" -> READY 5.8 s, forward 73.59 median over 5, smoke PASS | `s4_serve_{default,knob0}.log`, `s4_probe_*.log` |

### Accuracy on real images (CO3D substitute)

The CO3Dv2 data `eval_mast3r.py` needs (`/home/ttuser/experiments/vggt/co3d_data`) is **not on
this box**, so the §2 CO3D / pose-AUC gate could not be re-run. Substitute: the same metric
family (`probe_multipair.py`, symmetric forwards, torch fp32 reference) on the 7 real image
pairs available in the model repos' `media/` dirs (CO3Dv2 apple 0/40 = the demo pair, three
KITTI pairs from depth-anything-3, gaze-lle 1/2, moge/sam3, rf-detr/superpoint). xyz PCC vs the
reference, mean / min over the 7 pairs (raw head outputs, like `eval_pair`):

| configuration (all traced) | head1 xyz | head2 xyz | conf h1 / h2 (mean) | device forward (median of 14 real forwards) |
|---|---|---|---|---:|
| legacy (`TT_FUSED=0`) | 0.98789 / 0.97219 | 0.99426 / 0.98561 | 0.9588 / 0.9724 | 257.4 ms |
| fused, RoPE only (`MAST3R_FUSED_MM=linear MAST3R_SDPA_CHUNKS=default`) | 0.98862 / 0.97168 | 0.99430 / 0.98808 | 0.9591 / 0.9709 | 93.9 |
| + SDPA 128,256 | 0.98911 / 0.97269 | 0.99427 / 0.98499 | 0.9594 / 0.9706 | 80.1 |
| + dit | 0.98962 / 0.97099 | 0.99464 / 0.98462 | 0.9595 / 0.9723 | 87.2 |
| **+ dit + SDPA 128,256 (new default)** | **0.99032 / 0.97179** | **0.99491 / 0.98628** | 0.9601 / 0.9723 | **76.4** |

Every fused configuration is within 0.0025 of the legacy graph on every mean / min (gate:
0.002 on the means; all means are *above* legacy), confidence PCC within 0.002. On the one
genuine stereo pair (apple) the PairViewer est-focal pose differs from the torch reference by
2.57 deg (legacy) vs 3.68-4.42 deg (fused variants; 3.76 for the default); the served demo-pair
pose moved 58.65 -> 55.95 deg (reference 58.50), focal_2 320 -> 303 (reference 324). The
attribution runs show this comes from the RoPE lever itself (the `slices` variant reproduces
the legacy pose exactly, `MAST3R_ROPE_LUT=bf16` gives 8.97 deg, `MAST3R_ROPE=legacy` 3.23 deg):
PnP-RANSAC on a single pair is sensitive to bf16-level pointmap changes and is not a usable
gate here. **The 12-pair CO3Dv2 xyz / pose-AUC gate therefore remains unverified for the fused
path**; the card marks those rows as measured on the legacy graph.

### Served A/B (shipped image `06f41b4a2319`, repo code bind-mounted, 30 warm requests after 5, one 512x512 pair, npz)

| | legacy (`TT_FUSED` unset, HEAD 9a047cd) | fused (`TT_FUSED=1 MAST3R_FUSED_MM=dit MAST3R_SDPA_CHUNKS=128,256`) | final code, no env (default) | final code, `TT_FUSED=0` |
|---|---|---|---|---|
| boot | warm-up 3560 ms, READY 7.4 s | warm-up 10249 ms (fused kernels JIT-compiled into the image cache once), "Metal trace captured", READY 13.4 s | warm-up 2383 ms (cached), "Metal trace captured", READY 5.5 s | "Port path: legacy (TT_FUSED=0)", warm-up 2350 ms, READY 5.5 s |
| `timing_ms.forward` median / min / max | **236.91 / 234.48 / 246.64** | **73.57 / 73.21 / 79.89** | **73.68 / 73.05 / 82.39** | 236.94 / 234.67 / 240.35 |
| `timing_ms.total` median / min / max | 404.51 / 401.47 / 414.46 | 241.39 / 239.95 / 279.72 | 241.29 / 240.51 / 254.61 | 404.65 / 402.23 / 414.34 |
| `encode` (host `np.savez_compressed`) | 160.3 | 160.3 | 160.0 | 160.2 |
| pose request (forward / forward_sym / pose / total) | 272 / 259 / 1126 / 1856 | 84 / 73 / 1089 / 1436 | 83 / 74 / 1073 / 1419 | 272 / 258 / 1046 / 1767 |
| smoke `--require-pose` | PASS 236 / 239 / 1695 ms, rot 58.7 f1 310 f2 320 | PASS 74 / 85 / 1363 ms, rot 55.9 f1 320 f2 303 | PASS 76 / 82 / 1367 ms, rot 55.9 f1 320 f2 303 | PASS 235 / 237 / 1677 ms, rot 58.7 f1 310 f2 320 |
| malformed image / bad `output_format` | 400 / 400 | 400 / 400 | 400 / 400 | 400 / 400 |
| npz identical across the 30 requests | 30/30 | 30/30 | 30/30, and byte-identical to the fused column | 30/30, and byte-identical to the legacy column |
| `docker stop` (SIGTERM) | "Released 864 cached device tensors", "Device closed", exit 0, 1.1 s, tt-smi OK | "Released 954 ...", "Device closed", exit 0, 1.1 s, tt-smi OK | "Released 954 ...", exit 0, 1.2 s, tt-smi OK | "Released 864 ...", exit 0, 1.1 s, tt-smi OK |

Legacy vs fused served outputs on the demo pair (`compare_responses.py`): activated pts3d1 PCC
0.99982, pts3d2 0.99973, conf1 0.99879, conf2 0.99478.

### Lever verdicts

| lever | verdict | numbers |
|---|---|---|
| single-kernel 2-D RoPE (`rotary_embedding_llama`, `P_il` fold, fp32-angle LUT) | **keep (default)** | encoder 55.9 -> 21.1 ms, decoder 85.4 -> 29.6 ms; e2e PCC 0.9968 -> 0.9979; the two fallbacks work too (`legacy` op 0.9967 / 96.8 ms, `slices` = legacy bit for bit) |
| fp32-angle cos/sin tables | keep (default) | vs bf16-angle tables: e2e 0.9979 vs 0.9966, real-pair pose diff 3.7 vs 9.0 deg |
| whole-graph metal trace (`TtDust3r`, 512 MiB region) | keep (default) | bit-identical to eager (`torch.equal`); 93.88 -> 93.68 ms alone (the fused eager graph is already device-bound), but it is what makes the served forward equal the harness number (73.6 vs 73.0 ms) |
| device-cached prepared conv weights | keep (always on with the fused path) | DPT head 45.5 -> 18.3 ms; bit-identical |
| DPT TILE tap reshape + conv/relu fusion | keep (default) | 18.27 -> 16.60 ms; bit-identical |
| `dit_minimal_matmul_addcmul_fused` residual linears | **keep (new default `MAST3R_FUSED_MM=dit`)** | -5.0 ms alone / -6.4 ms on top of SDPA; 7-pair xyz PCC mean up (0.98862 -> 0.98962 / 0.99430 -> 0.99464); synthetic head1 0.9911 -> 0.9874 |
| `minimal_matmul` for qkv/fc1/cq/ckv | **drop (knob only)** | 90.89 ms vs dit's 88.71; e2e 0.9970, real-pair depth error h2 0.0550 (legacy 0.0224) |
| SDPA program config 128,256 | **keep (new default `MAST3R_SDPA_CHUNKS=128,256`)** | -14.4 ms; e2e 0.9974; 7-pair xyz mean 0.98911 / 0.99427 (>= legacy); 128,128 was 1.6 ms slower and less accurate (0.9969) |

### Default flip (commit e2d5e5b)

`TT_FUSED` unset or `1` = fused (llama RoPE, fp32 LUT, trace, DPT cleanups, conv cache, dit,
SDPA 128/256); `TT_FUSED=0` = the legacy eager graph, bit for bit. `tt-model.yaml` `serve.env`
pins `TT_FUSED=1 MAST3R_FUSED_MM=dit MAST3R_SDPA_CHUNKS=128,256`, a `verify` line checks the
knob, the card's speed / accuracy rows are the measured numbers above (no cold-boot, no
best-case rows; the CO3Dv2 rows are marked as legacy-graph measurements). `/info.reference_numbers`
carries the fused and legacy numbers.

### Not verified / left open

- The 12-pair CO3Dv2 xyz PCC and PairViewer pose AUC@30 of §2 (`eval_mast3r.py`): data set
  absent on this box. The 7-pair substitute passes; the single-stereo-pair pose shift above is
  the one signal an owner may want to re-check with the real data set before merging.
- `MAST3R_TRACE_REGION`: 512 MiB was never exceeded (no capture error); the actual high-water
  mark of the 944-program trace was not measured, 1 GiB was not needed.
- The dynamic trace-allocation mode (`trace_region_size=0`): still not exercised by any script.
- The image's kernel cache now holds the fused kernels (`~/.cache/tt-model/mast3r-p150/cache`);
  a consumer's first boot pays the ~10 s JIT once (legacy: ~3.6 s warm / 49 s cold).
- The package was **not** rebuilt (`tt-model package`) and nothing was pushed; the shipped
  image still carries the legacy default until the owner re-packages from this branch.
- `minimal_matmul` (dropped) was measured only at the default 8/4/4 blocking; other block
  configs were not explored.

## Point-map quality fix (2026-09-14)

Owner report: "MASt3R seems to produce bad results based on visualization inspection" -- the
demo `media/output.png` showed the two predicted pointmaps as two separate sheets instead of
one scene. Evidence for everything below: `/home/deepgadget/experiments/tt-models/logs/pointmap/mast3r/`
(scripts `pm_common.py`, `step0_upstream_vs_ref.py`, `step2_preproc_study.py`, `device_pairs.py`,
`analyze_pairs.py`, `served_client.py`; chains `s1_before_chain.sh`, `s4_after_chain.sh`,
`s4b_fern_chain.sh`, `s5_served_check.sh`; renders `before_*_pad.png`, `after_*_pad.png`,
`after_*_crop.png`, `step2_fixed_*.png`; tables `*_metrics.md/json`; raw device outputs `dev_*.pt`).
Host python_env of the gbp-tt tree, one p150a, nothing left running (tt-smi OK after every step).

Metrics used throughout (host, on the ACTIVATED pointmaps): per-head PCC and relative error
`||a-b||/||b||` vs the torch fp32 reference on the same preprocessed inputs; **overlap** =
median nearest-neighbour distance of the conf-filtered (top 50 %) view-2 cloud into the
view-1 cloud and vice versa, divided by the robust bbox diagonal of the view-1 cloud
(`sym_med`; both clouds are in the camera-1 frame, so one scene gives a few 1e-3, two sheets
give ~0.5), plus the fraction of view-2 points within 2 % of the scale (`inlier 2%`);
PairViewer pose (est. focal, `conf_pct` 50) as rotation magnitude and rotation difference to
the reference pose. Renders: union of both heads coloured by source pixels (two 3-D views,
top-down) plus a top-down coloured by head, top 70 % confidence per view.

### 1. Root cause: the decoder taps, not the padding

Upstream DUSt3R (`dust3r/heads/dpt_head.py` `create_dpt_head`, dust3r tree on this box at
`/home/deepgadget/monst3r/MASt3R-SLAM/thirdparty/mast3r/dust3r`) hooks `hooks_idx=[0, 6, 9, 12]`
into the 13-entry list `_decoder` builds -- `[enc, blk0, ..., blk11]` with `dec_norm` applied to
the last entry -- i.e. the DPT heads see the encoder output, decoder **block 5**, **block 8**
and **dec_norm(block 11)**. The port's torch reference (`Decoder.forward`, `tap_layers=(0, 6, 11)`)
and the ttnn port (`full_decoder`, `tap_layers = (0, 6, 11)`, `compute_norm=False`) fed them
blocks 0 / 6 / 11 **un-normed**. Every PCC gate compared the port with that reference, so the
1.0000 decoder / DPT PCCs and the 0.997 end-to-end PCC were correct statements about the wrong
network. Upstream `AsymmetricCroCo3DStereo` run on the CPU from the same HF snapshot
(`step0_upstream_vs_ref.py`, gray-pad 512x512 inputs, activated outputs):

| pair (pad 512x512) | network | pts3d1 PCC / rel err | pts3d2 PCC / rel err | conf PCC 1 / 2 | overlap sym_med | inlier 2% | PairViewer rot | rot vs upstream |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| apple | upstream DUSt3R | 1 | 1 | 1 | 0.0083 | 0.587 | 64.30 deg (CO3D GT 63.71) | - |
| apple | port reference, taps 0/6/11 (before) | 0.97888 / 0.266 | 0.97965 / 0.466 | 0.843 / 0.530 | **0.4165** | **0.000** | 58.50 | **17.10 deg** |
| apple | port reference, taps 5/8/norm(11) (fixed) | **1.00000 / 0.0000** (max abs 1.8e-7) | 1.00000 / 0.0000 | 1.0000 / 1.0000 | 0.0083 | 0.587 | 63.86 | 0.59 |
| kitchen 00/03 | upstream DUSt3R | 1 | 1 | 1 | 0.0029 | 0.737 | 43.43 | - |
| kitchen 00/03 | port reference, taps 0/6/11 (before) | 0.95409 / 0.375 | 0.91328 / 0.830 | 0.836 / 0.703 | **0.6127** | **0.000** | 35.05 | **22.21 deg** |
| kitchen 00/03 | port reference, taps 5/8/norm(11) (fixed) | 1.00000 / 0.0000 (7.2e-7) | 1.00000 / 0.0000 | 1.0000 / 1.0000 | 0.0029 | 0.737 | 43.43 | 0.00 |

So the old reference was a different network (0.95-0.98 PCC, 27-83 % pointmap error, two
disjoint sheets, 17-22 deg of pose error), and with the upstream taps it **is** upstream to
fp32 round-off -- with the gray-padded square input. The gray padding was not the defect.

### 2. Reproduction on the device (before the fix, HEAD 3a47b3d, pad 512x512)

`s1_before_chain.sh` (legacy `TT_FUSED=0`, then fused default) on the apple, kitchen 00/03,
kitchen 00/08 and the depth-anything-3 `source_1/2` pair; `analyze_pairs.py --taps current`
against the OLD reference (`before_pad_metrics.md`, renders `before_{apple,kitchen,kitchen08,kitti}_pad.png`):

| pair | config | pts3d1 / pts3d2 PCC vs old ref | overlap sym_med (old ref) | inlier 2% | PairViewer rot | rot vs old ref |
|---|---|---:|---:|---:|---:|---:|
| apple | legacy | 0.9991 / 0.9990 | 0.390 (0.417) | 0.000 | 58.65 | 2.57 |
| apple | fused | 0.9992 / 0.9988 | 0.416 (0.417) | 0.000 | 55.95 | 3.76 |
| kitchen | legacy | 0.9884 / 0.9958 | 0.572 (0.613) | 0.000 | 47.44 | 13.71 |
| kitchen | fused | 0.9884 / 0.9956 | 0.545 (0.613) | 0.000 | 31.42 | 11.40 |
| kitchen08 | legacy | 0.9882 / 0.9921 | 0.633 (0.650) | 0.000 | 39.82 | 15.32 |
| kitchen08 | fused | 0.9884 / 0.9921 | 0.646 (0.650) | 0.000 | 33.82 | 25.78 |

The device reproduced the reference faithfully (0.99 PCC) -- and the reference's two sheets
(`before_apple_pad.png`: red view-1 cloud and blue view-2 cloud do not overlap in any row).
The `source_1/source_2` "KITTI pair" of depth-anything-3 turned out to be two **different
streets** (`source_3` is a third one): DUSt3R gives it head-2 confidence ~1.0 and no coherent
overlap in every configuration, which is the right answer. It is kept in the tables as a
negative control and replaced by LLFF `fern` 000/002 as the third genuine pair.

### 3. Preprocessing study (torch fp32, fixed taps, `step2_preproc_study.py`, `step2_fixed_*.png`)

(i) `pad` = the port's gray-pad square 512x512, (ii) `native` = upstream `load_images(size=512)`
(long side 512, centre crop to a multiple of 16; the padded CO3D demo pair is un-padded first),
(iii) `crop` = centre square crop -> 512x512, (iv) `edgepad` = edge-replication padding instead
of gray. The reference DPT head needed upstream's `refinenet4(...)[:, :, :H3, :W3]` crop for the
odd 32x21 token grid of 512x336 (added; a no-op at 32x32).

| pair | preprocessing | input | overlap sym_med | inlier 2% | PairViewer rot | focals |
|---|---|---|---:|---:|---:|---:|
| apple | pad | 512x512 | 0.0083 | 0.587 | 63.86 (GT 63.71) | 494 / 477 (GT 456) |
| apple | native | 512x224 | 0.0074 | 0.545 | 63.66 | 481 / 474 |
| apple | crop | 512x512 | 0.0120 | 0.655 | 56.84 | 777 / 751 (cropped grid) |
| apple | edgepad | 512x512 | 0.0039 | 0.697 | 64.29 | 524 / 500 |
| kitchen | pad | 512x512 | 0.0029 | 0.737 | 43.43 | 455 / 450 |
| kitchen | native | 512x336 | 0.0021 | 0.756 | 43.91 | 447 / 443 |
| kitchen | crop | 512x512 | 0.0091 | 0.645 | 46.35 | 687 / 675 |
| kitchen | edgepad | 512x512 | 0.0034 | 0.677 | 43.30 | 461 / 473 |
| kitchen08 | pad | 512x512 | 0.0035 | 0.749 | 42.01 | 464 / 458 |
| kitchen08 | native | 512x336 | 0.0022 | 0.904 | 42.17 | 449 / 443 |
| kitchen08 | crop | 512x512 | 0.0071 | 0.642 | 43.02 | 675 / 737 |
| kitchen08 | edgepad | 512x512 | 0.0046 | 0.743 | 44.68 | 484 / 497 |
| kitti (negative control) | pad / native / crop / edgepad | | 0.10 / 0.41 / 0.24 / 0.25 | 0.00-0.34 | 166 / 174 / 108 / 57 | head-2 conf mean 1.0-1.1 |

Decision: with the taps fixed, gray padding already gives coherent single-scene pointmaps on
every genuine pair (overlap 0.003-0.008, apple rotation 0.15 deg from the GT magnitude);
`native` is only marginally tighter (0.002-0.007, focal ~3 % closer to CO3D's) and would need
per-shape device graphs (positions, RoPE tables, DPT reshapes, trace shape, server warm-up) --
**not implemented**, rejected by the knob with that explanation; `crop` is worse (0.007-0.012,
7 deg off on apple: it discards the periphery) but harmless and cheap, kept as a knob;
`edgepad` is mixed (better on apple, worse on kitchen; its fabricated border geometry gets
real confidence) -- not adopted. **`MAST3R_PREPROC=pad` (default) | `crop`.**

### 4. The fix (this branch)

- `reference/torch_dust3r.py`: `DPT_TAP_BLOCKS = (5, 8)`; `Decoder.forward` returns the two
  block taps plus the dec_norm'ed final output as the third tap (upstream hooks); DPT head crops
  `refinenet4` to layer 3's grid (odd token grids; no-op at 32x32).
- `tt/ttnn_dust3r.py`: `full_decoder` taps `DPT_TAP_BLOCKS`, always runs the two `dec_norm`
  `ttnn.layer_norm`s on device and hands `dec_norm(block 11)` to the DPT head as the last tap
  (`compute_norm` now only controls the host download). +2 ttnn calls per pair: legacy graph
  2397, fused 946 in the trace; `test_fused_host.py` counts updated, **32 passed**.
- `postprocess.py`: `preprocess_image(img, size, mode)` with `pad` | `crop`, record gains
  `mode` (offsets <= 0 for crop, same `(x + offx) * scale` mapping, `intrinsics_to_canonical`
  unchanged); `preprocess_mode()` reads `MAST3R_PREPROC` (`native` -> explicit ValueError).
- `server/app.py`: `_Config.preproc`, used in `/predict`, echoed in `preprocess[].mode`,
  `/info.input.preprocess(_mode)`, startup log. `tt-model.yaml`: `MAST3R_PREPROC: "pad"` in
  `serve.env`, verify lines for the knob and `DPT_TAP_BLOCKS`, card rows re-measured (below).
- `media/output.png` re-rendered from the fixed device output (fused default);
  `media/pose_accuracy.md`, README, SERVING.md updated; the pre-fix CO3Dv2 12-pair / AUC@30 /
  ETH3D numbers describe the old network and were removed from the card (CO3D / ETH3D data
  are not on this host).

### 5. Device validation of the fixed port (pad 512x512 unless stated; `after_*_metrics.md`)

`s4_after_chain.sh` + `s4b_fern_chain.sh`: legacy (`TT_FUSED=0`), fused default, fused with
`MAST3R_ROPE=slices`, fused on `crop`; harness gates. Reference = the fixed torch fp32 network.

| pair | config | pts3d1 PCC / rel | pts3d2 PCC / rel | conf PCC 1 / 2 | overlap sym_med (torch) | inlier 2% (torch) | PairViewer rot (torch) | rot vs torch | focals (torch) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| apple | legacy | 0.9994 / 0.030 | 0.9996 / 0.027 | 0.996 / 0.994 | 0.0123 (0.0083) | 0.540 (0.587) | 63.92 (63.86) | 0.31 | 479/465 (494/477) |
| apple | **fused (default)** | **0.9995 / 0.029** | **0.9996 / 0.026** | 0.996 / 0.995 | 0.0119 | 0.577 | **63.97** | **0.27** | 481/459 |
| apple | fused, ROPE=slices | 0.9995 / 0.029 | 0.9996 / 0.026 | 0.996 / 0.995 | 0.0121 | 0.563 | 63.76 | 0.26 | 481/464 |
| kitchen 00/03 | legacy | 0.9902 / 0.131 | 0.9906 / 0.120 | 0.992 / 0.995 | 0.0039 (0.0029) | 0.690 (0.737) | 43.47 (43.43) | 0.14 | 442/436 (455/450) |
| kitchen 00/03 | **fused** | 0.9901 / 0.132 | 0.9907 / 0.120 | 0.992 / 0.994 | 0.0040 | 0.694 | 43.29 | 0.23 | 442/436 |
| kitchen 00/03 | slices | 0.9903 / 0.131 | 0.9908 / 0.119 | 0.992 / 0.995 | 0.0040 | 0.686 | 43.59 | 0.19 | 442/436 |
| kitchen 00/08 | legacy | 0.9878 / 0.147 | 0.9921 / 0.116 | | 0.0043 (0.0035) | 0.712 (0.749) | 41.84 (42.01) | 0.25 | 451/445 (464/458) |
| kitchen 00/08 | **fused** | 0.9875 / 0.149 | 0.9920 / 0.117 | | 0.0043 | 0.707 | 41.95 | 0.07 | 450/444 |
| kitchen 00/08 | slices | 0.9877 / 0.147 | 0.9920 / 0.117 | | 0.0044 | 0.708 | 41.92 | 0.14 | 452/445 |
| fern 000/002 | legacy | 0.9977 / 0.058 | 0.9967 / 0.070 | | 0.0211 (0.0218) | 0.447 (0.422) | 5.71 (5.54) | 0.20 | 442/447 (453/465) |
| fern 000/002 | **fused** | 0.9978 / 0.057 | 0.9968 / 0.069 | | 0.0210 | 0.449 | 5.38 | 0.22 | 440/450 |
| fern 000/002 | slices | 0.9976 / 0.059 | 0.9968 / 0.069 | | 0.0203 | 0.460 | 5.29 | 0.32 | 440/448 |
| kitti (negative control) | fused | 0.9975 / 0.070 | 0.9940 / 0.109 | | 0.112 (0.101) | 0.017 (0.001) | 165 (166) | 34 (PnP on noise) | |
| apple, `MAST3R_PREPROC=crop` | fused | 0.9989 / 0.041 | 0.9991 / 0.038 | | 0.0115 (0.0120) | 0.611 (0.655) | 56.86 (56.84) | 0.23 | 743/716 (777/751) |
| kitchen 00/03, crop | fused | 0.9899 / 0.132 | 0.9989 / 0.040 | | 0.0080 (0.0091) | 0.607 (0.645) | 45.75 (46.35) | 0.79 | |
| kitchen 00/08, crop | fused | 0.9896 / 0.134 | 0.9958 / 0.080 | | 0.0075 (0.0071) | 0.616 (0.642) | 42.66 (43.02) | 0.51 | |
| fern, crop | fused | 0.9989 / 0.040 | 0.9981 / 0.053 | | 0.0258 (0.0280) | 0.352 (0.323) | 4.55 (4.74) | 0.26 | |

Before -> after on the served default (fused, pad): apple overlap 0.416 -> 0.012 (torch
0.417 -> 0.008), kitchen 0.545 -> 0.004 (0.613 -> 0.003), kitchen08 0.646 -> 0.004
(0.650 -> 0.0035); inlier fraction 0 % -> 58 / 69 / 71 %; PairViewer rotation vs the (now
correct) reference 3.8 / 11.4 / 25.8 deg -> 0.27 / 0.23 / 0.07 deg. Renders
`after_{apple,kitchen,kitchen08,fern}_pad.png`: one scene in every row.

Against the targets set for this fix: **pose within 1 deg of torch** -- 0.07-0.32 deg on all
four genuine pairs, every configuration (met). **Overlap within 10 % of torch** -- met on fern
(0.0210 vs 0.0218) and on every `crop` pair; on apple / kitchen / kitchen08 the device's
`sym_med` is 0.004-0.012 against torch's 0.003-0.008, i.e. +25-45 % relative, while both are
<= 1.2 % of the scene scale and the pre-fix values were 0.39-0.65: the metric is a median
nearest-neighbour distance and bf16 point jitter sets its floor, so the relative gap is the
port's noise floor, not a coherence difference; the `inlier 2%` fraction is within 2-6 % of
torch on those pairs (not met as literally stated on three pairs; the absolute values say the
clouds are one scene). **Per-head pts3d PCC >= 0.9970** -- met on apple (0.9995 / 0.9996) and
fern (0.9978 / 0.9968 on head 2, 0.0002 short), not on the kitchen pairs (0.9875-0.9907;
relative error 12-15 %): this is the same fidelity class the pre-fix port had on the same
pairs (0.988 / 0.996, `before_pad_metrics.md`) and its 7-pair 2026-09-13 minimum (0.972), so
the fix did not regress it; the kitchen scene's depth range (0.5-3 m of DUSt3R units, thin
structures) is where bf16 shows. The gate the number 0.9970 came from is the synthetic
end-to-end harness, which improved:

| gate (fixed port, 2026-09-14) | result | log |
|---|---|---|
| `test_mast3r.py --layer end_to_end --runs 25`, fused default | PCC **0.9987** (head1 0.9983 / head2 0.9985), **73.11 ms** best-of-25 (before: 0.9970 / 72.9 ms) | `s4_e2e_fused_after.log` |
| same, `TT_FUSED=0` | PCC 0.9989 (0.9985 / 0.9987), 238.99 ms; idle-CPU re-run 238.85 ms (before: 0.9968 / 236 ms) | `s4_e2e_legacy_after.log`, `s4b_e2e_legacy_idle.log` |
| `test_mast3r.py --layer full_decoder --runs 5`, fused | PCC 1.0000 (b1 0.9999 / b2 1.0000), 20.93 ms | `s4_full_decoder_after.log` |
| `pytest models/tests/test_fused_host.py` | 32 passed (op counts 2397 / 1066 incl. the two `dec_norm` layer_norms) | `s2_host_tests_fix.log` |
| device forward, real pairs (`device_pairs.py`, median of 5) | fused 74.2-74.5 ms; `MAST3R_ROPE=slices` 198-209 ms; legacy 261 ms (idle) | `s4_*_after.log`, `s4b_*_fern.log` |

**RoPE lever verdict**: on the real pairs the fused default (`llama` RoPE) is not measurably
worse than `slices` (legacy numerics inside the fused graph, `torch.equal` to legacy) or legacy:
identical PCC to 4 decimals, pose differences 0.07-0.32 deg for all three, no ordering. The
pre-fix "2.57 -> 3.76 deg" pose shift attributed to the RoPE lever was PnP on the old network's
non-overlapping sheets. **`MAST3R_ROPE=llama` stays the default**; `slices` costs 198-209 ms
per pair vs 74 ms (2.7x) and buys nothing measurable.

### 6. Not verified / left open

- CO3Dv2 12-pair PCC / pose AUC@30 and the ETH3D `kicker` run: the data are not on this host;
  the pre-fix numbers were dropped from the card, no replacement measured. `eval_mast3r.py` /
  `eval_eth3d.py` run unchanged against the fixed reference (`load_image_for_dust3r` default
  `pad`), so an owner with the data can re-measure.
- Only the GT rotation *magnitude* of the demo pair is on this host (63.71 deg); 63.97 deg is a
  magnitude comparison, not a rotation error.
- `native` aspect (512x384-class) is not implemented (per-shape device graphs); measured in torch
  only as marginally better than `pad`.
- The package was not rebuilt (`tt-model package`) and nothing was pushed; the shipped image
  `tt-model/mast3r-p150:2e8fe3cc610d` still carries the old taps until the owner re-packages.
  The served-path check below ran that image with the fixed code bind-mounted.

### 7. Served-path check (`s5_served_check.sh`, shipped image `2e8fe3cc610d`, fixed code bind-mounted, fused default)

Boot: "Preprocessing: MAST3R_PREPROC=pad (gray-pad to square, bicubic 512x512)", "Metal trace
captured", READY 6.9 s. `served_client.py` (apple pair, `return_pose: true`, npz decoded on the
host): served pts3d1 / pts3d2 PCC vs the fixed torch reference **0.99948 / 0.99961**, conf
0.9957 / 0.9945, overlap 0.0119 (torch 0.0083, inlier 57.7 % vs 58.7 %), PairViewer rotation
**63.97 deg** (torch 63.86, 0.27 deg apart), focals 481 / 459 -- identical to the host-probe
numbers of the same configuration, `preprocess[].mode = "pad"`, `/info.input.preprocess_mode =
"pad"`; `timing_ms.forward` 81.8 (first request) / `forward_sym` 73.9 ms. `smoke_test.py
--require-pose`: PASS, forward 73 ms, pose rot 64.0 deg f1 481 f2 459. `docker stop`: "Released
954 cached device tensors", "Device closed", exit 0 in 1.0 s; `docker ps` empty, tt-smi OK.
Logs `s5_served_check.log`, `s5_serve.log`, `s5_client.log`, `s5_smoke.log`, `s5_info.json`,
`served_apple_resp.json`, `served_apple.npz`.

## Two-view point-map demo through the served path (2026-09-14)

The card demo was re-made from the SERVED model on the kitchen pair (VGGT `kitchen`
frames 00 / 03, 779x520, now `media/source_1.png` / `source_2.png`; the CO3Dv2 apple pair
moved to `logs/pointmap/mast3r/apple_source_{1,2}.png`). Server = the built image
`tt-model/mast3r-p150:2e8fe3cc610d` launched with the `tt-model serve ... --print` flags plus
the working-tree `code/models/{demos,server}` bind-mounted (the tap fix is not rebuilt into the
image yet; `logs/pointmap/mast3r-p150/serve_current.sh`, boot log `serve.log`: fused, pad,
"Metal trace captured", READY 8.0 s). `code/make_demo.py` (rewritten: server npz by default,
`--local` for the in-process ttnn port) POSTed the pair with `return_pose`, saved the raw
response (`served_kitchen_00_03.npz` + `_resp.json`) and rendered `media/pointmap.png`
(`make_demo.log`):

| quantity (served, fused default, pad) | value |
|---|---:|
| device forward / total request (`timing_ms`, with the symmetric forward + PnP) | 89.5 / 534.9 ms (first call 83.1 / 674.0) |
| view-1 / view-2 confidence median within the photo | 12.45 / 14.80 (view 1 has 32 % of its pixels at conf < 1.5: the garden seen through the window) |
| points plotted per view (top 70 %, conf >= 3, padding dropped) | 122k / 122k |
| two-cloud coherence, median NN distance view-1 <-> view-2 / scene scale | **0.0083** (torch fp32 reference on the same pair: 0.0029, `after_pad_metrics.md`; before the tap fix 0.39-0.65) |
| PairViewer rotation / focals (512 grid) | 43.29 deg / 442, 436 px (torch 43.43 deg / 455, 450) |

`served_kitchen_00_03_by_head.png` (evidence, coloured by head) shows the red view-1 and blue
view-2 clouds on one table plane in the top-down / front / side projections; `media/pointmap.png`
was inspected: one coherent scene (bulldozer on the two mats, table plane), no second sheet.
Container stopped with `docker stop` (SIGTERM): "Released 954 cached device tensors", "Device
closed", `docker ps` empty, `tt-smi -s` OK.
