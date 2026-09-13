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
