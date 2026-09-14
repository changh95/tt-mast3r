# mast3r-p150 (DUSt3R ViT-L/16, 512x512 pair) — Blackhole p150a vs RTX 5090 (same host, same weights, same input)

Date 2026-09-14. Facts only; every GPU number below was measured in this pass, every p150a number is copied (with its
source line) from the validation / publish reports. The p150a was NOT touched.

## What was run

| | |
|---|---|
| Model | The port's own torch reference: `models/mast3r-p150/code/models/demos/mast3r/reference/torch_dust3r.py::load_dust3r(state)` -> `DUSt3R` (ViT-L/16 encoder 24 blocks dim 1024 / 16 heads, dual-branch decoder 2 x 12 blocks dim 768 / 12 heads with cross-attention + `dec_norm`, DPT taps `DPT_TAP_BLOCKS=(5, 8)` + `dec_norm(11)`, two DPT heads), 2D RoPE base 100, explicit `q@k^T` softmax attention (no SDPA). 571,170,440 parameters. This is the network the p150a port is PCC-gated against (`test_mast3r.py --layer end_to_end`, the point-map study, the served A/B) and, since the 2026-09-14 tap fix, equal to upstream `AsymmetricCroCo3DStereo` to fp32 round-off (DEVICE_VALIDATION.md "Point-map quality fix" §1) |
| Weights | `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` @ `61c57447d7b0adc8a1a30b2b0adec7a8935aa2a3` (tt-model.yaml `weights.revision` = `serve.env.TT_WEIGHTS_REVISION`), `model.safetensors` (2.28 GB, 1001 fp32 tensors) from the HF cache `~/.cache/huggingface/hub/models--naver--DUSt3R_ViTLarge_BaseDecoder_512_dpt/snapshots/61c57447…`, resolved by the port's `resolve_checkpoint_path`, `HF_HUB_OFFLINE=1` |
| Input | `media/source_1.png` + `media/source_2.png` (kitchen 00/03, 779x520 RGB PNG each — the pair of the p150a Hub run and the demo) -> `postprocess.preprocess_image(im, 512, "pad")` exactly as `server/app.py::predict` with `MAST3R_PREPROC=pad`: gray-pad (128) to 779x779, PIL bicubic 512x512, /255, (x-0.5)/0.5 -> two `[1,3,512,512]` fp32 tensors (3.1 MB each), batch 1 pair, `symmetric=False` (no pose) |
| Output | two raw `[1,4,512,512]` fp32 maps (view 1, view 2: xyz pre-activation + raw confidence, 4.2 MB each), readback to the host; activation (`activate_pts3d` expm1, `activate_conf` 1+exp) is host post-processing on both sides |
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 580.126.18, power limit 600 W, 32607 MiB; idle 31.3 W |
| venv | `/home/deepgadget/experiments/tt-models/.venv-gpu/main` — Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19.0, torchvision 0.26.0, numpy 1.26.4, safetensors 0.8.0, huggingface_hub 1.31.0, pillow 12.3.0, triton 3.6.0 |
| Repo state | `models/mast3r-p150` @ `d68e9b9` (branch `tt-model-package`), read-only; `reference/torch_dust3r.py` and `postprocess.py` imported unchanged |
| Script | `logs/gpu-vs-p150/mast3r/bench_mast3r_gpu.py` (uses `logs/gpu-vs-p150/bench_common.py`); logs `full_run.log` (the run below), `smoke.log` (3-iteration dry run); raw JSON `reports/gpu-vs-p150/mast3r.json` (= `logs/gpu-vs-p150/mast3r/result.json`, incl. every raw wall-clock sample); CPU reference tensors `cpu_fp32_reference.pt` |
| Command | `cd logs/gpu-vs-p150/mast3r && HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python bench_mast3r_gpu.py --iters 50 --warmup 10 > full_run.log 2>&1` (one process; exit 0) |
| Loop | per precision and variant 10 warm-ups + 50 timed iterations, `torch.cuda.synchronize()` before and after each; wall-clock (`perf_counter`) is the primary number, CUDA-event time recorded alongside (within 0.02 ms of wall everywhere) |
| p150a source | `reports/gpu-vs-p150/p150_numbers.json` -> `reports/megakernel/POINTMAP_SUMMARY.md:80` (Hub `tt serve` after the point-map fix, same demo pair, 30 warm: forward **73.90 ms** median, min 72.97, total **254.90**), `models/mast3r-p150/DEVICE_VALIDATION.md` "## Results" served A/B (final code column, one 512x512 pair: forward 73.68 / 73.05 / 82.39, total 241.29, encode 160.0), `reports/megakernel/VALIDATION_SUMMARY.md:33`, `PUBLISH_SUMMARY.md:12` (73.9) |

Timing definitions (they match the p150a `timing_ms` keys of `server/app.py`):

- **incl_h2d** = `img1.to("cuda")` + `img2.to("cuda")` (2 x 3.1 MB pageable host tensors, as the server preprocess produces) + forward + both maps `.float().cpu()` (2 x 4.2 MB). Compare with p150a `timing_ms.forward` = `_forward(img1, img2, symmetric=False)`: upload of both views into the persistent input buffer + one whole-graph metal-trace replay (encoder x2, decoder x2, two DPT heads) + readback of both raw maps (**73.9 ms**).
- **excl_h2d** = forward only, inputs already resident, outputs left on the device.
- **served-like** = base64 decode + PNG decode of both views (`decode`) + `preprocess_image` pad x2 (`preprocess`) + incl_h2d forward (`forward`) + `activate_pts3d`/`activate_conf` + `np.savez_compressed` of pts3d float32 / conf float16 (`encode`) = the interval `server/app.py` reports as `timing_ms.total` (HTTP/JSON framing is outside on both sides). Compare with p150a `timing_ms.total` (**254.9 ms**, of which ~181 ms is the same host work; the container's own split on a 512x512 pair was decode 5.5 / preprocess 1.9 / encode 160).

GPU implementation variants (same module object, same weights, same input; nothing in the network changes):

- **ref** (primary) = the reference exactly as shipped. Its `RoPE2D.__call__` evaluates `int(positions.max())` on a CUDA tensor for every q and every k — 192 device syncs per forward, which serialise kernel launch with execution.
- **ref_nosync** = the same module with that sequence length memoised per (shape, device); verified **bit-identical** to `ref` in fp32 strict (`torch.equal` on both maps). `torch.compile` runs on this variant (the `.max()` sync is a graph break otherwise).
- **bf16_weights / fp16_weights** = `model.to(dtype)` once (weights and activations in that dtype, closer to the p150a's bf16-weights-on-device class), inputs cast on device inside the timed region, no autocast; ref_nosync RoPE.
- **SDPA** (informational only) = the three attention modules' `forward` swapped for `F.scaled_dot_product_attention` (what a stock GPU deployment of DUSt3R with xformers/SDPA runs); mathematically the same attention, fp rounding differs; ref_nosync RoPE.

## Correctness check (GPU vs CPU fp32 reference)

CPU fp32 (same process, 16 threads, same padded pair): forward 4209 ms; activated view-1 / view-2 z medians 0.4744 / 0.4913, conf means 10.67 / 12.03 (saved to `cpu_fp32_reference.pt`).

| GPU precision (variant) | raw head-1 / head-2 PCC (`test_mast3r.py` metric) | max abs diff raw | activated pts3d1 / pts3d2 PCC (card metric) | rel. err pts3d1 / pts3d2 | conf1 / conf2 PCC |
|---|---:|---:|---:|---:|---:|
| **fp32 strict** (ref) | **1.000000 / 1.000000** | 1.1e-4 / 2.5e-4 | **1.000000 / 1.000000** | 1e-6 / 1e-6 | 1.000000 / 1.000000 |
| tf32 (ref) | 1.000000 / 1.000000 | 0.043 / 0.041 | 1.000000 / 1.000000 | 3.4e-4 / 2.0e-4 | 1.000000 / 1.000000 |
| bf16 autocast (ref) | 0.999972 / 0.999984 | 0.93 / 0.62 | 0.999972 / 0.999989 | 7.2e-3 / 4.1e-3 | 0.99988 / 0.99985 |
| fp16 autocast (ref) | 1.000000 / 1.000000 | 0.10 / 0.11 | 0.999999 / 1.000000 | 1.0e-3 / 7.1e-4 | 0.999999 / 0.999999 |
| bf16_weights (module in bf16) | 0.999975 / 0.999981 | 0.65 / 1.04 | 0.999982 / 0.999988 | 5.9e-3 / 4.4e-3 | 0.99988 / 0.99986 |
| fp16_weights (module in fp16) | 1.000000 / 1.000000 | 0.089 / 0.095 | 1.000000 / 1.000000 | 9.7e-4 / 6.8e-4 | 0.999999 / 0.999999 |
| SDPA fp32 strict / tf32 / bf16 / fp16 / bf16_weights | 1.0 / 1.0 / 0.999972 / 1.0 / 0.999978 (head 1) | | 1.0 / 1.0 / 0.999961 / 1.0 / 0.999983 (pts3d1) | | |
| compile tf32 / fp16 autocast / bf16_weights (ref_nosync) | 1.0 / 1.0 / 0.999985 (head 1) | 0.031 / 0.093 / 0.69 | 1.0 / 1.0 / 0.999984 | 2.6e-4 / 6.9e-4 / 5.5e-3 | |
| p150a fused default (bf16 device), this pair (DEVICE_VALIDATION §5 kitchen 00/03) | e2e harness 0.9987 (synthetic pair, POINTMAP_SUMMARY:44) | | **0.9901 / 0.9907** | 0.132 / 0.120 | 0.992 / 0.994 |

PCC > 0.999 holds for fp32 strict (1.000000 on both raw maps, max abs diff 2.5e-4 on values of magnitude ~1-30), so the GPU runs the right network
and weights. All GPU precisions, including bf16 autocast (the p150a's activation class), stay above 0.99997 raw / 0.99996 activated PCC on this
pair; the p150a's own fused graph is at 0.9901 / 0.9907 activated pts3d PCC on the same pair (relative pointmap error 12-13 % vs 0.4-0.7 % for GPU
bf16). fp16 autocast is numerically fine here (1.000000 raw, no non-finite values: the raw confidence channel is pre-activation, so `1+exp(c)`
overflow is not an issue inside the network).

## GPU latency (batch 1, one 512x512 pair, median / min / p90 of 50 iterations, wall-clock ms)

Eager PyTorch, the reference **as shipped** (primary):

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power mean W (excl loop) | GPU util % (excl) | peak mem alloc / reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **103.409** / 102.898 / 108.976 | **102.278** / 101.735 / 107.502 | 102.266 | 102.4 (219.4 on the very first call incl. CUDA/cuDNN init) | 509.4 | 87.6 | 2619 / 2892 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **70.753** / 70.473 / 76.025 | **69.983** / 69.360 / 75.091 | 69.968 | 70.0 | 393.3 | 77.8 | 2744 / 3022 |
| bf16 autocast (+TF32 remainder) | **63.753** / 63.461 / 69.001 | **62.613** / 62.408 / 67.689 | 62.597 | 64.4 | 377.5 | 79.1 | 2676 / 3022 |
| fp16 autocast (+TF32 remainder) | **56.552** / 56.320 / 61.862 | **55.505** / 55.201 / 56.150 | 55.493 | 55.9 | 393.6 | 76.2 | 2676 / 3022 |

The incl-excl gap of ~1.1 ms is the PCIe traffic (6.2 MB upload + 8.4 MB readback); the fp32-tensor readback dominates. Power in the incl loop was
within 5 W of the excl loop in every row.

Same module, sync-free RoPE length lookup (**ref_nosync**, bit-identical output):

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | power W (excl) | GPU util % | peak mem MiB |
|---|---:|---:|---:|---:|---:|
| fp32 strict | 95.279 / 94.704 / 101.652 | 94.295 / 93.440 / 100.442 | 547.8 | 93.9 | 2619 |
| tf32 | 61.879 / 61.773 / 68.073 | 60.828 / 60.748 / 66.077 | 435.7 | 93.0 | 2744 |
| bf16 autocast | 55.128 / 55.050 / 56.561 | 54.046 / 53.955 / 54.149 | 435.9 | 89.4 | 2676 |
| fp16 autocast | 48.452 / 48.319 / 54.246 | 47.440 / 47.355 / 52.193 | 442.1 | 88.0 | 2676 |
| tf32, pinned host input (informational) | 61.760 / 61.674 / 67.695 | | | | |
| bf16 autocast, pinned host input (informational) | 55.011 / 54.939 / 58.718 | | | | |

Removing the 192 per-forward syncs is worth 8-9 ms at every precision (GPU utilisation 76-88 % -> 88-94 %). Pinned host memory changes the upload
by < 0.15 ms.

Module converted once to a low-precision dtype (weights + activations; ref_nosync), and the informational SDPA variant:

| variant | incl_h2d median / min / p90 | excl_h2d median / min / p90 | power W (excl) | peak mem MiB | raw PCC h1 / h2 |
|---|---:|---:|---:|---:|---:|
| bf16_weights (`model.to(bfloat16)`) | **42.329** / 42.244 / 42.954 | **41.277** / 41.177 / 41.885 | 432.3 | 3705 | 0.999975 / 0.999981 |
| fp16_weights (`model.to(float16)`) | **39.459** / 39.096 / 41.429 | **38.308** / 38.065 / 39.316 | 441.0 | 4839 | 1.000000 / 1.000000 |
| SDPA, fp32 strict | 93.925 / 93.151 / 100.193 | 92.966 / 92.718 / 99.322 | 501.1 | 4915 | 1.0 / 1.0 |
| SDPA, tf32 | 63.207 / 63.045 / 65.396 | 62.023 / 61.883 / 68.214 | 416.6 | 5044 | 1.0 / 1.0 |
| SDPA, bf16 autocast | 41.130 / 41.061 / 43.331 | 40.176 / 40.086 / 41.559 | 371.7 | 4976 | 0.999972 / 0.999983 |
| SDPA, fp16 autocast | 40.754 / 40.678 / 42.144 | 39.741 / 39.675 / 40.089 | 392.9 | 4978 | 1.0 / 1.0 |
| SDPA, bf16_weights | 35.327 / 35.258 / 36.075 | 34.285 / 34.242 / 35.815 | 401.3 | 4843 | 0.999978 / 0.999979 |

Autocast re-casts the 2.3 GB of fp32 weights on every forward, which is why the converted module is 13-14 ms faster than autocast at the same
dtype. The explicit `softmax(q@k^T)` attention over 1024 tokens costs ~13-14 ms more than the fused SDPA kernels in bf16/fp16 (the fp32 SDPA path
falls back to the math kernel, so no gain there).

`torch.compile` (inductor, `dynamic=False`, on ref_nosync; inductor's default on-disk cache; compile 52-59 s per variant, under the 5-min budget):

| variant | compile s | incl_h2d median / min / p90 | excl_h2d median / min / p90 | power W (excl) | peak mem MiB | raw PCC h1 / h2 | pts3d1 / pts3d2 PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| tf32 + compile default | 51.6 | 51.342 / 51.137 / 52.504 | 50.159 / 50.087 / 55.948 | 435.5 | 4909 | 1.0 / 1.0 | 1.0 / 1.0 |
| fp16 autocast + compile default | 54.9 | 25.803 / 25.739 / 26.407 | 24.871 / 24.809 / 25.095 | 446.2 | 4649 | 1.0 / 1.0 | 1.0 / 1.0 |
| fp16 autocast + compile reduce-overhead (CUDA graphs) | 58.6 | **23.214** / 23.165 / 23.264 | **22.149** / 22.122 / 22.229 | 501.2 | 4490 | 1.0 / 1.0 | 1.0 / 1.0 |
| bf16_weights + compile reduce-overhead | 54.4 | **21.106** / 21.070 / 22.041 | **20.109** / 20.092 / 20.265 | 487.5 | 4494 | 0.999985 / 0.999986 | 0.999984 / 0.999989 |

fp16 autocast was chosen for the compile rows as the faster of the two autocast modes with PCC >= 0.99 (bf16 autocast was not compiled separately).
After the compile the benchmark helper's own first call took another 4.9-7.4 s for the three fp32-module variants (a second dynamo/inductor pass;
21 ms for the bf16-module variant) — every timed number comes after the 10 warm-ups (min ~= median in all compile rows). Compiled fp16 output
equals eager fp16 to 1e-7 in PCC.

Other facts: `load_dust3r(state)` + `.cuda()` 3.09 s (module build + `load_state_dict` from the fp32 state dict; the safetensors read itself is
memory-mapped, 0.01-0.03 s); first fp32 call 219 ms (CUDA/cuDNN init); idle GPU power 31.3 W; the model is compute-bound at batch 1 (GPU utilisation
76-97 %, 370-550 W in every forward loop).

Served-like loop (same host work as `server/app.py::predict`, npz output, no pose; reference as shipped; 50 iterations after 10 warm-ups, medians ms):

| GPU precision | decode (base64 + 2x PNG 779x520) | preprocess (2x pad + bicubic) | forward (incl_h2d) | encode (activate + `np.savez_compressed`) | **total** | p90 total | power W (loop) | npz bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict | 13.00 | 11.59 | 103.87 | 172.42 | **301.37** | 306.9 | 236.9 | 6,180,437 |
| tf32 | 12.99 | 11.72 | 71.26 | 172.15 | **268.16** | 272.3 | 160.7 | 6,182,191 |
| bf16 autocast | 12.94 | 11.88 | 64.66 | 151.47 | **241.22** | 246.1 | 157.6 | 5,105,967 |
| fp16 autocast | 12.98 | 11.68 | 57.40 | 172.79 | **254.82** | 260.4 | 146.8 | 6,080,787 |
| p150a Hub run, same pair (POINTMAP_SUMMARY.md:80) | (total - forward = ~181 ms of host work; container split on a 512x512 pair: decode 5.5 / preprocess 1.9 / encode 160.3, DEVICE_VALIDATION Results) | | 73.90 | | **254.90** | | not measured | |

The host `np.savez_compressed` of the two float32 pointmaps dominates both sides (172 ms here, 160 ms in the p150a container). The bf16 row's encode
is 21 ms cheaper only because bf16-rounded maps compress to a smaller npz (5.1 vs 6.2 MB). PNG decode of this pair costs 13 ms here; the p150a
container's 5.5 ms figure is for a 512x512 pair, so the served totals are compared on the Hub run's 254.9 (same 779x520 pair) with the 241.29
DEVICE_VALIDATION figure as the alternative.

## Comparison with the p150a (matching definitions)

Ratio = p150a ms / GPU ms (> 1 means the GPU is faster). p150a precision: bf16 activations and bf16 weights on device, fp32-angle RoPE LUT, dit
residual matmuls, SDPA chunks 128/256, one whole-graph metal trace (`TT_FUSED=1`, tt-metal 0.65.2.dev9100 @ 8b98410e7). GPU precision per row as stated.

| row | p150a (definition) | GPU variant / precision | GPU ms | ratio p150a/GPU |
|---|---:|---|---:|---:|
| device forward (p150a `timing_ms.forward` = upload both views + trace replay + readback both maps vs GPU incl_h2d) | 73.9 (POINTMAP_SUMMARY.md:80, Hub tt serve, same pair; 73.68 DEVICE_VALIDATION served A/B) | ref, fp32 strict | 103.409 | **0.71** (p150a faster) |
| | | ref, tf32 | 70.753 | **1.04** |
| | | ref, bf16 autocast (the p150a's precision class) | 63.753 | **1.16** |
| | | ref, fp16 autocast | 56.552 | **1.31** |
| | | ref_nosync, fp32 strict / tf32 / bf16 / fp16 | 95.279 / 61.879 / 55.128 / 48.452 | 0.78 / 1.19 / 1.34 / 1.53 |
| | | bf16_weights (module in bf16, eager) | 42.329 | **1.75** |
| | | fp16_weights (module in fp16, eager) | 39.459 | **1.87** |
| | | tf32 + compile default | 51.342 | 1.44 |
| | | fp16 autocast + compile reduce-overhead | 23.214 | **3.18** |
| | | bf16_weights + compile reduce-overhead | 21.106 | **3.50** |
| | | SDPA (informational): tf32 / bf16 autocast / fp16 autocast / bf16_weights | 63.207 / 41.130 / 40.754 / 35.327 | 1.17 / 1.80 / 1.81 / 2.09 |
| GPU forward only (excl_h2d, no PCIe) vs the same p150a 73.9 (which cannot exclude its transfers) | 73.9 | ref, fp32 strict / tf32 / bf16 / fp16 | 102.278 / 69.983 / 62.613 / 55.505 | 0.72 / 1.06 / 1.18 / 1.33 |
| | | bf16_weights / fp16_weights eager | 41.277 / 38.308 | 1.79 / 1.93 |
| | | fp16 autocast + compile RO / bf16_weights + compile RO | 22.149 / 20.109 | 3.34 / 3.68 |
| served e2e (p150a `timing_ms.total` vs GPU served-like total, same host stages, npz) | 254.9 (POINTMAP_SUMMARY.md:80, same pair) | ref, fp32 strict | 301.37 | **0.85** |
| | | ref, tf32 | 268.16 | **0.95** |
| | | ref, bf16 autocast | 241.22 | **1.06** |
| | | ref, fp16 autocast | 254.82 | **1.00** |
| served e2e vs the DEVICE_VALIDATION served A/B total (512x512 pair, cheaper decode) | 241.29 | ref, fp32 strict / tf32 / bf16 / fp16 | 301.37 / 268.16 / 241.22 / 254.82 | 0.80 / 0.90 / 1.00 / 0.95 |

Reading: on the device-forward definition (upload + network + readback of both maps) the p150a's fused bf16 trace (73.9 ms) is **faster than the
RTX 5090 running the port's fp32 reference eagerly in strict fp32 (103.4 ms, ratio 0.71)** and on par with it under TF32 (70.8 ms, 1.04); in the
p150a's own precision class (bf16 autocast) the eager reference is 1.16x faster (63.8 ms), 1.31x in fp16 autocast. The eager reference leaves a lot
on the table for the GPU, though — it re-casts 2.3 GB of fp32 weights per forward under autocast, uses explicit softmax attention and syncs the
device 192 times per forward: converting the module once to bf16 gives 42.3 ms (1.75x), and `torch.compile` with CUDA graphs on the bf16 module
gives 21.1 ms (3.50x) at 0.99998 raw PCC — with the SDPA attention as a further informational 35.3 ms eager point. On accuracy, every GPU precision
tracks the fp32 reference far more closely (activated pts3d PCC >= 0.99996) than the p150a's bf16 graph does on this pair (0.9901 / 0.9907).
End-to-end the ~170-200 ms of identical host work (PNG decode, pad, npz compression) on both sides compresses everything to 0.85-1.06x
(241-301 vs 254.9 ms): the served latency of this model is host-bound on either accelerator as long as the response is a compressed fp32 npz.

Not measured / not claimed: p150a power (not measured in any pass -> no power or efficiency comparison; the GPU drew 370-550 W mean in the forward
loops, 150-240 W in the host-bound served-like loops, 31 W idle). p150a numbers were not re-measured. The GPU numbers exclude HTTP/JSON framing and
the base64 encoding of the npz response, as do the p150a `timing_ms` keys. The pose path (`return_pose`, symmetric forward + PnP) was not run on
either side in these rows. bf16 autocast was not `torch.compile`d (fp16 autocast, the faster of the two with PCC >= 0.99, was); no
`reduce-overhead` run of the tf32 module.

## Reproduce

```bash
cd /home/deepgadget/experiments/tt-models/logs/gpu-vs-p150/mast3r
HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python bench_mast3r_gpu.py --iters 50 --warmup 10 | tee full_run.log
# outputs: /home/deepgadget/experiments/tt-models/reports/gpu-vs-p150/mast3r.json, ./result.json, ./cpu_fp32_reference.pt
# --no-compile skips the torch.compile rows; --skip-cpu reuses cpu_fp32_reference.pt
```

GPU released after the run: `nvidia-smi --query-compute-apps=pid --format=csv,noheader` -> empty.
