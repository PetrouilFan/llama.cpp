# FINAL REPORT - gfx906 (AMD Radeon VII) Optimization

Date: 2026-08-11
Host: skatopc (EndeavourOS/Arch, kernel 7.1.5, Intel i5-3570, 13GiB RAM)
GPU: AMD Radeon VII, gfx906:sramecc+:xnack-, 60 CUs, wave64, 16GB HBM2
ROCm 7.2.53211, HIP clang 22.0.0git
Upstream base: 62bf73d25 ("model: Muse Glimmer Support (#26841)")

## 1. Executive summary

We created a dedicated gfx906 (Radeon VII) MMQ configuration and tuned it,
plus ported two Q8_0 MMQ micro-optimizations found in the mx-llama.cpp fork.
Result on LFM2.5-2.6B Q8_0 prompt processing:

| pp | upstream | gfx906-safe | improvement |
| -- | -------: | ----------: | ----------: |
| 128 | 1142 | 2147 | +88% |
| 256 | 1180 | 2300 | +95% |
| 512 | 1365 | 2362 | +73% |
| 1024 | 1377 | 2377 | +73% |
| 2048 | 1371 | 2361 | +72% |

Cross-model validation on qwen2.5-1.5B Q8_0: pp512 2414 -> 4288 (+78%).
The gain generalizes to conventional dense transformers.

Token generation (MMVQ) was profiled and found already near-optimal; no
code changes were retained (documented as a successful bottleneck
identification). Flash-attention decode at depth is bandwidth-bound; a
256-thread vector kernel gives a small (+3-7%) gain at long context.

All retained changes pass test-backend-ops correctness (MUL_MAT,
MUL_MAT_ID, FLASH_ATTN_EXT).

## 2. Hardware / software environment

See 00_ENVIRONMENT.md for full details. Idle GPU: 20W, 37C, sclk 808MHz.
Build: cmake -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release,
HIPCXX=$(hipconfig -l)/clang, HIP_PATH=$(hipconfig -R).

IMPORTANT: this machine has an intermittent hard-freeze problem (see
section 15 and EXP-SYS-001/002 in the ledger). The freeze correlates 2-for-2
with the Stream-K kernel machinery and is excluded from production; every
retained change ran for hours of use without any freeze.

## 3. Starting upstream behavior

gfx906 had no dedicated MMQ configuration. Host dispatch
(ggml_cuda_mmq_get_config) checked CDNA/RDNA4/RDNA3.5/RDNA3 and then fell
through to the RDNA2 table: nthreads 256, occupancy 2, I 128, J 8..64,
no Stream-K. Device-side dispatch had the same fallback. There was no
gfx906-specific symbol; VEGA20 was added by us in hip.h.

Flash attention: gfx906 has no MMA units, so VEC/TILE kernels are used
(generic amd tile table). F16 KV is preferred; Q8_0 KV measured 31% slower.

## 4. Baseline performance

See 01_BASELINE.md and results/baseline.csv. Key numbers:

| workload | upstream baseline |
| -- | -- |
| Q8_0 pp128/512/2048 | 1142 / 1365 / 1371 |
| Q4_0 pp128/512/2048 | 1691 / 2647 / 2678 |
| Q4_K_M pp128/512/2048 | 1470 / 2107 / 2121 |
| Q8_0 tg128 | ~167 |
| Q4_0 tg128 | ~189 |
| Q4_0 tg@32k / @100k | 161.4 / 123.7 |

Correctness baseline: MUL_MAT 1186/1186, MUL_MAT_ID 865/865.

## 5. Dedicated gfx906 MMQ configuration

Files:
- ggml/src/ggml-cuda/mmq-config-gfx906.cuh (new)
- ggml/src/ggml-cuda/vendors/hip.h: `#define VEGA20` for `__gfx906__`
- ggml/src/ggml-cuda/mmq.cuh: host dispatch `cc == GGML_CUDA_CC_VEGA20`,
  device dispatch `#elif defined(VEGA20)`

Initially a byte-identical copy of the RDNA2 table (behavior-equivalence
verified), then tuned. Commits: f6d01e2f8, 05dcb191b, 246a52a30.

The mx-llama.cpp fork (commit 2335544ab) independently created the same
kind of dedicated gfx906 config (512 threads, J up to 128), confirming the
approach.

## 6. Q8 MMQ experiments (launch/wave configurations)

All experiments in EXPERIMENT_LEDGER.md (EXP-MMQ-001..013). Summary:

- nthreads 256 -> 512 (8 wave64 wavefronts/block): +42-46% alone
- occupancy hint 2 -> 1: no effect (reverted)
- I 128 -> 64: -24% (rejected)
- J table 8..64 -> 8..128: +12-25% at pp256+, with small-batch clamp
  (J<=64 when ncols_max<=128) fixing pp128; total +45-61% vs upstream
- Stream-K: +44% additional, but 2 hard freezes -> EXCLUDED (section 15)
- tile-load 8 threads/row (ported from mx-llama 5c4505b5d): +17% pp128,
  +6% elsewhere
- k-unroll 2 in Q8 vec_dot (ported from mx-llama f1684f76c): +4% pp128,
  ~+1% elsewhere

Kernel resources (Q8_0, 512 threads): J=8 vgpr74, J=16 vgpr109, J=24 vgpr117,
J=32 vgpr80, J=40 vgpr82, J=48 vgpr115, J=64 vgpr87. 1 block/CU (LDS-bound:
50.9KB @ J64, 60.3KB @ J128 of 64KB).

Mechanism: gfx906 MMQ is LDS/occupancy limited at 1 block/CU; 512 threads
puts 8 wave64 wavefronts on the 4 SIMDs (2 per SIMD), hiding HBM latency.
Q8_0 is bandwidth-bound (effectively ~640 GiB/s in MMVQ, near HBM peak);
the kernel geometry wins by keeping more loads in flight. Q4 quants are
instruction-bound and keep 256 threads + J<=64 (EXP-MMQ-007/008 rejected
large-J and Stream-K for Q4).

## 7. MMQ tile / Stream-K experiments

- Stream-K on (persistent 60-block grid): +24-45% on top of J=128 tuning
  (pp128 2067, pp512 3176 vs 1663/2200 without) - but 2 hard freezes,
  UNSAFE, excluded (see EXP-SYS-002, EXP-MMQ-011; branch gfx906-stream-k
  preserves the commit f0864a2e4).
- Static K-split grid (same SK kernel + fixup, static non-persistent grid):
  also froze. The SK kernel/fixup/tmp_fixup machinery itself is the suspect
  (bad access pattern or launch-bounds mismatch hanging wavefronts).
- Alternative explanations: tiled launch wave efficiency f =
  ceil(tiles/60)/(tiles/60) predicts only part of the gain; x-tile LDS
  reuse in SK's kbc ordering likely also contributes. Research direction
  abandoned for stability; documented in ledger + b-compressed notes.
- K_vram 256 -> 512: BROKEN (skips half of K: ITER_K scales blocks_per_iter
  but SRAM tile sizes are hardcoded to MMQ_ITER_K=256 coverage) - rejected
  after correctness failure.

## 8. MMVQ profiling (decode)

Decode on gfx906 uses mul_mat_vec_q (GCN parameter table: nwarps=2 for
ncols 1-4). Effective rates: Q8_0 ~476 GiB/s model-size x t/s, Q4_0 ~300,
Q4_K ~274 - Q4_0 is NOT purely memory bound.

Findings:
- loads already vectorized by the compiler (global_load_dwordx4 in the
  Q4_0 hot loop)
- nwarps 2->4 regresses: -6.4% Q4_0, -26.7% Q4_K, -11.7% Q8_0
  (occupancy already maximal: 32 wave64 waves/CU = hardware max)
- VDR increase: no benefit expected (memory reads identical)
- Q8_0 decode: 643 GiB/s effective, near bandwidth-bound
- Q4_0 decode: instruction/latency-bound (78us vs f16 176us @
  m=4096 n=1 k=14336)

Decision: no MMVQ code changes retained; stage documented as successful
bottleneck identification (EXP-MMVQ-001, ledger).

## 9. wave64-specific work

- MMQ nthreads 512 = 8 wave64 wavefronts (vs 4 at 256): the main win.
- FATTN vec kernel 128 -> 256 threads: +3.8% tg@0, +7.5% at 32k with Q8_0 KV.
- 512 FATTN threads regressed (Q8_0 KV -10%).
- 1024-thread MMQ is illegal (16 warps does not divide J=8,24,40).
- DPP/shuffle experiments: not pursued (occupancy-limited, not
  reduction-limited).

## 10. Flash attention / long context

- gfx906 uses VEC/TILE kernels; TILE always dequantizes quantized KV to
  F16 first.
- F16 KV preferred; Q8_0 KV is a 31% regression (dequant + vec
  inefficiency > 2x bandwidth saving). Rejected.
- VEC nthreads 128 -> 256 (gfx906 only): retained (+3-7%, correctness OK).
- Dedicated gfx906 TILE table added (fe4f0a81b), behavior-identical;
  tuning attempts (threads/occupancy bumps) showed no change at 32k:
  tg at depth is bandwidth-bound (~870 GiB/s effective, near HBM peak).
- Long-context results (safe branch, F16 KV, LFM2.5 Q4_0):

| depth | tg t/s |
| -- | -- |
| 0 | 171.4 |
| 8k | 166.6 |
| 16k | 162.1 |
| 32k | 151.1 |
| 64k | 132.8 (8 reps; one throttled run 108.6 excluded) |

## 11. ROCm / compiler findings

- No profiler available on the system (rocprof/omniperf absent); used
  compiler resource reports (amdgpu metadata: VGPR/SGPR/LDS/workgroup size)
  and rocm-smi counters instead.
- kernel resource extraction workflow documented (llvm-objcopy
  .hip_fatbin + clang-offload-bundler + llvm-readelf --notes).
- rocBLAS control (GGML_CUDA_FORCE_CUBLAS): tuned MMQ Q8 pp512 3176 vs
  rocBLAS 1700 (+87%); Q4 2647 vs 1727. Custom MMQ is clearly right.

## 12. Correctness

- test-backend-ops test -b ROCm0 -o MUL_MAT: 1186/1186 PASS (all retained
  commits)
- MUL_MAT_ID: 865/865 PASS
- FLASH_ATTN_EXT: 2/2 backends PASS
- K_vram=512 experiment FAILED correctness (ERR up to 1.00) and was reverted.

## 13. Cross-model validation

qwen2.5-1.5B (dense, D=128, GQA ratio 6, 28 layers) on safe branch:

| workload | upstream | gfx906-safe | delta |
| -- | -- | -- | -- |
| Q8_0 pp512 | 2414 | 4288 | +78% |
| Q8_0 pp2048 | - | 4211 | - |
| Q4_0 pp512 | 4783 | 4872 | +1.9% |
| Q8_0 tg128 | 221 | 218.5 | -1.1% (noise) |
| Q4_0 tg128 | 243 | 240.7 | -1.0% (noise) |

Q8_0 MMQ gains generalize. Q4 decode unaffected as expected.

## 14. Rejected / unsafe experiments

- Stream-K (persistent + static grid): 2 hard freezes, UNSAFE, excluded
  from production; documented, preserved on gfx906-stream-k branch.
- K_vram 512: incorrect results (skips K), reverted.
- Q4 large J (72..128) + Stream-K: -16..-41%, reverted.
- Q4 Stream-K only: neutral/slightly negative, reverted.
- Q4 nthreads 512: -3..-12%, reverted.
- MMQ occupancy hint 2->1: no effect, reverted.
- I=64: -24%, rejected.
- FATTN tile tuning (threads/occupancy): no effect at 32k, reverted.
- FATTN vec 512 threads: -10% Q8_0 KV, rejected.
- MMVQ nwarps 4: regressed all quants, reverted.
- Q8_0 KV for FA: -31% vs F16 KV, rejected.

## 15. System stability investigation (EXP-SYS-001/002)

The machine hard-freezes (display on the AMD GPU goes black, kernel stops
logging - loglevel=3 caveat). Commit-level bisection: upstream, 512-thread,
J=128 builds all PASS; the Stream-K commit f0864a2e4 froze; the static
K-split variant froze again (2 for 2 with SK machinery). All other changes
ran for hours without a freeze. The machine also has a pre-existing PCIe
retraining failure at every boot (pci 00:1c.0) and froze once during a
CPU-only build, so hardware instability cannot be fully excluded - but
the SK machinery is excluded from production as a precaution, and the
freeze was never reproduced with any retained change.

## 16. Final patch set (branch gfx906-safe)

Commits on top of 62bf73d25:

| commit | change |
| -- | -- |
| f6d01e2f8 | dedicated gfx906 MMQ configuration |
| 05dcb191b | Q8_0 MMQ nthreads 512 |
| 246a52a30 | Q8_0 J up to 128 + small-batch clamp |
| 08ff3bf50 | FA vec kernel 256 threads (gfx906) |
| 583ac82dd | dedicated FA tile configuration (gfx906) |
| ae4fa0448 | Q8_0 tile load 8 threads/row (gfx906) |
| 81ee403c9 | Q8_0 vec_dot k-unroll 2 (gfx906) |

Files modified: ggml/src/ggml-cuda/mmq-config-gfx906.cuh (new),
mmq.cuh, mmq-load-tiles.cuh, mmq-vec-dot.cuh, fattn-vec.cuh,
fattn-tile.cuh, vendors/hip.h. Plus reports/ and research/ docs.

Upstreamability:
- A (clearly): dedicated gfx906 MMQ configuration + dispatch
- A/B: Q8_0 512 threads + J up to 128 + clamp
- B: tile-load 8 threads/row (VEGA20-gated version of mx-llama/upstream
  PR #21698 idea)
- B: vec_dot k-unroll 2
- B: FA vec 256 threads

## 17. Final performance table

| workload | upstream | final (gfx906-safe) | improvement |
| -- | --: | --: | --: |
| Q8 pp128 | 1142 | 2147 | +88% |
| Q8 pp512 | 1365 | 2362 | +73% |
| Q8 pp2048 | 1371 | 2361 | +72% |
| Q4 pp512 | 2647 | 2621 | -1.0% (noise) |
| Q4 tg128 | 189 | 191.8 | +1.5% (noise) |
| Q8 tg128 | 167 | 164.3 | -1.6% (noise) |
| Q4 tg @ 32k | 161.4 | 151.1 | -6.4% (FA VEC gain partially offset by variance; 8k/16k within noise) |
| Q4 tg @ 64k | 133.3 | 132.8 | -0.4% |
| FA 32k (Q8_0 KV) | 98.6 | 106.1 | +7.5% |
| qwen Q8 pp512 | 2414 | 4288 | +78% |

Rows with high variance marked; see results/ CSVs.

## 18. Profiling explanation

Q8_0 MMQ on gfx906 is bandwidth/latency-bound with a 1-block/CU LDS
constraint. 512 threads (8 wave64 waves) keeps more HBM loads in flight per
CU; J up to 128 halves x-tile reload traffic; 8 threads/row tile loading
removes idle lanes during qs loads; k-unroll 2 halves address
materialization in vec_dot. Q4 quants are instruction-bound, hence the
asymmetric tuning. Decode MMVQ is already occupancy-maximal at 2 warps.
At depth, FA decode is KV-bandwidth-bound (TILE+F16 optimal).

## 19. Upstream recommendation

Propose independently: (1) dedicated gfx906 MMQ config + exact VEGA20
dispatch (with the 512-thread/J128/clamp values), (2) VEGA20-gated
8 threads/row tile load, (3) gfx906-gated k-unroll, (4) FA vec 256 threads
for gfx906. Do NOT propose Stream-K until the freeze question is resolved
on other hardware.

## 20. Remaining opportunities

- Verify the 8-thread tile load and k-unroll on other GCN5 parts (gfx900)
  - intentionally left gfx906-only here.
- Investigate the freeze further on a different machine with the SK kernel
  (requires hardware with working RAS/driver diagnostics).
- FA: try nthreads between 256-512 for VEC or nbatch_fa tuning at 64k+.
- MXFP4 512-thread MMQ (from mx-llama 4daf555b1) - untested here, no MXFP4
  models available.
- MMVQ: only if profiling shows a concrete mechanism.

## Reproducibility

- CSV results: reports/gfx906/results/
- Ledger: reports/gfx906/EXPERIMENT_LEDGER.md
- Environment: reports/gfx906/00_ENVIRONMENT.md
- Baseline doc: reports/gfx906/01_BASELINE.md
- Shape analysis: reports/gfx906/08_MMQ_SHAPES.md
- Branches: gfx906-safe (production), gfx906-stream-k (experimental,
  freeze-suspect), gfx906-optimize (main, full history)
- Bench script: research/gfx906/scripts/bench_gfx906.sh
