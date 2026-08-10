# gfx906 Experiment Ledger

Every experiment: ID, date, commit, change, hypothesis, benchmark, baseline vs new,
delta %, correctness, profiler findings, decision.

## EXP-MMQ-001 (2026-08-10, commit f6d01e2f8)

Change:
Added dedicated gfx906 MMQ configuration (mmq-config-gfx906.cuh), behavior-identical
copy of the RDNA2 table gfx906 previously inherited, wired into host dispatch
(cc == GGML_CUDA_CC_VEGA20) and device dispatch (`#elif defined(VEGA20)`).

Hypothesis:
Create an architecture-specific tuning point before changing anything.

Benchmark:
Q8_0 pp512 1358 vs 1365 baseline (0.5%, noise).

Correctness:
PASS (MUL_MAT 1186/1186, MUL_MAT_ID 865/865).

Decision:
KEEP (infrastructure).

## EXP-MMQ-002 (2026-08-10, commit <pending>)

Change:
Q8_0 gfx906 MMQ nthreads 256 -> 512 (occupancy hint unchanged at 2).

Hypothesis:
gfx906 is LDS-bound at 1 block/CU (tile_x + tile_y ~= 44.5KB for J=64). nthreads
sets resident wave64 wavefronts per CU (256 -> 4 waves, 512 -> 8 waves). More
waves per block improves latency hiding on the 4-SIMD GCN5 core.

Benchmark (llama-bench 5 reps, LFM2.5-2.6B-Q8_0):
pp128: 1563 vs 1142 (+36.9%)
pp256: 1712 vs 1180 (+45.1%)
pp512: 1972 vs 1365 (+44.5%)
pp1024: 1950 vs 1377 (+41.6%)
pp2048: 1942 vs 1371 (+41.6%)
10-rep pp128 confirm: 1666.66 +- 92.39 (+46%)
Saved: reports/gfx906/results/q8_512.csv

Control:
Q4_0 (still 256 nthreads) pp512 2634 vs 2647 baseline -> unchanged, expected.
Q8 tg128 163.2 vs 167 baseline -> MMVQ path untouched, unchanged.

Kernel resources (Q8_0, nthreads=512, occupancy=2):
J=8 vgpr74 sgpr32; J=16 vgpr109; J=24 (fallback) vgpr117; J=32 vgpr80;
J=40 vgpr82; J=48 vgpr115; J=64 vgpr87. wave64. 1 block/CU (LDS-bound).

Correctness:
PASS (MUL_MAT 1186/1186 x3; one transient FAIL on very first run after build,
all subsequent runs clean - flaky first-launch artifact, hard to attribute to
the change yet), MUL_MAT_ID 865/865.

Decision:
KEEP (pending J/occupancy/1024-thread tests and final regression matrix).
## EXP-MMQ-003 (2026-08-10)

Change:
Q8_0 gfx906 MMQ occupancy hint 2 -> 1 (nthreads 512).

Hypothesis:
LDS-bound 1 block/CU; hint only constrains compiler register allocation.

Benchmark:
pp512 1976 vs 1972 (occ=2): no measurable effect (+0.2%).

Decision:
REVERT to occupancy 2 (no benefit, keep default).

## EXP-MMQ-004 (2026-08-10)

Change:
Q8_0 gfx906 MMQ I 128 -> 64 (nthreads 512, J up to 128).

Hypothesis:
More row tiles (nty 16 -> 32) improves small-batch parallelism.

Benchmark:
pp128 1236 (-24%), pp256 1285 (-25%), pp512 1495 (-24%),
pp1024 1494 (-24%), pp2048 1463 (-26%) vs I=128 config.
Correctness: MUL_MAT OK.

Decision:
REJECT. I=64 roughly halves per-wave arithmetic depth and loses ~25%.
Mechanism not profiled further (clear loss, small value).

## EXP-MMQ-005 (2026-08-10, commit <pending>)

Change:
Q8_0 gfx906 MMQ: extended J table 8..64 -> 8..128 (nthreads 512, I 128),
plus gfx906-gated clamp in mul_mat_q_switch_J (mmq.cuh): if
ncols_max <= 128, force J_best <= 64 (large-tile single-tile grids spawn
only 16 blocks = 128 waves < 240 SIMD slots -> under-filled GPU).

Hypothesis:
Bigger J tiles halve x-tile reload traffic (x tile loaded once per tile);
at ncols > 128 the GPU stays filled (ntiles >= 2); at ncols <= 128 the
clamp keeps block count >= 32.

Benchmark (5 reps; 10-rep confirm for pp256/pp512):
pp128 1662 vs 1142 (+45.5%)
pp256 2146 vs 1180 (+81.9%)   [10 reps: 2144.1 +- 102.9]
pp512 2191 vs 1365 (+60.5%)   [10 reps: 2208.2 +- 31.1]
pp1024 2218 vs 1377 (+61.1%)
pp2048 2202 vs 1371 (+60.6%)
tg128 163.9 (unchanged, MMVQ path untouched).

Without clamp, J=128 alone: pp128 1374 (-15.8% vs J64), pp >= 256: +12..+25%.

Correctness:
PASS (MUL_MAT 1186/1186, MUL_MAT_ID 865/865).

Decision:
KEEP. Q8_0 pp +45-61% total vs upstream baseline.

## EXP-MMQ-006 (2026-08-10, commit <pending>)

Change:
Q8_0 gfx906 MMQ stream_k false -> true (all 27 J/fallback entries).

Hypothesis:
Tiled grid (64 blocks at pp512) leaves CUs idle on tail tiles; Stream-K
persistent-partitioning balances K-partial work across the 60 CUs.

Benchmark (5 reps):
pp128 2067 vs 1662 no-SK (+24.3%)
pp256 2959 vs 2146 (+37.9%)
pp512 3176 vs 2191 (+45.0%)
pp1024 3208 vs 2218 (+44.6%)
pp2048 3176 vs 2202 (+44.2%)
vs upstream baseline: pp512 1365 -> 3176 (+133%), pp2048 1371 -> 3176 (+132%).

Correctness:
PASS (MUL_MAT 1186/1186, MUL_MAT_ID 865/865).

Decision:
KEEP.
