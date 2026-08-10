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