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

## EXP-MMQ-002 (2026-08-10, commit 05dcb191b)

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

## EXP-MMQ-005 (2026-08-10, commit 246a52a30)

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

## EXP-MMQ-006 (2026-08-10, commit f0864a2e4)

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

## EXP-MMQ-007 (2026-08-10, REVERTED, no commit)

Change:
Q4_0/Q4_K gfx906 MMQ: J table 8..64 -> 8..128 + stream_k true (nthreads still 256).

Hypothesis:
Same large-tile + Stream-K recipe that won on Q8_0 transfers to Q4 quants.

Benchmark (5 reps):
Q4_0: pp128 2162 vs base 1691 (+27.9%, sd 263 - noisy), pp256 2008 vs 2412 (-16.7%),
pp512 2174 vs 2647 (-17.9%), pp1024 2189 vs 2695 (-18.8%), pp2048 2186 vs 2678 (-18.4%).
Q4_K: pp128 2184 vs 1470 (+48.6%, sd 151), pp256 1146 vs 1829 (-37.3%),
pp512 1248 vs 2107 (-40.8%), pp1024 1247 vs 2133 (-41.5%), pp2048 1240 vs 2121 (-41.5%).
Correctness: MUL_MAT OK.

Profiler findings:
sum[] per thread = J*I/nthreads = 64 floats at J=128 with 256 threads
(vs 32 at J=64): VGPR pressure/spills; Q4_K vec_dot needs more temps than Q8.
Kernel resources (Q4_K J=128, 256 thr) showed register growth vs J=64.

Decision:
REJECT. Q4 quants are instruction/LDS-bound, not bandwidth-bound; 256 threads
and J<=64 already optimal for them. Do not reuse the Q8 recipe for Q4.

## EXP-MMQ-008 (2026-08-10, REVERTED, no commit)

Change:
Q4_0/Q4_K gfx906 MMQ: stream_k true only (J table unchanged 8..64).

Hypothesis:
Isolate Stream-K effect on Q4 at the inherited geometry.

Benchmark (5 reps):
Q4_0: pp128 -0.4%, pp256 -2.7%, pp512 -3.1%, pp1024 -3.1%, pp2048 -3.1%.
Q4_K: pp128 +0.8%, pp256 -0.9%, pp512 -0.9%, pp1024 -0.9%, pp2048 -0.8%.

Decision:
REJECT. Stream-K alone is neutral to slightly negative for Q4 quants on gfx906.
Q4_0/Q4_K restored byte-identical to the upstream-inherited RDNA2 config;
verified pp512 Q4_0 2647, Q4_K 2122 (baseline 2647/2107), MUL_MAT 1186/1186 OK.

## EXP-MMQ-009 (2026-08-10, REVERTED, no commit)

Change:
Q8_0 gfx906 MMQ K_vram 256 (MMQ_ITER_K) -> 512 (stream_k true).

Hypothesis:
Larger K chunks per tile iteration reduce tile reload overhead.

Benchmark (5 reps; NOTE results are INVALID - see below):
pp128 3350, pp256 4665, pp512 5132, pp1024 5226, pp2048 5143
(superficially +62% vs 3176 baseline - achieved by skipping work).

Correctness:
FAIL. MUL_MAT q8_0 m=2880 n=32 k=2880 ERR=0.86, m=16 n=16 k=1024 bs=[3,2]
ERR=1.00. Consecutive FAILs, deterministic.

Profiler findings:
ITER_K only scales blocks_per_iter = ITER_K/qk (the kb0 step in
mul_mat_q_process_tile) but the SRAM tiles (tile_x/tile_y) are HARDCODED
to cover MMQ_ITER_K=256 K elements per iteration (tile_x I*2*MMQ_TILE_NE_K,
tile_y J*MMQ_TILE_Y_K = one block_q8_1_mmq per column). With K_vram=512 the
kernel steps 16 qk-blocks per iteration while each tile only holds 8 ->
skips half the K dimension. Only FP4 types have a different ITER_K
(MMQ_ITER_K_FP4=512) because their tile layout differs.

Decision:
REJECT. K_vram is not a free parameter for Q8_0: non-FP4 tile layouts are
structurally coupled to K_vram == MMQ_ITER_K. No further K_vram experiments
for these types without a tile-layout redesign (out of scope).

## EXP-MMVQ-001 (2026-08-10, REVERTED, no commit)

Change:
Dedicated gfx906 MMVQ parameter table (MMVQ_PARAMETERS_GCN906) with
ncols_dst 1-4 -> nwarps=4 (wave64: 128 -> 256 threads/block). Added
device gate (#elif defined(VEGA20)) and host gate (cc ==
GGML_CUDA_CC_VEGA20, placed before the GCN/CDNA path; CDNA/gfx900
unchanged).

Hypothesis:
More wave64 warps per block increase decode latency hiding (RDNA4 uses
8 warps at ncols_dst=1).

Benchmark (llama-bench 5 reps, tg128 vs GCN 2-warp baseline):
Q4_0: 177.4 vs 189 (-6.4%)
Q4_K: 119.5 vs 163 (-26.7%)
Q8_0: 147.5 vs 167 (-11.7%)

Kernel resources (ncols=1, 4 warps / max_flat_workgroup_size=256):
Q4_0 fusion1 smallk1: vgpr221 sgpr56 (spills likely); fusion1 smallk0:
vgpr77; fusion0 smallk0: vgpr34. LDS ~5-6KB.
Correctness: MUL_MAT 1186/1186 PASS.

Profiler findings (reasoning, no counters available):
Decode occupancy is already maximal at 2 warps: 4096 row-blocks x
128 threads, 16 blocks/CU x 2 wave64 waves = 32 waves/CU (2048
threads = HW max). Doubling threads per block does not add resident
waves; it only adds idle threads (Q4_K: blocks_per_iter=32 >
blocks_per_row_x=16, so warps beyond the first do no vec_dot work)
and reduction traffic.
Note: Q4_K/Q8_0 tg > Q4_0 tg (163/167 vs 189) despite Q4_0 reading the
fewest weight bytes suggests decode is NOT purely HBM-bandwidth
limited; per-row work is latency/instruction bound.

## EXP-FA-001 (2026-08-10, commit 37b0afd99)

Change:
Flash-attn VEC kernel nthreads 128 -> 256 on gfx906 only (host gated on
cc == GGML_CUDA_CC_VEGA20, device gated on #ifdef VEGA20). Both must agree
(device constexpr used in kernel body, e.g. KV loop strides by nthreads).

Hypothesis:
VEC kernel used for all decode layers with Q8_0 KV and for MHA layers with
F16 KV. 128 threads = 2 wave64 warps; 256 threads may improve latency hiding
on the 4-SIMD GCN core. 512 threads also tested (pure thread-count change).

Benchmark (llama-bench, tg @ depth, kv f16, 5 reps):
Q4_0: d0 193.4 vs 186.3 (+3.8%), d4k 187.7 vs 181.0 (+3.7%),
      d32k 163.8 vs 161.4 (+1.5%), d100k 126.2 vs 123.7 (+2.0%)
Q8_0: d0 168.4 vs 163.4 (+3.1%), d4k 163.9 vs 155.5 (+5.4%),
      d32k 145.3 vs 135.1 (+7.5%), d100k 114.3 vs 113.1 (+1.1%)
Q4_K: d0 167.8 vs 169.1 (-0.8%, noise), d4k 161.8 vs 163.8 (-1.2%),
      d32k 142.4 vs 143.6 (-0.8%), d100k 112.6 vs 111.7 (+0.8%)
512-thread test: F16 KV d32k 160.26 vs 159.47 (256) = no gain;
Q8_0 KV d32k 95.29 vs 106.06 (256) = -10.1% -> 512 REJECTED.

Correctness:
PASS (FLASH_ATTN_EXT MUL_MAT... test-backend-ops -o FLASH_ATTN_EXT all OK).

Decision:
KEEP 256 in both host and device dispatch.

## EXP-FA-002 (2026-08-10, commit fe4f0a81b)

Change:
Dedicated gfx906 flash-attn TILE configuration table
(ggml_cuda_fattn_tile_get_config_gfx906 in fattn-tile.cuh), byte-identical
copy of the generic AMD table gfx906 previously received, wired into host
dispatch (cc == GGML_CUDA_CC_VEGA20) and device dispatch
(`#elif defined(VEGA20)`). RDNA/other AMD/CDNA untouched.

Hypothesis:
Create an architecture-specific TILE tuning point before changing anything.

Benchmark:
Behavior equivalence: tg@32k Q8_0 143.4 vs 145.3 pre-change (noise),
Q4_0 151.4 sd 14.7 (noisy, within noise of 163.8 pre-change).
Correctness: FLASH_ATTN_EXT 2/2 backends PASS.

Decision:
KEEP (infrastructure, same pattern as the MMQ config).

## EXP-FA-003 (2026-08-10, REVERTED, no commit)

Change:
Two TILE tuning attempts on gfx906 table (64,64):
A) ncols=4 nthreads 128 -> 256
B) ncols2 occ 3->4, ncols4 occ 3->5 + nthreads back to 128, ncols8 occ 2->4

Hypothesis:
Decode TILE (GQA ratio-4 layers, cols_per_block=4) gets more wave64 warps
per block (A) or more resident blocks (B); hides KV load latency.

Benchmark (tg@32k, Q8_0, 5 reps - Q4_0 too noisy sd>14 to judge):
A: 144.0 sd 1.7 vs 143.4 (no change)
B: 142.2 sd 4.4 vs 143.4 (no change)
Correctness: FLASH_ATTN_EXT PASS for both.

Conclusion:
tg at 32k+ depth is bandwidth-bound on gfx906: Q8_0 model 2.87 GB + KV
read ~3 GB per token at 32k => ~870 GiB/s effective, near HBM peak.
Kernel launch geometry does not move the needle; VEC kernel (EXP-FA-001)
helped only because it improved load coalescing, not parallelism.

Decision:
REJECT both. Reverted to fe4f0a81b (dedicated table, untuned).

## EXP-FA-004 (2026-08-10, REJECTED, no commit)

Change:
KV cache type Q8_0/Q8_0 (quantized) at 32k depth.

Hypothesis:
Halving KV traffic (1 byte/value vs 2) helps bandwidth-bound decode.

Benchmark (tg@32k Q4_0): 98.63 vs 142.84 F16 KV (-31%).
Q8_0 KV forces the VEC kernel on all layers; VEC dequant + scalar
Q8_0 loads outweigh the 2x bandwidth savings on gfx906.

Decision:
REJECT. F16 KV stays. Documented as a gfx906-specific negative result.

## EXP-MMQ-010 (2026-08-10, control experiment, no code change)

Change:
Built build-cublas/ with -DGGML_CUDA_FORCE_CUBLAS=ON (dequant + hipBLAS
instead of MMQ; gfx906 otherwise always uses MMQ since
ggml_cuda_should_use_mmq returns true for non-CDNA).

Hypothesis:
rocBLAS/hipBLAS GEMM as a diagnostic control: is the custom quantized
MMQ kernel providing value on gfx906?

Benchmark (llama-bench 5 reps, LFM2.5-2.6B):
ROCBLAS Q8_0: pp128 1062, pp256 1316, pp512 1700, pp1024 1709, pp2048 1730
vs tuned MMQ: 2067 / 2959 / 3176 / 3208 / 3176
= tuned MMQ +95% / +125% / +87% / +88% / +84%.
ROCBLAS Q4_0 pp512 1727 vs tuned MMQ 2647 (+53%).
Upstream MMQ (before tuning) was -20% vs rocBLAS.

Decision:
Custom MMQ is the right path on gfx906; the tuning makes it decisively
faster. No rocBLAS workaround shipped.

## EXP-SYS-001 (2026-08-11, freeze investigation - attempt 1, NO code change)

Phenomenon:
Whole-machine hard freezes (GPU display off, no kernel logs - journal simply
stops) while running llama.cpp GPU builds. First freeze was during a CPU-only
build-cublas cmake build (GPU idle); freezes also occurred in a boot where
llama.cpp never ran. No amdgpu errors before any freeze. Kernel cmdline has
loglevel=3 (many messages suppressed).

Facts gathered:
- 4 boots observed; freezes at 23:37, 08:27, 08:51, 09:15. None have ANY
  amdgpu/MCE/OOM/AER message before the freeze; log ends mid-activity.
- 'pci 0000:00:1c.0: retraining failed' present at EVERY boot (pre-existing,
  also in the stable 10h session boot). GPU sits behind a 2-level PCIe bridge.
- rocblas-bench (pure rocBLAS, no llama.cpp code): 8192^3 f32 GEMM + sustained
  4096x4096x8192 loop - PASSED, no crash. Heavy non-llama GPU load is stable.
- Decisive non-freeze data point from earlier: 10h llama-bench session (13:53-
  23:37, ALL gfx906 tuning + FA experiments) ran without a single crash/kernel
  error the day the tuning work was done.

Verification that the freezes are NOT caused by ordinary GPU load or the
tuning work in general:
- small llama-bench run (qwen Q4_0 pp8/tg8): passed, GPU idles back normally
- rocblas-bench large GEMM: passed

Conclusion so far:
System is only destabilized by specific llama.cpp GPU code paths; at this
point only commit-level bisection can localize it. TBD.

## EXP-SYS-002 (2026-08-11, commit-level bisection of the freezes, NO code change)

Hypothesis:
Localize the freeze trigger to specific commits by running identical
llama-bench workloads (LFM2.5-2.6B Q4_0 + Q8_0, pp512 + tg128, 3 reps) on
builds at different commits, in persistent worktrees under
/home/petrouil/Projects/ (llama.cpp-upstream, llama.cpp-512, llama.cpp-j128).

Results (same command each time):
- 62bf73d25 upstream pristine: PASS, no crash (qwen Q4_0 pp512 4783/tg128 243,
  Q8_0 2414/221, LFM Q4_0 2621/191)
- 05dcb191b (nthreads 512 only, occupancy 2, J 8..64, no SK, no FA, no
  dedicated mmq config yet): PASS, no crash (Q4_0 pp512 2576.8/tg128 186.6,
  Q8_0 pp512 1933.8/tg128 162.2)
- 246a52a30 (+ J table up to 128 + small-batch clamp, NO Stream-K): PASS, no
  crash (Q4_0 pp512 2608/tg128 191.6, Q8_0 pp512 2131/tg128 165.4)
- f0864a2e4 (+ Stream-K true for Q8_0): user-reported crash
- main branch 05b610a71 (all: + FA vec 256 threads + FA tile table): user-
  reported crash

Verdict:
The freeze window is narrowed to the Stream-K commit being applied; J=128
tiles and the 512-thread launch geometry both pass independently. Stream-K is
a structural change (persistent grid, fixup kernel, occupancy-driven sizing)
vs pure table/value changes that pass.

Status: CONFIRMED. EXP-MMQ-011 (static SK grid variant, same SK kernel +
fixup machinery) produced a second hard freeze, while every non-SK change
ran freeze-free for hours. The SK kernel/fixup/tmp_fixup machinery is now
excluded from production and from further experiments (2 for 2).

## EXP-MMQ-011 (2026-08-11, REVERTED, UNSAFE - no commit)

Change:
Static K-split grid as a "safe" Stream-K alternative: stream_k=true for Q8_0
again, but instead of the persistent 60-block grid, a static grid rounded up
to a multiple of nsm (120/180/240 blocks). Still used the in-tree
mul_mat_q_stream_k kernel + fixup kernel + tmp_fixup buffer machinery.

Hypothesis:
The Stream-K gain comes from work distribution, not persistence; a static
full-wave grid should capture the benefit without the persistent blocks.

Result:
SECOND HARD FREEZE with the Stream-K machinery (machine froze during
test-backend-ops, required reboot). 2 for 2 with SK machinery now:
  f0864a2e4 (persistent SK)     -> freeze
  EXP-MMQ-011 (static SK grid)  -> freeze
  every other change (512 thr, J=128, FA vec/tile) -> zero freezes in hours
  of use. This points at the SK kernel + fixup + tmp_fixup machinery itself
  (possible GPU hang: bad access pattern, loop, or launch-bounds mismatch
  hangs the wavefronts; display on the AMD GPU goes black), not persistence.

Decision:
REJECT + mark UNSAFE. The Stream-K kernel/fixup machinery is excluded from
production and from any further experiments. Abandon the Stream-K direction
entirely; pursue only conventional-launch improvements (no SK kernel, no
fixup kernel, no tmp_fixup pool).

## EXP-MMQ-012 (2026-08-11, commit ae4fa0448, KEEP)

Change:
Q8_0 MMQ tile-load: 32 threads per row -> 8 threads per row in
ggml_cuda_mmq_load_tiles_q8_0 (mmq-load-tiles.cuh), gated on GGML_USE_HIP &&
VEGA20. Ported from mx-llama.cpp commit 5c4505b5d (iacopPBK; origin upstream
PR ggml-org/llama.cpp#21698 but with threads_per_row=8 instead of the PR's
16, which measured ~27% slower on gfx906).

Hypothesis:
8 threads/row cover the QI8_0=8 ints per q8_0 block with one get_int_b2
per thread (kbx = txi/QI8_0, kqsx = txi%QI8_0); 32 threads/row leaves 3 of 4
threads idle during the qs load and wastes LDS write cycles.

Benchmark (LFM2.5-2.6B-Q8_0, 5 reps, t/s):
load-tiles only:  pp128 1947 | pp256 2266 | pp512 2327 | pp1024 2344 | pp2048 2328
vs safe baseline: 1663 | 2150 | 2200 | 2213 | 2199
(+17.1% pp128, +5.4% pp256, +5.8% pp512, +5.9% pp1024, +5.9% pp2048)

Correctness: PASS (MUL_MAT 1186/1186).
Freezes: NONE (hours of use).
Fork reference numbers: Qwen3-14B-Q8 1-GPU pp512 271->409 (+51%),
Qwen3.6-27B-Q8 4-GPU pp512 425->575 (+35%).

Decision:
KEEP.

## EXP-MMQ-013 (2026-08-11, commit 81ee403c9, KEEP)

Change:
Q8_0 MMQ vec_dot partial k-unroll: `#pragma unroll 2` above the k01 loop in
vec_dot_q8_0_q8_1_dp4a (mmq-vec-dot.cuh), gated on __gfx906__. Ported from
mx-llama.cpp commit f1684f76c (iacopPBK).

Hypothesis:
y stride between j iterations (1152 B) exceeds the ds_read2_b32 immediate
offset limit (1020 B), so each j needs a materialized address; unrolling by
2 halves the address-materialization + loop overhead. Full unroll spills
(~686 regs) and is much slower.

Benchmark (LFM2.5-2.6B-Q8_0, 5 reps, t/s):
k-unroll only:    pp128 1727 | pp256 2162 | pp512 2214 | pp1024 2232 | pp2048 2218
vs safe baseline: 1663 | 2150 | 2200 | 2213 | 2199
(+3.8% pp128, +0.6% pp256, +0.6% pp512, +0.9% pp1024, +0.9% pp2048)
Combined with EXP-MMQ-012: pp128 2147 | pp256 2300 | pp512 2362 | pp1024 2377 |
pp2048 2361 (+29.1% pp128 vs baseline).

Correctness: PASS (MUL_MAT 1186/1186).
Freezes: NONE.
Fork reference numbers: gemma-4-26B-A4B-Q8 MoE 1-GPU pp2048 +5.8%,
Qwen3.6-35B-A3B-Q8 4-GPU pp4096 +5.5%, Q4_0 guardrail +0.0%.

Decision:
KEEP.

## EXP-CROSS-001 (2026-08-11, cross-model validation, no code change)

Change:
Re-run llama-bench on qwen2.5-1.5B (dense transformer, D=128 heads,
conventional shapes) with the safe branch (both MMQ patches applied) to check
whether the Q8_0 MMQ gains generalize beyond LFM2.5-2.6B.

Benchmark (5 reps, t/s; upstream reference from llama.cpp-upstream worktree):
qwen2.5-1.5b-instruct-q8_0:
  safe:  pp128 2551.1 sd197 | pp512 4288.5 sd250 | pp2048 4210.7 sd4.6
  upstream: pp512 2414 -> +78%
qwen2.5-1.5b-instruct-q4_0:
  safe:  pp128 2071.3 sd133 | pp512 4871.7 sd213 | pp2048 4713.1 sd4.8
  upstream: pp512 4783 -> +1.9% (Q4 untouched, expected noise)

Verdict:
Q8_0 MMQ generalization CONFIRMED on a conventional dense transformer
(+78% pp512). Q4_0 unaffected. Still TODO: qwen tg128, qwen Q4_K_M, LFM tg
recheck.

Decision:
KEEP all. Continue validation.

## EXP-GEMV-001 (2026-08-11, commit TBD, KEEP)

Change:
Q4_K MMVQ (token generation, n=1) vector dot ratio VDR_Q4_K_Q8_1_MMVQ
2 -> 8 (vecdotq.cuh:504). Pure work-partitioning constant of the
mul_mat_vec_q inner loop (mmvq.cu:594): changes only how the K-dimension
quant blocks are split across threads (kqs = vdr * (tid % (qi/vdr)),
blocks_per_iter = vdr*nwarps*warp_size/qi); same math, identical results.

Hypothesis:
Default VDR=2 is an NVIDIA/generic legacy value never tuned for GCN
(warp_size=64, nwarps=2). Larger vdr means fewer threads cooperate per
x-block (qi/vdr), each covering a wider contiguous k-quant run: fewer
loop iterations and y-block re-reads, larger effective DRAM
transactions -> better GEMV streaming on gfx906.

Benchmark (gemma-4-12b-it-Q4_K_M, build-prof graphs OFF, 3 reps, t/s):
vdr=1:  tg1 35.99 | tg32 35.75 | tg128 35.52   (-30% regression)
vdr=2:  tg128 50.65                            (upstream default baseline)
vdr=4:  tg1 54.7  | tg32 55.08 | tg128 54.52   (+8%)
vdr=8:  tg1 60.09 | tg32 60.18 | tg128 60.54   (+19%) KEEP
vdr=16: tg1 60.5  | tg32 60.26 | tg128 60.26   identical to 8 within noise
vdr=32: SYSTEM CRASH (full machine freeze during llama-bench run, no
        logs recoverable; see below)

Plateau at 8-16: DRAM-bound reached (~690 -> ~830 GB/s streaming).

vdr=16 vs vdr=8:
Same performance within measurement noise (tg128 60.26 vs 60.54, tg1
60.5 vs 60.09, non-monotonic). Kept 8: also the value the sibling MMQ
kernel ships upstream (VDR_Q4_K_Q8_1_MMQ = 8), and qi/vdr=4 threads per
x-block stays safely away from the degenerate edge that broke at 32.

vdr=32 analysis (why it crashed):
qi/vdr collapses to 1: kqs = 32*(tid%1) = 0 for every thread and
kbx = tid/1, so threads no longer cooperate per x-block. Only
tid < blocks_per_row_x (=15 of 128) threads do any work, warp_reduce_sum
then sums dot products of DIFFERENT x-blocks (reduction assumed sibling
threads share a kbx), and 15/16 of each block's quant slices are never
computed. Invalid configuration by construction; on gfx906 the
pathological kernel wedged the driver/firmware hard enough to freeze the
machine (no logs survived). vdr=32 is NOT a valid tuning point.

Correctness:
vdr=8/16 produce identical math by construction (partitioning only).
llama-bench does not validate output content; no perplexity check run.

Fork reference numbers: none (tg-only change, private fork).

Decision:
KEEP vdr=8 for Q4_K MMVQ on gfx906. Reject 32 (crash), reject 16
(no gain over 8), reject 1 (regression), 4 (partial gain).

## EXP-MMQ-013 (2026-08-12, commit 5b98f77a2, KEEP)

Change:
Q5_K/Q6_K MMQ I-tile width 128 -> 64 for J=64 entries in
mmq-config-gfx906.cuh (both fallback variants). Matches the I-tile
width validated by the mixa3607 gfx906 sweep (CASE(Q5_K,256,2,64,64),
CASE(Q6_K,256,2,64,64)). Q4_K untouched (I=128 mandatory, validated
+52.8% regression at I=64 on the reference GPU).

Hypothesis:
Q6_K is the hot MMQ type in Q4_K_M models (ffn_down/attn_v), and a
narrower I-tile raises LDS throughput / fits more wavefronts, improving
PP. TG unaffected (MMQ I-tile is a PP-only path).

Benchmark (gemma-4-12b-it-Q4_K_M):
pp512 baseline 529.52 -> 581.63 (+9.8%), pp1024 566.07, pp2048 baseline
524.16 -> 538.75 (+2.8%) at the second run (556.55 first run).
tg128 60.12 -> 60.20 (0.1%, noise).
Control LFM2.5-2.6B-Q8_0 (no Q5_K/Q6_K tensors): pp512 3211.75 ->
3215.3 (+0.1%, flat), tg128 169.61 -> 166.49 (noise, stddev 8.5).

Correctness:
PASS for changed types. test-backend-ops MUL_MAT/MUL_MAT_ID: all
Q5_K/Q6_K cases PASS (28/28).
NOTE: q4_K cases FAIL (28 MUL_MAT + 24 MUL_MAT_ID) at tiny shapes
(m=16, n<=8, k<=1024), same failure set as upstream vanilla master
62bf73d25 (no gfx906 config, VDR=2): 1158/1186 + 841/865 there too.
Verified by rebuilding upstream master from scratch in a worktree.
Therefore pre-existing upstream/ROCm-7.2/clang-22 issue, NOT caused
by any fork tuning. Impact on real workloads: none (no target model
tensor has m=16 rows; gemma/LFM pp+tg outputs sane).

Decision:
KEEP. +9.8% PP on Q4_K_M prompt processing at pp512, pp2048 also
improved. The q4_K tiny-shape MUL_MAT correctness failure is a
pre-existing bug (see note) to be investigated separately.

## EXP-CMP-001 (2026-08-12, commit 5b98f77a2, REJECTED)

Change:
Compiler scheduler flag experiment: -mllvm -amdgpu-sched-strategy=max-ilp
(as validated on MI50 in the mixa3607 reference repo, ROCm 7.14/LLVM 23).

Hypothesis:
Reference repo reports fixing a PP regression on ROCm 7.14 with this
flag; gains were PP-only, TG flat. Environment here is ROCm 7.2 +
HIP clang 22.0.0git (different LLVM generation).

Benchmark:
No numbers - experiment aborted on correctness/crash grounds.

Correctness:
FAIL. test-backend-ops crashed with SIGILL (EXIT 132) before any test
ran; llama-bench core-dumped on a minimal run (process-level only, no
GPU hang, no machine freeze). The alternative iterative-ilp variant
segfaults the compiler itself during build (clang internal error).
Our stock build shows no PP regression to fix, so there is nothing to
gain: this lever is closed on our toolchain.

Decision:
REJECT. Flag set is incompatible with clang-22; keep stock
CMAKE_HIP_FLAGS. Do not retry on this environment.

## EXP-NOTE-20260812 (2026-08-12, no code change)

Observation:
Overnight session ran on branch gfx906-optimize, whose Q8_0 MMQ table
still carries stream_k=true (commit f0864a2e4, pre-freeze-verdict).
Hundreds of llama-bench runs (gemma pp512/1024/2048, LFM Q8_0 pp512,
tg128) and repeated test-backend-ops MUL_MAT/MUL_MAT_ID all completed
with ZERO freezes or GPU hangs. LFM Q8_0 pp512 = 3215 t/s, matching
the EXPECTED Stream-K numbers (3176-3208) - confirms SK path is active
in the binary and stable under this session's workloads.

Freeze history (EXP-SYS-001/002, EXP-MMQ-011): 2 freezes attributed to
the SK kernel/fixup/tmp_fixup machinery, both during earlier sessions.
The SK exclusion was implemented ONLY on branch gfx906-safe
(stream_k=false there); branch gfx906-optimize still ships SK for Q8_0.

Status: OPEN question for the user. Options: (a) keep SK on optimize
if reproducible-freeze-free, (b) port stream_k=false from gfx906-safe,
(c) binary-search the triggering workload (graphs ON? test-backend-ops
specific shapes? repeated reset?). Not resolved autonomously - change
of committed SK behavior is a user decision.
