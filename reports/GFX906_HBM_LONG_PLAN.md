# gfx906 (Vega20 / Radeon VII) HBM long plan - Ryzen 7 3700X DDR4 platform

Branch: `gfx906-ryzen-v2` `35b53a5f0` (+ `c010f2fb3` revert scalar, `HBM K512 1ccb8bbbd` `J128 577b0f32e` `+43%` `pp512 1773 vs1666 +6.5% KV tile`)
HW: Ryzen 7 3700X 8C16T 32GB DDR4 31Gi, Vega20 gfx906 60CU 16GB HBM2 1TB/s, GP102 GTX1080Ti idle
ROCm 7.2 hipcc 7.2.53211 clang 22.1.8, `GGML_HIP + VEGA20 dispatch` via `mmq-config-gfx906.cuh`
Date: 2026-08-28

## 0. Perspective correction vs old `i5-3570 DDR3` branch

Old `gfx906-exp-handoff`/`gfx906-optimize` baselines were:
- CPU bound: `i5-3570 12GB DDR3` -> `pp1142` flat for all quants, `tg 167->189` only, PCIe retrains on long runs
- Now: `DDR4 31Gi`, Ryzen not bottlenecked, HBM 1TB/s is the ceiling. Measured `Q4_0 pp512 1383 tg96`, `Q4_K_M pp512 1170 tg86` proves `ROCM` path is healthy, long think `6K` at `85 t/s` steady no freeze => old freezes were machine-level, not Stream-K

So we STOP reasoning as `VRAM 4GB / DDR3 starved` and START as `HBM-rich, VGPR-bound, occupancy-rich`:

- `tg` uses ~22% of 1TB/s (2.5GB * 90/s = 225GB/s) -> kernel overhead, not wire, is limit
- `pp` at 8192 should be faster than 512 (more reuse) but was `1224 vs 1383` -> tiling/concurrency, not compute
- `mmvq 256 thr -> 70 t/s regress` proved we hit VGPR/SGPR, not bandwidth

Personal branch => we can be aggressive, break simplicity, target any model (Q4_0 clean and K_M mixed).

## 1. What we have (locked 6-rank)

| rank | patch | file | gain (Q4_K_M p512 unless noted) | note |
|------|-------|------|----------------------------------|------|
| 6.1 | Q8 nthr 256->512 | mmq-config | +36% pp128 +10% pp512 | K_M still gains even though Q8 is KV only |
| 6.2 | Q8 J=128 clamp64 | mmq-config + mmq.cuh | part of above | halves x reloads, clamp keeps filler |
| 6.3 | VDR 2->8 | vecdotq.cuh | +4% tg32 | hybrid halves theoretical 19% on gemma |
| 6.4 | FA vec 128->256 | fattn-vec.cuh | +4% tg | HBM lets 4x wave64 hide |
| 6.5 | FA tile 81 configs | fattn-tile.cuh | neutral on this model | 75% DeltaNet masks FA |
| 6.6 | Q4_K 512 | mmq-config | +9% pp128 | verified on K_M |
| 6.7 | Q5/6 I128->64 | mmq-config | +4% pp128 +3% pp1024 | HBM can keep 64 |
| 7.1 | K_vram 256->512 HBM | mmq.cuh + mmq-config J72-128 | +43% p512 1676 vs1170 K_M, 1222 vs878 pp128 | LDS 32KB->64KB halves K stores |
| 7.2 | J128 with K512 | mmq-config | +4.8% p1024 1787 vs1705 | large K keeps occupancy |
| 7.3 | KV 32->64 HBM reuse | fattn-tile.cuh:363 | +6.5% K_M 1773 vs1666, +7.6% Q4_0 2183 vs2027 | 12KB LDS fits 64KB occ2 |
| 7.4 | sdot V_DOT2 Vega scalar fix (reverted) | common.cuh 706/753 | -90% 178 vs1771 REVERTED | fast sdot kept |

Quarantined: Stream-K Q8 true => +2-5% pp on K_M but -7.6% tg on Q4_0 => quant-dependent, not merged.
Rejected neutral this host: Q4_0 512 (1023 vs 1023 flat), MMVQ nwarps 4 (70 vs 86 regress), rows_per_block 2 (flat).

## 2. Ledger (live)

See `reports/EXPERIMENT_LEDGER.md` - every idea -> status `DONE / TESTED_NEUTRAL / QUEUED / IDEA`.

We run unattended: each wave = independent files, can repro + bench both `Q4_0.gguf` and `Q4_K_M.gguf` at `p 128/512/1024/8192 n32`.

## 3. Long plan - generic before specific (reordered 2026-08-28, HBM rich)

Generic wins stack across all quants, specific per-dtype last. Parallel where no same-file overlap.

Wave G1 - Generic attention (biggest tg share, 75% DeltaNet)
- g1a: DeltaNet `gated_delta_net 7.7% tg` - add VEGA20 dispatch `num_warps 4->8/2`, `S_v` tile, `__launch_bounds__ occ 2->1` (file `gated_delta_net.cu`, not mmq-config)
- g1b: FA QK/KV HBM tile - J already 128 for MMQ, try KV 32->64 with HBM (`fattn-tile.cuh`)

Wave G2 - Generic vec / dequant (HBM vs VGPR)
- g2a: VDR 16 (was 8->16 plateau on gemma, test HBM) (`vecdotq.cuh`)
- g2b: larger y-tile 128->256 with HBM, LDS layout switch (revisit DDR3 rejects)
- g2c: fused dequant+dot
Profiling with rocprof: VGPR/SGPR/lane.

Wave G3 - Generic system / compiler
- g3a: larger MMQ J 256 (HBM caches more x) (`mmq-config`)
- g3b: wave32 hint for Vega, `__launch_bounds__` tuning
- g3c: compiler `-munsafe-fp-atomics --amdgpu-target=gfx906` + occupancy 64
- g3d: 8K ctx `pp8192 >= pp512` prove reuse, 6K gen stability (no PCIe retrain)

Wave S - Specific per-quant MMQ (after generics, HBM lets waste y-tiles)
- s1: Q3_K 512 thr (9% K_M) - TESTED_NEUTRAL 1165 vs 1166 flat
- s2: Q2_K 512 thr, IQ sweeps, MXFP4 512 thr, Q6_K 256->512 thr test
Each same-file `mmq-config-gfx906.cuh` so serialize per quant, quick 1-build per guess.

## 4. Success criteria (any model)

- Q4_0: `pp512 >1450 tg >100` (current 1383/90) using only generic tweaks
- K_M: `pp512 >1250 tg >95` (current 1170/86) - tight because K_M is diverse
- HBM util: `pp8192 >= pp512` (prove reuse), `tg` bandwidth >40% of 1TB/s

## 5. Unattended exec loop

1. pick top queued ledger entry, cherry isolation `1-file 1-build`
2. `cmake --build build-hip -j12 && llama-bench p512/8192 n32` on both GGUFs
3. append to ledger with numbers + PASS/NEUTRAL/REGRESS
4. if PASS keep, else revert, next entry
5. every 3 passes push to `gfx906-ryzen-v2` with simple msg, no Co-authored
