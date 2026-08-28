# gfx906 (Vega20 / Radeon VII) HBM long plan - Ryzen 7 3700X DDR4 platform

Branch: `gfx906-ryzen-v2` `d5b9855c2` (6-rank stable)
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

Quarantined: Stream-K Q8 true => +2-5% pp on K_M but -7.6% tg on Q4_0 => quant-dependent, not merged.
Rejected neutral this host: Q4_0 512 (1023 vs 1023 flat), MMVQ nwarps 4 (70 vs 86 regress), rows_per_block 2 (flat).

## 2. Ledger (live)

See `reports/EXPERIMENT_LEDGER.md` - every idea -> status `DONE / TESTED_NEUTRAL / QUEUED / IDEA`.

We run unattended: each wave = independent files, can repro + bench both `Q4_0.gguf` and `Q4_K_M.gguf` at `p 128/512/1024/8192 n32`.

## 3. Long plan - 4 waves (parallel where no same-file overlap)

Wave 1 - MMQ quant coverage (HBM lets us waste y-tiles)
- w1a: Q3_K 512 thr (mirrors Q4_K, 9% of K_M)
- w1b: Q2_K 512 thr (rare but free)
- w1c: IQ1_S/IQ2/IQ3/IQ4 512 thr sweep (future IQ models)
- w1d: MXFP4 512 thr (new arch,  mxxm-t already does Q8+MXFP4)
- w1e: Q6_K 256->512 thr test (sibling of Q5_K, currently 256)
Each is same-file `mmq-config-gfx906.cuh` so serialize per quant type, but quick 1-build per guess.

Wave 2 - Vec / dequant (VGPR vs HBM tradeoff)
- w2a: VDR 16 (was 8->16 plateau on gemma, test HBM)
- w2b: LDS layout switch Q4_K vs Q4_0 (different sram_layout)
- w2c: fused dequant + dot (reduce trips, costs VGPR)
- w2d: larger y-tile 128->256 with HBM (revisit rejected K=512 with HBM)
Profiling with rocprof: VGPR/SGPR/lane.

Wave 3 - Attention heavy tails (the 75% you actually pay)
- w3a: DeltaNet `gated_delta_net 7.7% tg` - port gfx906 config (currently RDNA fallback), test 8 thr/row tile as for FA
- w3b: FA QK/KV HBM tile - J already 128 for MMQ, try KV tile 32->64 with HBM
- w3c: rope/norm fused with HBM (small but x many layers)
These touch `ggml-cuda/delta*`, `fattn-tile.cuh`, not `mmq-config`.

Wave 4 - System / compiler
- w4a: larger MMQ J 256 (HBM can cache more x)
- w4b: wave32 emulation hint for Vega (some kernels assume 32)
- w4c: compiler `-munsafe-fp-atomics --amdgpu-target=gfx906` + occupancy 64
- w4d: 8K long-context stability suite (8192 pp + 6K gen) on each wave, no more PCIe retrain

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
