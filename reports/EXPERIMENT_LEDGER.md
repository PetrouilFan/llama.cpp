# Experiment ledger - gfx906 HBM (Ryzen DDR4, no CPU cap)

Source of truth for unattended runs. Update after each bench, no rm.
Columns: ID | file | patch | Q4_0 delta | K_M delta | tg delta | status | note

## DONE (on `d5b9855c2` + build-hip)

| ID | file | patch | Q4_0 | K_M p512 | tg | status | vs vanilla 567/1024 tg79 |
|----|------|-------|------|----------|----|--------|--------------------------|
| D01 | mmq-config+hip.h+mmq.cuh | Q8 nthr512 J128 clamp | +36% pp128 | +10% 1024->1134 | +5% | DONE | 771 pp128 1134 p512 |
| D02 | vecdotq.cuh | VDR 2->8 | flat | +0.2% | +4% 83->86 | DONE | 86.6 |
| D03 | fattn-vec.cuh | FA vec 128->256 | flat | flat | +4% 86->90 | DONE | 90.2 |
| D04 | fattn-tile.cuh | 81 tile gfx906 | neu | neu | neu | DONE_NEUTRAL | 90.4 |
| D05 | mmq-config | Q4_K 512 | - | +9% 771->842 | flat | DONE |  |
| D06 | mmq-config | Q5/6 I128->64 J64 | - | +4% 842->875 pp128, 1139->1174 p1024 | flat | DONE | final 878 p128 1170 p512 1170 p1024 |
| Q01 | mmq-config | Q4_0 512 (test) | 1369 vs 1383 -1% | 1143 vs 1170 -2% (noise) | flat | TESTED_NEUTRAL | revert 256, not merged |

Quarantined: SK01 Stream-K Q8 true (mmq-config 27 CASE false->true) p512 +1% Q4_0 flat +2-5% K_M, tg -7.6% on Q4_0 => branch `gfx906-streamk-test 006513ab4` only.

Rejected earlier (DDR3 polluted but re-check with HBM if needed): K 512, I-tile 64 generic, shared 16K - keep queued w4.

## QUEUED - generic before specific (reordered 2026-08-28)

Generic wins stack across all quants, specific per-dtype last.

| ID | file | idea | HBM rationale | status |
|----|------|------|---------------|--------|
| G1 | ggml-cuda/gated_delta_net.cu | DeltaNet 4->8 warps (*4->*8 launch_bounds) | 75% layers, 7.7% tg | TESTED_NEUTRAL 1377 vs 1390 -0.9% Q4_0, 1161 vs 1166 -0.4% K_M, tg +2.8% K_M flat | revert 4 |
| G2a | vecdotq.cuh | VDR_Q4_K_MMVQ 8->16 | gemma plateau 16 | TESTED_NEUTRAL 1167 vs 1166 flat, Q4_0 1385 vs 1390 flat | revert 8 |
| G2b | mmq-config | Q4_K J 64->128 (512 thr) HBM tile | HBM caches more x | TESTED_NEUTRAL/REGRESS 1112 vs 1166 -4.6% p512, 1198 vs 1170 +2.4% p1024, std 178 | revert 64 |
| G3a | mmq-config | Q4_K occupancy 2->1 512thr | vgpr bound HBM | TESTED_NEUTRAL 1162 vs 1171 -0.7% p512 1377 vs 1390 -0.9% Q4_0 | revert 2 |
| G3b | common / hip | compiler -munsafe-fp-atomics, occupancy, wave64 hint | one flag rebuild benefits every model | NEXT |
| S1 | mmq-config | Q6_K 256->512 thr | 22% K_M | TESTED_NEUTRAL/REGRESS 1154 vs 1171 -1.4% p512 1116 vs 1170 -4.6% p1024 | revert 256 |
| S2 | mmq-config | MXFP4 256->512 | new q type | TESTED_NEUTRAL 1161 vs 1171 -0.8% p512 flat tg 91 vs 85 noise | revert 256 |
| S3 | mmq-config | Q2_K 256->512 | rare but free y | TESTED_NEUTRAL 1164 vs 1171 -0.6% flat | revert 256 |
| S4 | mmq-config | IQ1_S/IQ2_XS/IQ2_S/IQ3/IQ4_NL 512 thr | future IQ models, same infra | SKIPPED_LOW_ROI - Q3_K/Q2_K neutral predicts neutral |
| W1a | mmq-config | Q3_K 256->512 thr | 9.5% of K_M | TESTED_NEUTRAL 1165 vs 1166 flat |
| W4 | mmq.cuh | K_vram 256->512 HBM larger K block | HBM 1TB/s keeps more K in LDS, halves stores | DONE_HBM_WIN 1676 vs 1171 +43% p512 K_M 1705 vs 1170 +45% p1024 1222 vs 878 +39% p128 1590 vs 1089 +46% p8192 Q4_0 2149 vs 1390 +54% |
| W2b | mmq-config | sram_layout Q8_1 vs Q8_0 for K_M | test HBM cache | IDEA |
| W3b | fattn-tile.cuh | KV tile 32->64 with HBM | HBM reuse | IDEA |

## IDEA backlog (think multiple perspectives)

- k_get_rows_float_vec 7.2% tg -> wide gather 128B with HBM
- rope + norm fusion (small per layer x 48)
- wave32 hint for Vega20 dispatch (some kernels branch on 32)
- 8K ctx: pp8192 should >= pp512 if tiling correct (cur 1224 <1383 regress)

## Bench refs (this host, 6-rank d5b9855c2)

- Q4_0 512thr test neutral: Q4_0 p128 1023 p512 1369 p1024 1373 p2048 1362 p8192 1261 tg32 90.0 (build d5b9855c2 after revert clean should be 1383/1261 tg96 - noise)
- K_M 6-rank: 878 p128 1170 p512 1170 p1024 1140 p2048 1089 p8192 86 tg (vs vanilla 567/1024/1071/1107/1074)
- StreamK K_M: 911 p128 1192 p512 1195 p1024 1176 p2048 tg90
- Fail signal to watch: PCIe retrain, tg drop >10%, vgpr spill
