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

## QUEUED (next, in priority order)

W1 = MMQ coverage, W2 = vec/VGPR, W3 = attention, W4 = system

| ID | file | idea | HBM rationale | status |
|----|------|------|---------------|--------|
| W1a | mmq-config | Q3_K 256->512 thr (all J 8..64 both SK) | 9.5% of K_M, same super-block win as Q4_K, HBM lets 512 | TESTED_NEUTRAL 1165 vs 1166 flat |
| W1b | mmq-config | Q2_K 256->512 | rare but free y | QUEUED |
| W1c | mmq-config | IQ1_S/IQ2_XS/IQ2_S/IQ3/IQ4_NL 512 thr | future IQ models, same infra | QUEUED |
| W1d | mmq-config | MXFP4 512 thr (mirrors mxxm-t) | new q type, 512 thr already proven | QUEUED |
| W1e | mmq-config | Q6_K 256->512 thr test | sibling of Q5_K, 22% of K_M | QUEUED |
| W2a | vecdotq.cuh | VDR 16 (from 8) | gemma plateau 16, HBM may lift | QUEUED |
| W2b | mmq-config | sram_layout Q8_1 vs Q8_0 for K_M | test HBM cache | IDEA |
| W3a | ggml-cuda/gated_delta | DeltaNet gfx906 tile | 75% of Qwen3.5 layers, 7.7% tg | QUEUED_HIGH |
| W3b | fattn-tile.cuh | KV tile 32->64 with HBM | HBM reuse | IDEA |
| W4a | mmq-config | J 256 (x-tile) | HBM caches more x, retry old reject | IDEA |
| W4b | common | compiler flags / occupancy 64 | VGPR trade | IDEA |

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
