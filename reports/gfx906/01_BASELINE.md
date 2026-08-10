# Baseline Performance - gfx906 (Radeon VII), upstream 62bf73d25

Branch: gfx906-optimize @ 62bf73d25 (== origin/master), no modifications.
Build: build-gfx906, GGML_HIP=ON, GPU_TARGETS=gfx906, Release, ROCm 7.2 (HIP clang 22.0.0git).

Model family: LFM2.5-2.6B (MoE, 2.7B params, 2.87 GB Q8_0 file size), n_ctx per test,
kv f16, flash-attn auto, batch 2048/ubatch 512.

## Prompt processing (llama-bench, 5 reps, mean t/s)

| quant | pp128 | pp256 | pp512 | pp1024 | pp2048 |
|------:|------:|------:|------:|-------:|-------:|
| Q8_0  | 1142  | 1180  | 1365  | 1377   | 1371   |
| Q4_0  | 1691  | 2412  | 2647  | 2695   | 2678   |
| Q4_K_M| 1470  | 1829  | 2107  | 2133   | 2121   |

Note: earlier observed ~1338 t/s pp512 Q8_0 in prior research is reproduced
(~1365 here; 1259-1365 range across runs, sd 26-122 for small pp).

## Token generation (llama-bench tg128, mean t/s)

| quant | tg128 (sd) |
|------:|-----------:|
| Q8_0  | ~167 (1-5) |
| Q4_0  | ~189 (2-16)|
| Q4_K_M| ~163 (1)   |

## Token generation at depth (context pre-fill; flash-attn auto, kv f16)

| quant | d=0   | d=4k  | d=16k | d=32k | d=64k | d=100k |
|------:|------:|------:|------:|------:|------:|-------:|
| Q4_0  | 186.3 | 181.0 | 170.1 | 161.4 | 133.3 | 123.7  |
| Q4_K_M| 169.1 | 163.8 | 154.0 | 143.6 | 122.7 | 111.7  |
| Q8_0  | 163.4 | 155.5 | 151.3 | 135.1 | 127.0 | 113.1  |

Raw CSV: reports/gfx906/results/baseline.csv (pp/tg128), reports/gfx906/results/tg_depths.csv (depth matrix).

## Correctness baseline

test-backend-ops test -b ROCm0 -o MUL_MAT  : 1186/1186 OK
test-backend-ops test -b ROCm0 -o MUL_MAT_ID : 865/865 OK

## Notes

- GPU clocks at idle: sclk 808 MHz, mclk 350 MHz, 20 W. Under load clocks rise; watch for
  throttling when comparing small deltas (Radeon VII boost behavior).
- No profiler tools (rocprof*) installed on this system; compiler resource reports and
  rocm-smi counters are the available instrumentation.
- No F16 model available locally (user: run without F16). Q8 vs F16 pp anomaly from earlier
  research cannot be re-tested directly here without a new download.
- Effective model-byte rates at tg128: Q8_0 ~476 GiB/s, Q4_0 ~300 GiB/s, Q4_K_M ~274 GiB/s
  (model-size x tokens/s; NOT measured HBM traffic).