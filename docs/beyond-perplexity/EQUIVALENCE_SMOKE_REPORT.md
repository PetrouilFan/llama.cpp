# Clean-vs-legacy quality equivalence smoke report

Date: 2026-08-05

## Scope

Bounded smoke matrix per release plan:

- policies: Q4K-Q4V, Q4K-Q3V (q4xq3_64), C2
- implementations: legacy frozen (bc80cdb03 paper/full-evaluation) and clean current-upstream (e031d9567 + feature commits)
- datasets: 64 fixed Wiki held-out chunks, 64 fixed OWT held-out chunks, fixed GSM8K-200 subset
- identical chunk/example IDs and inference arguments

## Binaries

| component | implementation | sha256 |
|---|---|---|
| llama-perplexity | legacy build-paper-release | 99d7efb2e0a0ecfdb549d3eafd9c9104a894e9eb1b7284eb796fcded390af9a3 |
| llama-perplexity | clean build-bp-cuda-release | e2c40afca435fa4799aba1aa93922fdfdd7305f49ab02bb318d97b78e095fb03 |
| model | Qwen3-4B-Q4_K_M.gguf | 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 |

## Method

Per-chunk NLL via `--ppl-output-type 2`, n_ctx=2047, 64 chunks, flash attention auto.

Legacy invocation (policy via -ctk/-ctv plus LLAMA_KV_V_LAYERS for C2):

```
llama-perplexity -m MODEL -ctk <k> -ctv <v> -ngl 99 -c 2047 --chunks 64 --ppl-output-type 2 -f corpus
LLAMA_KV_V_LAYERS='q4_0:0;q2_0:1-6;q4_0:7;q2_0:8;q3_0_64:9;q2_0:10-13;q3_0_64:14;q2_0:15;q3_0_64:16;q2_0:17-19;q4_0:20;q2_0:21-22;q4_0:23;q2_0:24-33;q3_0_64:34;q2_0:35'
```

Clean invocation (policy via --kv-cache-policy file):

```
llama-perplexity -m MODEL -ngl 99 -c 2047 --chunks 64 --ppl-output-type 2 --kv-cache-policy <policy>.json -f corpus
```

Policy files: examples/kv-cache-policies/{q4xq4,q4xq3_64,c2}.json

Raw logs and machine-readable per-chunk data:

- artifacts/beyond-perplexity/equivalence_smoke/{legacy,clean}_{wiki,owt}_{q4xq4,q4xq3_64,c2}.log
- artifacts/beyond-perplexity/equivalence_smoke/nll_per_chunk.json

## Aggregate PPL

| dataset | policy | clean PPL | legacy PPL | identical |
|---|---|---|---|---|
| wiki | q4xq4 | 12.3227 | 12.3227 | yes |
| wiki | q4xq3_64 | 12.3552 | 12.3552 | yes |
| wiki | c2 | 12.4243 | 12.4243 | yes |
| owt | q4xq4 | 20.3030 | 20.3030 | yes |
| owt | q4xq3_64 | 20.3160 | 20.3160 | yes |
| owt | c2 | 20.2493 | 20.2493 | yes |

## Policy-effect delta (NLL per token, policy minus that implementation Q4K-Q4V)

| dataset | policy | clean delta | legacy delta | clean - legacy |
|---|---|---|---|---|
| wiki | q4xq3_64 | +0.002633 | +0.002633 | 0.000000 |
| wiki | c2 | +0.008213 | +0.008213 | 0.000000 |
| owt | q4xq3_64 | +0.000639 | +0.000639 | 0.000000 |
| owt | c2 | -0.002650 | -0.002650 | 0.000000 |

Primary acceptance gate:

```
abs(clean(policy NLL - clean Q4 NLL) - legacy(policy NLL - legacy Q4 NLL)) <= 0.002
```

PASS on all four cells; measured value 0.000000.

## Per-chunk agreement

For every one of the 768 chunk comparisons (12 runs x 64 chunks), the mean per-chunk NLL of clean and legacy agree to all six printed decimals; maximum absolute per-chunk difference is 0.000000.

Policy ordering within each implementation is identical: wiki c2 > q4xq3_64 > q4xq4 (worse), owt c2 < q4xq4 < q4xq3_64 (c2 best).

## GSM8K-200

Fixed subset: first 200 docs of the openai/gsm8k test split, 5-shot, via
lm-evaluation-harness 0.4.13.dev0 local-chat-completions against
llama-server (n_ctx 4096, -fa auto), identical arguments for both
implementations.

| policy | legacy strict-match | clean strict-match | clean - legacy |
|---|---|---|---|
| q4xq4 | 0.645 | 0.645 | 0.000 |
| q4xq3_64 | 0.685 | 0.685 | 0.000 |
| c2 | 0.61 | 0.61 | 0.000 |

- Legacy q4xq4 (0.645) reproduces the frozen revision4 gsm8k_200.csv value
  (q4_0 = 0.645) exactly.
- Per-example agreement: every cell reports identical correct-set counts
  across implementations; clean-vs-legacy agreement is 100% (all 200
  samples identical outcome).
- The legacy q4xq3_64 cell requires LLAMA_KV_V_LAYERS='q3_0_64:0-35' (the
  legacy -ctv flag does not accept q3_0_64); the clean cell uses
  --kv-cache-policy q4xq3_64.json.
- Note: legacy c2 (0.61) and q4xq3_64 (0.685) values are new measurements
  on the frozen subset; the historical paper values on the full 1319-example
  set are 0.6262 (c2) and 0.671 (q4xq3_64). See EQUIVALENCE_REPORT.md for
  the full-set runs.

## Interpretation

The clean current-upstream implementation is numerically identical to the legacy implementation on the flash-attention path: both use FP32 flash attention kernels, so per-chunk NLL matches to every printed digit and the policy-effect differences are exactly zero. The policy resolution mechanism (flat per-layer policy file in clean versus -ctk/-ctv + LLAMA_KV_V_LAYERS env in legacy) has zero effect on NLL.

Absolute PPL values differ from the historical paper numbers because the smoke matrix uses 64 chunks at n_ctx=2047, whereas the paper used the full frozen corpus at n_ctx=4096.
