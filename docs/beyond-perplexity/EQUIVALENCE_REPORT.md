# Equivalence report

Clean current-upstream implementation of the Beyond Perplexity paper results,
compared against the historical paper values and the legacy frozen
implementation.

## Scope

All measurements use the frozen paper model
Qwen3-4B-Instruct Q4_K_M (GGUF SHA256
7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5) and the
frozen held-out corpora (wiki / owt splits of the paper's rate-distortion
splits, rd4.3).

## Full-corpus held-out PPL (ctx 4096, full corpus, FlashAttention)

Clean values produced with:

```
llama-perplexity -m <model> -ngl 99 -c 4096 --kv-cache-policy <policy> \
    --chunks -1 --ppl-output-type 2 -f <corpus>
```

| policy    | corpus | clean PPL | paper PPL | diff |
|-----------|--------|-----------|-----------|------|
| q4xq4     | wiki   | 12.2919   | 12.2919   | 0.0000 |
| q4xq4     | owt    | 18.4667   | 18.4667   | 0.0000 |
| q4xq3_64  | wiki   | 12.3452   | 12.3452   | 0.0000 |
| q4xq3_64  | owt    | 18.5276   | 18.5276   | 0.0000 |
| c2        | wiki   | 12.2435   | 12.2435   | 0.0000 |
| c2        | owt    | 18.3585   | 18.3585   | 0.0000 |

All six cells reproduce the paper values exactly.

## Full GSM8K (n = 1319, 5-shot, strict match)

Clean values produced with lm-evaluation-harness 0.4.13 via
local-chat-completions against llama-server (-c 4096, FlashAttention,
--kv-cache-policy).

| policy    | clean strict | paper strict | diff |
|-----------|--------------|--------------|------|
| q4xq4     | 0.6543       | 0.6543       | 0.0000 |
| q4xq3_64  | 0.6710       | 0.671        | 0.0000 |
| c2        | 0.6262       | 0.6262       | 0.0000 |

All three cells reproduce the paper values exactly.

## Bounded smoke equivalence (clean vs legacy frozen implementation)

Per-chunk NLL (--ppl-output-type 2, ctx 2047, 64 chunks) for wiki and owt is
bit-identical between the clean and legacy implementations on every chunk
(maximum per-chunk mean difference 0.000000). GSM8K-200 strict match is
identical per policy (q4xq4 0.645, q4xq3_64 0.685, c2 0.61).

Primary gate:

```
abs(clean(policy NLL - clean q4 NLL) - legacy(policy NLL - legacy q4 NLL)) <= 0.002
```

Result: 0.000000 on all cells. PASS.

See docs/beyond-perplexity/EQUIVALENCE_SMOKE_REPORT.md for the full matrix.

## Policy-effect ordering

- wiki PPL: c2 (12.2435) < q4xq4 (12.2919) < q4xq3_64 (12.3452)
- owt PPL: c2 (18.3585) < q4xq4 (18.4667) < q4xq3_64 (18.5276)

The ordering matches the paper (c2 improves both corpora; q4xq3_64 is the
lossiest of the three production policies on held-out PPL).

## NU-Q2 versus uniform-Q2 (equal storage, frozen paper numbers)

The uniform Q2 (q2_0) KV format is not expressible in the clean CUDA
runtime (no SET_ROWS support; see LIMITATIONS.md), so the equal-storage
comparison uses the frozen paper measurements:

| benchmark      | allq2_nu | allq2_uniform |
|----------------|----------|---------------|
| GSM8K strict   | 58.53    | 31.01         |
| MBPP           | 48.00    | 30.80         |
| TruthfulQA bleu| 11.1249  | 7.7857        |

llama-3.2-1b cross-model held-out PPL: uniform Q2 all-V 13.0153 (wiki) /
19.0996 (owt) versus Q4xQ4 baseline 11.8001 / 17.3042.

NU-Q2 remains materially better than uniform Q2 at equal storage.

The uniform-Q2 numbers above are frozen paper measurements: the clean CUDA
runtime has no SET_ROWS support for `q2_0`, so uniform-Q2 KV is not
reproducible at runtime. This gate is WAIVED and recorded in
RELEASE_MANIFEST.json (`waived_gates`); see LIMITATIONS.md.

## Systems

C2 decode is slightly slower than the clean q4xq4 baseline at every
measured context (2.4-3.4% overhead):

| ctx | q4xq4 dec t/s | c2 dec t/s | overhead |
|-----|---------------|------------|----------|
| 4K  | 84.0          | 82.0       | -2.38%   |
| 16K | 50.1          | 48.4       | -3.39%   |
| 32K | 31.9          | 31.0       | -2.82%   |

Sign convention: negative means c2 is slower than q4xq4.

Release gate (<= 4% C2 decode overhead at 32K and 64K): PASS at 32K. 64K
decode is not measurable in any clean binary because upstream
tools/server/server-context.cpp caps slots at n_ctx_train (40960 for this
model; commit ab49f094d2, present in the pinned base). 64K VRAM was measured
via llama-perplexity: q4xq4 5674 MiB, c2 4986 MiB (analytical KV saving
544 MiB; see SYSTEMS_EQUIVALENCE_REPORT.md).

See docs/beyond-perplexity/SYSTEMS_EQUIVALENCE_REPORT.md.

## Interpretation

- The clean current-upstream implementation reproduces every paper number
  exactly (PPL and GSM8K-1319, all cells).
- Clean-vs-legacy policy-relative deltas are 0.000000 (bit-identical FA
  numerics on the smoke matrix).
- Absolute drift between smoke (64 chunks, ctx 2047) and full (ctx 4096)
  numbers is a harness difference (chunk count / context), not an
  implementation difference: the full-corpus runs match the paper exactly.
