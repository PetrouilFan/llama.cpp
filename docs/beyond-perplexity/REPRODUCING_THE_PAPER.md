# Reproducing the Paper

This document explains how to reproduce the Beyond Perplexity measurements on
the clean current-upstream implementation, and how the clean numbers relate to
the historical paper numbers.

## Frozen inputs

| Input | Location / hash |
|-------|-----------------|
| Model | Qwen3-4B-Instruct Q4_K_M, SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` (paper model, not committed to this repo) |
| Held-out Wiki corpus | frozen split `final_test_wiki.txt` (from the paper rd4.3-splits; not committed) |
| Held-out OWT corpus | frozen split `final_test_owt.txt` (from the paper rd4.3-splits; not committed) |
| GSM8K | `openai/gsm8k` test split, first 200 examples for the smoke subset, full 1,319 for the release matrix |
| Policies | `examples/kv-cache-policies/*.json` (committed) |

The frozen corpus splits and the model are part of the paper artifact set and
are intentionally not committed to this repository. The reference documents
under `docs/beyond-perplexity/reference/` describe the paper-side manifests.

## Building

```sh
cmake -S . -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build -j 8
```

The release verification used CUDA 13.3 on an RTX 3070 Laptop (cc 8.6); see
BUILD_ENVIRONMENT.md.

## Held-out perplexity

The paper reported aggregate PPL over the full frozen corpora at context
4096. The clean branch reproduces the same measurements with the policy file
instead of `-ctk`/`-ctv`/`LLAMA_KV_V_LAYERS`.

Per-chunk NLL (smoke, 64 chunks at context 2047):

```sh
./build/bin/llama-perplexity -m MODEL -ngl 99 -c 2047 --chunks 64 \
  --ppl-output-type 2 --kv-cache-policy examples/kv-cache-policies/c2.json \
  -f final_test_wiki.txt
```

Full corpus (context 4096, all chunks):

```sh
./build/bin/llama-perplexity -m MODEL -ngl 99 -c 4096 \
  --ppl-output-type 2 --kv-cache-policy examples/kv-cache-policies/c2.json \
  -f final_test_wiki.txt
```

FlashAttention is enabled by default (auto). Quantized V requires it and the
context creation fails with an explicit error if it is disabled.

## GSM8K

The paper used lm-evaluation-harness with a llama-server
`local-chat-completions` backend, 5-shot, strict match, first-200 subset for
the smoke runs and full 1,319 for the release matrix:

```sh
./build/bin/llama-server -m MODEL -ngl 99 -c 4096 \
  --kv-cache-policy examples/kv-cache-policies/c2.json --host 127.0.0.1 --port PORT --reasoning off
python -m lm_eval run --model local-chat-completions \
  --model_args model=local,base_url=http://127.0.0.1:PORT/v1/chat/completions,num_concurrent=1,max_retries=3,timeout=300 \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 --apply_chat_template \
  --confirm_run_unsafe_code --output_path OUT --limit 200
```

## Policies

The four production operating points are committed under
`examples/kv-cache-policies/`:

| Policy | File | K | V | bytes/token |
|--------|------|---|----|-------------|
| Q4K-Q4V | `q4xq4.json` | q4_0 | q4_0 x36 | 41,472 |
| Q4K-Q3V | `q4xq3_64.json` | q4_0 | q3_0_64 x36 | 35,712 |
| C2 | `c2.json` | q4_0 | q4_0 x4, q3_0_64 x4, q2_nu x28 | 32,768 |
| Q4K-all-NU-Q2V | `allq2_nu.json` | q4_0 | q2_nu x36 | 31,104 |

## Clean vs historical paper numbers

The paper numbers (aggregate PPL at context 4096, GSM8K, memory, throughput)
are preserved in `docs/beyond-perplexity/reference/06_PAPER_IMPLEMENTATION_TARGETS.json`
and reproduced in EQUIVALENCE_REPORT.md. The verification strategy accepts
absolute drift due to the newer upstream base and requires equivalence of the
policy effect relative to each implementation's own Q4K-Q4V baseline.

Smoke results on the clean branch (64 chunks, context 2047, FlashAttention):

- Wiki: q4xq4 12.3227, q4xq3_64 12.3552, c2 12.4243
- OWT: q4xq4 20.3030, q4xq3_64 20.3160, c2 20.2493

These are bit-identical to the legacy implementation run with identical
arguments (EQUIVALENCE_SMOKE_REPORT.md).

## Full release matrix

The full release matrix (full-corpus Wiki/OWT PPL, full GSM8K-1319, systems
measurements, NU-vs-uniform ablation) is reported in EQUIVALENCE_REPORT.md
and SYSTEMS_EQUIVALENCE_REPORT.md. The NU-vs-uniform Q2 equal-storage
ablation cannot be reproduced at runtime on the clean branch: the uniform
`q2_0` KV format is not supported by the CUDA cache path (see LIMITATIONS.md),
so that comparison uses the frozen paper numbers.
