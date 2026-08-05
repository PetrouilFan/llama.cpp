# Policy runtime verification

Date: Wed Aug 05 2026

Scope: verify the public policy interface (`--kv-cache-policy FILE`) on the
paper model Qwen3-4B-Instruct Q4_K_M (sha256
7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5) for the four
release policies, plus negative tests. Machine-readable results:
`artifacts/beyond-perplexity/runtime_policy_audit.json`.

## Environment

- Binary: `build-bp-cuda-release/bin/llama-perplexity` (GGML_CUDA=ON,
  GGML_CUDA_FA=ON, GGML_CUDA_FA_ALL_QUANTS=ON), built from the feature branch.
- Runner: `llama-perplexity -m <model> -ngl 99 -c 2047 --chunks 2
  --ppl-output-type 2 --kv-cache-policy <policy> -f <wiki smoke> --verbosity 4`
- Model geometry (runtime `print_info`): n_layer = 36, n_head = 32,
  n_head_kv = 8, n_embd_head_k = 128.

## Positive verification

Runtime evidence per policy (from `llama_context` and `llama_kv_cache` INFO
logs; full logs archived under /tmp/opencode/pos_*.log):

| policy | policy_id | policy_sha256 | runtime log | K bytes/tok | V bytes/tok | total B/tok | runtime KV buf (2048 cells) |
|---|---|---|---|---|---|---|---|
| Q4K-Q4V | qwen3-4b-q4xq4-v1 | 717ff660a5c089c053486431e1d981dc8e207b3ed8c314cae407f9bc8e7f9bbe | `KV cache policy: k=q4_0x36 v=q4_0x36` | 20736.0 | 20736.0 | 41472.0 | 81.00 MiB (K 40.50, V 40.50) |
| Q4K-Q3V | qwen3-4b-q4xq3_64-v1 | 3c24b621fd9fd95e6b45e8489da3c6e34a65515702a508ff9dc07df253045931 | `KV cache policy: k=q4_0x36 v=q3_0_64x36` | 20736.0 | 14976.0 | 35712.0 | 69.75 MiB (K 40.50, V 29.25) |
| C2 | qwen3-4b-c2-v1 | 88e3b00a06de6fa934974bc0ee609e592a80622403b2fa2f3f91a0badb110444 | `KV cache policy: k=q4_0x36 v=q4_0,q2_nux6,q4_0,q2_nu,q3_0_64,q2_nux4,q3_0_64,q2_nu,q3_0_64,q2_nux3,q4_0,q2_nux2,q4_0,q2_nux10,q3_0_64,q2_nu` | 20736.0 | 12032.0 | 32768.0 | 64.00 MiB (K 40.50, V 23.50) |
| Q4K-all-NU-Q2 | qwen3-4b-allq2_nu-v1 | 7e7c054b8a137f3f6ef946392b508acc7a8d9bbb1980b9fda380ed1b8c5a2282 | `KV cache policy: k=q4_0x36 v=q2_nux36` | 20736.0 | 10368.0 | 31104.0 | 60.75 MiB (K 40.50, V 20.25) |

Analytical formula (bytes/token = n_layers x n_kv_heads x head_dim x
bytes_per_element, per side, summed over layers): q4_0 = 18/32 = 0.5625,
q3_0_64 = 26/64 = 0.40625, q2_nu = 18/64 = 0.28125. The runtime-allocated KV
buffer divided by 2048 cells equals the analytical bytes/token exactly for all
four policies (e.g. 81 MiB x 1024^2 / 2048 = 41472).

Per-policy runtime checks (all passed):

- policy ID, canonical policy hash: verified against the canonical hash
  function (json.dumps sort_keys, indent 2, minus policy_sha256/created_at/git_commit).
- model architecture: `architecture=qwen3` matches runtime `general.architecture`.
- layer count: 36 = model n_layer.
- KV-head count: declared n_kv_heads=8 matches runtime n_head_kv=8.
- head dimension: declared head_dim=128 matches runtime n_embd_head_k=128.
- model-hash status: declared model_sha256 equals the sha256 of the GGUF file
  streamed by the runtime at policy apply time (7485fe6f...).
- codebook hash: declared codebook_sha256 equals the implemented NU codebook
  hash af450f1aec81b9a23e346640e27ab3150f5a9827b424750eda61d4104dea6713
  (levels -0.619326, -0.186363, 0.186363, 0.619301).
- resolved K type per layer: all 36 layers resolved (k=q4_0).
- resolved V type per layer: per-layer map as logged (see table); C2 map
  matches the release spec: Q4 at 0, 7, 20, 23; Q3_0_64 at 9, 14, 16, 34;
  Q2_NU everywhere else.
- type counts: C2 = q2_nu x28, q4_0 x4, q3_0_64 x4; others uniform.
- selected attention implementation: `resolve_fused_ops: Flash Attention
  enabled` in all four runs (mixed-precision KV requires FA; no fallback).

## Negative tests

Each invalid policy must terminate with a useful error and nonzero exit. All
nine cases exit 1 with the error line below (no silent fallback):

| case | policy mutation | runtime error |
|---|---|---|
| wrong architecture | architecture=llama2 | `KV cache policy architecture 'llama2' does not match the model architecture 'qwen3'` |
| wrong KV geometry (heads) | n_kv_heads=16 | `KV cache policy n_kv_heads (16) does not match the model (8)` |
| wrong KV geometry (dim) | head_dim=96 | `KV cache policy head_dim (96) does not match the model (128)` |
| wrong layer count | n_layers=35, 35 v entries | `KV cache policy layer count (35) does not match the model (36)` |
| wrong codebook hash | codebook_sha256=f*64 | `KV cache policy codebook_sha256 'ff...' does not match the implemented NU codebook 'af45...'` |
| strict model SHA mismatch | model_sha256=0*64 | `KV cache policy model_sha256 '000...' does not match the model file '7485fe6f...'` |
| unknown type | k_type=q9_9 | `llama_kv_cache_policy_parse: unknown k_type: q9_9` |
| missing layers / overlap / out-of-range | legacy range syntax `v_layer_types=["q4_0:0;q2_0:1-6;..."]` | `llama_kv_cache_policy_parse: invalid v_layer_types` |
| unsupported type | v all `q2_0` (uniform Q2) | `layer 0 uniform Q2_0 cache type is not supported (use q2_nu)` |

Note: the release schema is a flat positional per-layer array, so legacy
range-style assignments (`q4_0:0;q2_0:1-6`), overlaps and out-of-range layers
are rejected at parse time (`invalid v_layer_types`). This is fail-closed, not
a silent fallback.

## Changes required by this verification (already in the working tree)

The runtime previously dropped policy metadata after parse and validated
nothing beyond types/block sizes/flash-attention requirements. To satisfy
"every invalid policy must terminate with a useful error", the policy struct
now retains policy_id, architecture, model_sha256, codebook_sha256,
n_kv_heads, head_dim; the context init applies the checks above; the model
loader records the source file path and the context hashes the file
(sha256/sha256.h, 64 KiB streaming) when model_sha256 is declared.

## Files

- artifacts/beyond-perplexity/runtime_policy_audit.json
- /tmp/opencode/policies/{q4xq4,q4xq3_64,c2,allq2_nu}.json (positive)
- /tmp/opencode/policies/bad/*.json (negative)
- /tmp/opencode/pos_{q4xq4,q4xq3_64,c2,allq2_nu}.log (runtime logs)
- /tmp/opencode/neg{2,3,4}_*.log (negative-run logs)
