# Example KV cache policies

Policy files for the Qwen3-4B model (GGUF SHA256
7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5).

Each file follows the KV cache policy schema v1. See
docs/beyond-perplexity/POLICY_SCHEMA.md for the full schema and validation
rules.

| file           | K type | V types (36 layers)                    | bytes/token |
|----------------|--------|----------------------------------------|-------------|
| q4xq4.json     | q4_0   | q4_0 x36                               | 41472       |
| q4xq3_64.json  | q4_0   | q3_0_64 x36                            | 35712       |
| c2.json        | q4_0   | q2_nu x28, q3_0_64 x4, q4_0 x4         | 32768       |
| allq2_nu.json  | q4_0   | q2_nu x36                              | 31104       |
| q2k_f16v.json  | q2_nu  | f16 x36                                | 84096       |

Usage with the Qwen3-4B-Instruct Q4_K_M model:

```
llama-cli -m Qwen3-4B-Q4_K_M.gguf -ngl 99 --kv-cache-policy c2.json \
    -p "Hello" -n 16 --single-turn
```

Quantized V caches require Flash Attention. The runtime validates the policy
against the loaded model (architecture, layer count, KV geometry, model
SHA256, codebook SHA256) and aborts with an error on any mismatch; there is
no silent fallback.

q2k_f16v.json is an auxiliary example that keeps K quantized and V in f16; it
is not part of the paper's production policies.
