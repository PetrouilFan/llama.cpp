# Runtime smoke report

Date: 2026-08-05
Branch: feature/beyond-perplexity-kv-cache-v1 (commit 945fdc4a9 + uncommitted validation changes)
Upstream base: e031d956797f9a10f704fa523fe5fd9b72e9015c

## Binaries and model

| artifact | sha256 |
|---|---|
| build-bp-cuda-release/bin/llama-cli | cf29696f38b771a984037f8e417530c51562335a34574b4b529e8f6deebd4db6 |
| build-bp-cuda-release/bin/llama-server | 59150a181708d43037ec90737d8625b691c68a8aec82f290cc9c8f43cebaaf76 |
| build-bp-cuda-release/bin/llama-perplexity | e2c40afca435fa4799aba1aa93922fdfdd7305f49ab02bb318d97b78e095fb03 |
| Qwen3-4B-Instruct Q4_K_M.gguf | 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 (frozen paper model) |

Build: GGML_CUDA=ON, GGML_CUDA_FA=ON, GGML_CUDA_FA_ALL_QUANTS=ON.
GPU: RTX 3070 Laptop (8 GB, cc 8.6). FlashAttention enabled in all runs (`-fa on`).

## 1. Generation smoke (llama-cli, deterministic)

Command for every policy (identical arguments):

```
llama-cli -m MODEL -ngl 99 -c 2048 --kv-cache-policy <policy>.json \
  -f /tmp/opencode/smoke_prompt_8k.txt -n 128 --temp 0 -s 42 -fa on \
  --no-display-prompt --single-turn --verbose
```

- Prompt: 8000-byte head of the frozen wiki held-out corpus (1891 tokens, >= 512).
- Generation: 128 tokens, temperature 0 (greedy), seed 42, identical across policies.
- Raw logs: artifacts/beyond-perplexity/runtime_smoke/{q4xq4,q4xq3_64,c2,allq2_nu}_gen.log (plain) and _gen_verbose.log (full).

| policy | policy sha256 | resolved K (all 36 layers) | resolved V (all 36 layers) | KV buffer | rc | nan/inf |
|---|---|---|---|---|---|---|
| q4xq4 | 717ff660... | q4_0 | q4_0 | 81.00 MiB | 0 | none |
| q4xq3_64 | 3c24b621... | q4_0 | q3_0_64 | 69.75 MiB | 0 | none |
| c2 | 88e3b00a... | q4_0 | q4_0,q2_nux6,q4_0,q2_nu,q3_0_64,q2_nux4,q3_0_64,q2_nu,q3_0_64,q2_nux3,q4_0,q2_nux2,q4_0,q2_nux10,q3_0_64,q2_nu | 64.00 MiB | 0 | none |
| allq2_nu | 7e7c054b... | q4_0 | q2_nu | 60.75 MiB | 0 | none |

Every run verified:

- model loads (model screen, n_head=32, n_head_kv=8, n_embd_head_k=128, n_layer_all=36);
- `llama_context: KV cache policy: <summary>` matches the requested policy exactly (K and V per layer);
- `llama_context: flash_attn = enabled`;
- prompt evaluation completes (prompt eval time, ~3700 tok/s);
- generation completes (eval time 128 tokens, ~90 tok/s) and output is nonempty and coherent;
- no NaN/Inf in the output (the only matches for `nan|inf` are the word "information");
- no crash (exit code 0);
- cache type is the requested type, not silently Q4 (KV buffer sizes match analytical B/token x 2048 cells: 81.00/69.75/64.00/60.75 MiB).

## 2. Cache reuse / shift (llama-server)

Prompt: same 8000-byte prompt (1884 tokens), policy c2, context 2048, FlashAttention on.

Sequence of /completion requests:

1. Full prompt, n_predict=40 -> 40 tokens, stop=limit. Slot cached 1884 prompt tokens.
2. Continuation (`continue: true`, cache_prompt=true) -> 0 tokens, stop=none (prompt cached, slot reused).
3. Fresh full-prompt request, n_predict=300 -> 164 tokens then stop=limit (1884 + 164 = 2048 = context limit). `graphs reused = 163`.
4. Divergent prompt sharing the first 1000 chars, cache_prompt=true -> prompt_n=361 (prefix reused from slot), 20 tokens, `graphs reused = 181`.

Evidence of cache reuse: slot print_timing `graphs reused` counters increase across requests; prompt tokens from request 1 are not re-evaluated in requests 2-4 (prompt_n=361 instead of 1884 for the divergent prompt after prefix matching). This runtime stops generation at the context limit rather than shifting mid-generation (stop=limit at exactly 2048); the supported cache-copy/reuse path is prefix reuse across requests, which was exercised successfully with the C2 policy (mixed quantized K/V) without errors.

Server log: /tmp/opencode/server_shift.log (transient, not committed).

## 3. Found and fixed during smoke (important)

First smoke run aborted in decode with `ggml/src/ggml-cuda/fattn.cu:345: fatal error` for the three mixed-type policies. Root cause: in ggml_cuda_flash_attn_ext_vec, the dispatch cases for the new KV types (Q3_0, Q3_0_64, Q2_NU) had only been added to the `#else` (GGML_CUDA_FA_ALL_QUANTS=OFF) branch during Phase 6. The release build uses FA_ALL_QUANTS=ON (required for mixed K/V FA), and single-token decode selects the VEC FlashAttention kernel, which dispatches through the `#ifdef` branch - so mixed-type policies aborted at decode time while prompt processing (MMA kernel path) worked. Fixed by adding the same new-type cases to the ON branch (fattn.cu). After the fix all four policies run end-to-end. This fix also makes the ON-branch dispatch exactly match the 67 instance files retained by the FATTN_INSTANCE_AUDIT.
