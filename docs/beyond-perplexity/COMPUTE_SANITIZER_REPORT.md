# Compute Sanitizer report

Date: Wed Aug 05 2026
Tool: NVIDIA Compute Sanitizer 2026.2.1.0 (build 38334959), part of CUDA 13.3
GPU: NVIDIA GeForce RTX 3070 Laptop GPU (cc 8.6, 8 GiB)
Host: build-bp-cuda-release, GGML_CUDA=ON, GGML_CUDA_FA=ON, GGML_CUDA_FA_ALL_QUANTS=ON
Model: Qwen3-4B-Instruct Q4_K_M (sha256 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5)
llama-cli sha256: cf29696f38b771a984037f8e417530c51562335a34574b4b529e8f6deebd4db6
llama-server sha256: 59150a181708d43037ec90737d8625b691c68a8aec82f290cc9c8f43cebaaf76

## Method

Real llama-cli and llama-server executions were traced with `compute-sanitizer`
under `--tool memcheck` and `--tool initcheck`, with `--error-exitcode 1`.

CUDA graphs were disabled for the sanitizer runs only via
`GGML_CUDA_DISABLE_GRAPHS=1` (the documented environment switch in
ggml-cuda/common.cuh:1279). The normal runtime keeps its standard graph
setting (USE_GRAPHS=1 in system_info); graphs were not changed in any other
code path. All other inference arguments match the runtime smoke runs.

Every run used a real 1009-token prompt (head of the frozen held-out wiki
corpus, identical across runs) and greedy decoding (temperature 0, seed 42,
flash attention on). The same binary was used with the three mixed-type
policies; q4xq3_64, c2 and q4k-all-nu-q2 all exercise the quantized KV write
path (SET_ROWS packed writes), quantized KV reads in flash attention
(including the mixed-type VEC decode kernels), and token generation.

Prompt/cache shift and reuse were exercised on the llama-server path (c2
policy): two completions were traced, the second with cache_prompt=true and an
identical prompt, which reuses the populated KV cache slot (prompt_n=1, i.e.
only one new token was processed while 19 prompt tokens were served from the
cache).

## Results

All runs completed with rc=0 and zero sanitizer findings:

| run | tool | policy | executable | result |
|-----|------|--------|------------|--------|
| memcheck q4xq3_64 | memcheck | q4k-q3v | llama-cli | 0 errors |
| memcheck c2 | memcheck | c2 | llama-cli | 0 errors |
| memcheck allq2_nu | memcheck | q4k-all-nu-q2 | llama-cli | 0 errors |
| initcheck q4xq3_64 | initcheck | q4k-q3v | llama-cli | 0 errors |
| initcheck c2 | initcheck | c2 | llama-cli | 0 errors |
| initcheck allq2_nu | initcheck | q4k-all-nu-q2 | llama-cli | 0 errors |
| memcheck server c2 (reuse) | memcheck | c2 | llama-server | 0 errors |

Example sanitizer summary (identical for all runs):

```
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
```

## Paths exercised

- cache allocation: KV buffer allocation at context creation (81.00 / 69.75 /
  64.00 / 60.75 MiB per policy at 2048 cells);
- packed writes: quantized KV cache writes (SET_ROWS -> quantize_f32_q2_nu_block,
  quantize_f32_q3_0_block, quantize_f32_q3_0_64_block) on every policy;
- reads: GET_ROWS / CPY reads of the quantized cache and flash attention reads;
- mixed flash attention: all three mixed K/V type policies run through the
  flash attention kernels, including the decode-path VEC kernels for
  (q4_0,q3_0_64), (q4_0,q2_nu) and (q2_nu,q3_0_64) mixed pairs;
- generation: greedy decode over 4-10 tokens in every run;
- cache reuse: llama-server prefix reuse with a populated mixed-type KV slot.

## Command lines

llama-cli (policy p in {q4xq3_64, c2, allq2_nu}):

```
GGML_CUDA_DISABLE_GRAPHS=1 compute-sanitizer --tool memcheck --error-exitcode 1 \
  --log-file san_memcheck_<p>.log \
  ./build-bp-cuda-release/bin/llama-cli -m <model> -ngl 99 -c 2048 \
  --kv-cache-policy <p>.json -f san_prompt.txt -n 4 --temp 0 -s 42 -fa on \
  --no-display-prompt --single-turn
```

(initcheck identical with `--tool initcheck`.)

llama-server (cache reuse):

```
GGML_CUDA_DISABLE_GRAPHS=1 compute-sanitizer --tool memcheck --error-exitcode 1 \
  --log-file san_memcheck_server_c2.log \
  ./build-bp-cuda-release/bin/llama-server -m <model> -ngl 99 -c 2048 \
  --kv-cache-policy c2.json --port 8125 -fa on -dt 0
```

followed by two /completion requests (see above).

## Conclusion

Zero errors on all selected production paths for q4k-q3v, c2 and
q4k-all-nu-q2 under both memcheck and initcheck, including the cache-reuse
server path. No sanitizer findings remain on the new KV cache types.
