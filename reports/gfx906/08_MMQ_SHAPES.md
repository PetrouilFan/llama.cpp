# gfx906 MMQ Shape Analysis

## Model architectures

### LFM2.5-2.6B (lfm2, 30 layers, ctx 128k)

| dim | value |
| --- | --- |
| embedding_length | 2048 |
| feed_forward_length | 10752 |
| head_count | 32 |
| head_count_kv | per-layer array: MHA layers use 32 kv heads, GQA layers use 8 |
| head size D | 64 |

Weight matrices per layer (ggml layout src0 = [ne00=K, ne01=N]):

| tensor | K (ne00) | N (ne01) |
| --- | --- | --- |
| attn_q / attn_k / attn_v / attn_output | 2048 | 2048 |
| ffn_gate / ffn_up | 2048 | 10752 |
| ffn_down | 10752 | 2048 |

### qwen2.5-1.5B (qwen2, 28 layers, ctx 32k)

| dim | value |
| --- | --- |
| embedding_length | 1536 |
| feed_forward_length | 8960 |
| head_count | 12 |
| head_count_kv | 2 (GQA ratio 6) |
| head size D | 128 |

| tensor | K (ne00) | N (ne01) |
| --- | --- | --- |
| attn_q / attn_k / attn_v / attn_output | 1536 | 1536 |
| ffn_gate / ffn_up | 1536 | 8960 |
| ffn_down | 8960 | 1536 |

## How MMQ tiles these shapes (gfx906, current gfx906-safe branch)

MMQ launch (mmq.cuh): nty = ceil(nrows_x/I), ntx = ceil(ncols_max/J), where
nrows_x = src0 ne00 = K, ncols_max = src1 ne11 = prompt length M, I = 128,
J from the gfx906 table (Q8_0: 8..128 step 8, nthreads 512).

ncols_max = prompt tokens (M) drives J selection: J_best = smallest J in the
table that satisfies LDS budget; the gfx906 small-batch clamp forces
J_best <= 64 when M <= 128.

### Blocks per launch (Q8_0, gfx906, before/after tuning)

M = pp size, K/N per tensor family:

| M | J_best | ntx (M/J) | nty attn (K=2048) | nty ffn (K=10752) | attn blocks | ffn blocks |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 64 (clamped) | 2 | 16 | 84 | 32 | 168 |
| 256 | 128 | 2 | 16 | 84 | 32 | 168 |
| 512 | 128 | 4 | 16 | 84 | 64 | 336 |
| 1024 | 128 | 8 | 16 | 84 | 128 | 672 |
| 2048 | 128 | 16 | 16 | 84 | 256 | 1344 |

Note: ffn_down (K=10752) dominates block count at every pp size, so the
batch tail-effects that Stream-K was targeting are most pronounced on
ffn_down.

### qwen2.5-1.5B equivalents

| M | J_best | ntx | nty attn (K=1536) | nty ffn (K=8960) | attn blocks | ffn blocks |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 64 | 2 | 12 | 70 | 24 | 140 |
| 512 | 128 | 4 | 12 | 70 | 48 | 280 |
| 2048 | 128 | 16 | 12 | 70 | 192 | 1120 |

## Selected MMQ config per quant (gfx906-safe)

| quant | nthreads | occupancy | I | J range | stream_k |
| --- | --- | --- | --- | --- | --- |
| Q8_0 | 512 | 2 | 128 | 8..128 | false |
| Q4_0 | 256 | 2 | 128 | 8..64 | false |
| Q4_K | 256 | 2 | 128 | 8..64 | false |
| others | 256 | 2 | 128 | 8..64 | false |

## Observed tokens/s by shape (LFM2.5-2.6B Q8_0, 5 reps)

| pp | upstream | gfx906-safe | delta |
| --- | --- | --- | --- |
| 128 | 1142 | 2147 | +88% |
| 256 | 1180 | 2300 | +95% |
| 512 | 1365 | 2362 | +73% |
| 1024 | 1377 | 2377 | +73% |
| 2048 | 1371 | 2361 | +72% |

## Conclusion

The Q8_0 gain is broadly applicable: it is not shape-specific to LFM.
qwen2.5-1.5B (dense, D=128, GQA ratio 6, different N/K ratios) shows
+78% at pp512 (2414 -> 4288 t/s). The wins come from (1) 512 threads =
8 wave64 wavefronts/block for better latency hiding on the LDS-bound
1-block/CU kernel, (2) J up to 128 halving x-tile reload traffic,
(3) 8 threads/row tile loading, (4) k-unroll 2 in vec_dot. No shape
special-casing beyond the ncols<=128 clamp is needed.
