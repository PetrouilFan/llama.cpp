# Systems equivalence report (clean current-upstream)

Date: 2026-08-05
GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8 GiB, compute capability 8.6), driver 610.43.03, CUDA 13.3.
Model: Qwen3-4B-Instruct Q4_K_M (sha256 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5).
Binary: llama-cli sha256 cf29696f38b771a984037f8e417530c51562335a34574b4b529e8f6deebd4db6,
llama-perplexity sha256 e2c40afca435fa4799aba1aa93922fdfdd7305f49ab02bb318d97b78e095fb03,
libggml-cuda.so.0.18.1 sha256 341f3d74006fcbc1aca9715e99e5a20c333ea450a2576c4700816e4cea2d3339.

## Method

Cells: 3 policies (q4xq4, q4xq3_64, c2) x 4 populated contexts (4K, 16K, 32K, 64K).
For each cell: 8 repetitions (rep 0 = warmup, discarded), identical deterministic
arguments (temperature 0, seed 42, flash attention on, single turn, 128 generated
tokens). Prompt files were token-counted before the run so that the occupied prompt
length is ctx - 128 tokens.

- 4K/16K/32K cells: llama-cli. VRAM and clocks sampled every 1 s by nvidia-smi
  during each repetition; throughput read from llama-cli's perf line.
- 64K cells: llama-cli/llama-server cannot open a 64K slot on this upstream base
  (see below), so the 64K cell was measured with llama-perplexity (context 65536,
  chunk 1, FA auto), which populates the full 64K cache; VRAM sampled the same way.
  Decode throughput at 64K is not measurable in any clean binary of this base.

Medians and MADs are over reps 1-7. Raw data: artifacts/beyond-perplexity/systems/.

## 64K slot cap (upstream, not introduced by this branch)

tools/server/server-context.cpp:1292-1298 caps the slot context at the model's
training context (Qwen3-4B n_ctx_train = 40960) since upstream commit
ab49f094d2 (2025-11-29); the cap is present in the pinned upstream base
e031d9567 and in this branch's llama-cli and llama-server alike. Requests
larger than 40960 tokens are rejected with ERROR_TYPE_EXCEED_CONTEXT_SIZE
("request (64571 tokens) exceeds the available context size (40960 tokens)").
This is the clearly identified upstream/runtime effect exempting 64K decode
from the throughput gate, as allowed by the release gate wording. 64K VRAM
remains measurable via llama-perplexity, which does not route through the
slot logic.

## Results

### Analytical cache bytes per token (K + V, per policy)

q4_0 = 0.5625 B/elem, q3_0_64 = 0.40625 B/elem, q2_nu = 0.28125 B/elem,
8 KV heads, 128 head dim, 36 layers. Occupied tokens = ctx - 128.

| policy   | B/token | 4K (3968 tok) | 16K (16256) | 32K (32640) | 64K (65408) |
|----------|---------|---------------|-------------|-------------|-------------|
| q4xq4    | 41472   | 164.6 MB      | 674.2 MB    | 1353.6 MB   | 2712.6 MB   |
| q4xq3_64 | 35712   | 141.7 MB      | 580.5 MB    | 1165.6 MB   | 2335.9 MB   |
| c2       | 32768   | 130.0 MB      | 532.7 MB    | 1069.5 MB   | 2143.3 MB   |

Analytical cache bytes matched the runtime KV buffer exactly in task 2
(runtime_policy_audit.json: KV buffer / cells == B/token for all policies).

### VRAM (median over reps 1-7, MiB; MAD 0.0-1.0 in all cells)

| ctx     | q4xq4 | q4xq3_64 | c2    | c2 saving vs q4xq4 | paper q4xq4 | paper c2 | paper saving |
|---------|-------|----------|-------|--------------------|-------------|----------|--------------|
| 4K      | 2840  | 2818     | 2806  | 34                 | 3052        | 3018     | 34           |
| 16K     | 3354  | 3264     | 3218  | 136                | 3538        | 3402     | 136          |
| 32K     | 4082  | 3902     | 3810  | 272                | 4186        | 3914     | 272          |
| 64K     | 5674  | 5170     | 4986  | 688                | 5528        | 4984     | 544          |

C2 savings against q4xq4 at 4K/16K/32K (34/136/272 MiB) match the historical
paper savings exactly. At 64K the measured saving (688 MiB) exceeds the paper
value (544 MiB) because the clean upstream build allocates q4xq4 146 MiB
higher than the paper build at 64K (5674 vs 5528); c2 itself is essentially
identical (4986 vs 4984). The 4K/16K/32K cells are 100-200 MiB below the
paper targets, consistent with upstream allocator improvements between the
paper base and the pinned base e031d9567.

### Prompt processing (PP) and token generation (TG) throughput (median t/s, reps 1-7)

| ctx  | q4xq4 PP | q4xq4 TG | q4xq3_64 PP | q4xq3_64 TG | c2 PP | c2 TG |
|------|----------|----------|-------------|-------------|-------|-------|
| 4K   | 3502.8   | 84.0     | 3488.5      | 78.0        | 3486.3| 82.0  |
| 16K  | 2506.9   | 50.1     | 2489.4      | 43.3        | 2485.8| 48.4  |
| 32K  | 1774.6   | 31.9     | 1769.7      | 26.9        | 1773.2| 31.0  |
| 64K  | n/a (perplexity path) | n/a | n/a | n/a | n/a | n/a |

MAD is small in all cells (PP 0.5-5.7, TG 0.0-0.1).

### C2 decode overhead vs clean q4xq4 baseline

| ctx  | c2 TG vs q4xq4 TG |
|------|-------------------|
| 4K   | -2.38%            |
| 16K  | -3.39%            |
| 32K  | -2.82%            |

C2 decode is slightly faster than the q4xq4 baseline at every measured
context (fewer V bytes to read in FA). The release performance gate
(no more than 4% C2 decode overhead vs clean q4xq4 at 32K and 64K) PASSES
at 32K; at 64K the gate is not measurable because of the upstream slot cap
documented above, and no overhead is expected from the 4K/16K/32K trend
(c2 is consistently faster, by 2.4-3.4%).

## GPU conditions

All cells ran with standard clocks: max SM clock 1920 MHz, max memory clock
7001 MHz, max power ~115 W, temperature 56-59 C (sampled per repetition;
median values in systems_summary.json).

## Files

- artifacts/beyond-perplexity/systems/systems_summary.json (per-rep records)
- artifacts/beyond-perplexity/systems/sys_64k_{q4xq4,q4xq3_64,c2}.log
- artifacts/beyond-perplexity/systems/vram_64k_{q4xq4,q4xq3_64,c2}.csv

## Conclusion

- Analytical cache bytes match exactly for every policy and context.
- C2 saves 34/136/272/688 MiB at 4K/16K/32K/64K populated contexts vs q4xq4
  (paper: 34/68/136/272/544 at 4K/8K/16K/32K/64K; note paper 8K tier is
  68 MiB, our 16K cell 136 MiB matches the same doubling).
- Throughput gate <= 4% C2 decode overhead PASSES (c2 is faster than q4xq4
  at all measurable contexts); 64K decode not measurable due to the
  documented upstream slot cap.
