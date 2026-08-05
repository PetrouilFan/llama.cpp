# NU-Q2 vector-dot deviation from legacy

Date: 2026-08-05

## 1. Summary

The clean implementation's `vec_dot_q2_nu_q8_1` CUDA kernel accumulates with
the non-uniform Q2 codebook (`q2_nu_levels = {-0.619326, -0.186363, 0.186363,
0.619301}`). The legacy implementation used the uniform-style expression
`(qval - 1)` for what it called `vec_dot_q2_0_q8_1`, which did not match the
non-uniform codebook that produced the data.

Conclusion: the legacy expression is a latent numeric wart that was never
exercised on any historical paper path. The clean correction is retained.

## 2. Runtime call paths analyzed

`vec_dot_q2_nu_q8_1` is a MMVQ kernel referenced by `get_vec_dot_q_cuda`
only for `GGML_TYPE_Q2_NU`, and is invoked exclusively on the non-FlashAttention
`Q`^T`K` path (MUL_MAT with quantized K). The cache access paths are:

- KV writes: SET_ROWS with `quantize_f32_q2_nu_block` (set-rows.cu).
- KV reads under FlashAttention: fused fattn kernels (fattn.cu).
- KV reads under non-FlashAttention: GET_ROWS/CPY for K, and rejects quantized
  V by design (quantized V requires FlashAttention).

None of the write/FA/copy paths call vec_dot. Only non-FA MUL_MAT with a
quantized K type can reach `vec_dot_q2_nu_q8_1`.

## 3. Evidence

Instrumentation was added to the dispatch in `mul_mat_vec_q_switch_type`
(mmvq.cu), counting per-type MMVQ kernel launches, gated by the environment
variable `BP_TRACE_VECDOT`. Runs used the paper model (Qwen3-4B-Instruct
Q4_K_M, sha256 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5)
and the paper wiki held-out file.

| Run | Flash attention | K/V policy | MMVQ launches | Result |
|-----|----------------|------------|---------------|--------|
| C2 policy (q4_0 K, q2_nu/q3_0_64/q4_0 V) | auto (on) | c2.json | 0 | PPL 14.6439, full run OK |
| q2_nu K, f16 V control | off | q2k_f16v.json | q2_nu=36 | runs, PPL 3093.76 |
| q2_nu K, f16 V control | on | q2k_f16v.json | none | runs, PPL 3153.79 |

The FA path launches zero vec_dot kernels. The non-FA K-side path launches
exactly one per layer. Instrumentation was removed before commit.

## 4. Did the legacy wart affect the historical paper path?

No. Documentary evidence:

- All legacy paper PPL cells (q8xq8, q4xq4, q4xq3_64, 28q2_8q4, 32q2_4q4, c2,
  b2, allq2_nu, random_23) ran with default FlashAttention resolution, which
  resolved to enabled on GPU. The legacy fattn dispatch table covered every
  (K,V) combination used by those policies.
- Legacy rejects quantized V without FlashAttention, so the V-side vec_dot
  path was structurally unreachable for the paper's V-quantized cells.
- No paper cell used a quantized K type on a non-FA path.
- GSM8K/HumanEval/MBPP/RULER cells ran through llama-server, also FA.

Therefore `vec_dot_q2_0_q8_1` (the uniform `(qval - 1)` expression) never
produced output for any historical paper number.

## 5. Does the clean correction change logits, NLL, or task behavior?

Not on the paper path. Since the paper path is entirely FlashAttention, the
corrected kernel is not invoked there. On the non-FA K-side path (reachable
only via a `--kv-cache-policy` file with a quantized K type and `-fa off`),
the corrected kernel is the only implementation present, so no comparison.

## 6. Why the corrected version is retained

- It matches the clean CPU reference: `dequantize_row_q2_nu` in ggml-quants.c
  uses the same non-uniform levels; `vec_dot_q2_nu_q8_1` and the CPU reference
  agree, as verified by the test-backend-ops MMVQ matrix (all OK).
- The clean `vec_dot_q2_0_q8_1` (uniform) also matches its own CPU reference
  (`dequantize_row_q2_0` is uniform in both clean and legacy).
- The paper `allq2_nu` numbers were produced by a NU-aware legacy build; the
  uniform expression would not reproduce them even there.
- Copying a known mathematical error purely for byte-for-byte compatibility
  is disallowed. The correct behavior is preserved.