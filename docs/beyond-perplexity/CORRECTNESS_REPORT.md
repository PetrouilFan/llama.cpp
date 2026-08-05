# Correctness Report

This report summarizes the correctness evidence for the three new ggml
formats (NU-Q2, Q3_0, Q3_0_64) and the per-layer KV cache policy system on the
clean current-upstream branch.

## Formats

Packed layouts and sizes are asserted statically and against golden vectors:

- `sizeof(block_q2_nu) == 18`, `sizeof(block_q3_0) == 14`,
  `sizeof(block_q3_0_64) == 26` (static asserts in
  `tests/test-beyond-perplexity-quants.cpp`);
- CPU quantize produces byte-identical packed output to the legacy
  implementation on fixed vectors (golden vectors, memcmp-exact);
- CPU dequantize matches the legacy reference within 1e-6;
- NU-Q2 thresholds: midpoint thresholds `[-0.4028445, 0.0, 0.402832]`
  (values below -0.4028445 -> level 0 = -0.619326, etc.);
- Q3/Q3_0_64 unpacking covers every cross-byte bit position (golden
  all-code/all-position vectors).

Test: `test-beyond-perplexity-quants` (run with the reference dir as argv[1],
golden JSONs are committed under `tests/beyond-perplexity/reference/`).

## CPU/CUDA parity

- test-backend-ops on the CUDA backend after the additions:
  14,109/14,109 cases passed (includes MUL_MAT, MUL_MAT_ID, GET_ROWS,
  GET_ROWS_BACK, SET_ROWS, CPY, and FlashAttention cases for the new types);
- CPY of the new types from F32 and to F32 verified; the F32-export path uses
  a dedicated pair kernel (see Q3_COPY_FIX.md) because the legacy generic
  template was wrong for these types;
- compute sanitizer memcheck and initcheck over real llama-cli and
  llama-server runs (all three production KV mixes) report zero errors
  (COMPUTE_SANITIZER_REPORT.md).

## Policy system

The policy parser and apply-time validation are covered by:

- `test-kv-cache-policy` (parse-level positives and negatives, canonical
  hash verification against the frozen reference policies);
- nine recorded negative runtime tests, each terminating with an explicit
  error and nonzero exit, no silent fallback (POLICY_RUNTIME_VERIFICATION.md);
- runtime identity logging: resolved K/V per layer, type counts, K/V/total
  bytes per token, KV buffer sizes, FlashAttention selection
  (artifacts/beyond-perplexity/runtime_policy_audit.json).

## Quantization reference tests

The upstream `test-quantize-fns` and `test-quantize-stats` suites pass with
the new types registered (baseline outputs recorded under
`docs/beyond-perplexity/baseline/`).

## Deviations

Two intentional deviations from the legacy implementation, both documented
with evidence:

1. NU-Q2 vector dot uses the NU codebook (correct, matches CPU reference);
   the legacy uniform expression was a bug never exercised on a paper path
   (NUQ2_LEGACY_DEVIATION.md).
2. F32 export of the new formats uses a consecutive-pair kernel; the legacy
   generic template produced incorrect values for these types and was never
   exercised on GPU (Q3_COPY_FIX.md).

## Known unsupported combinations

- uniform `q2_0` KV cache: rejected at policy apply with an explicit error
  (no SET_ROWS support in the CUDA path);
- FlashAttention mixed K/V pairs outside the audited dispatch grid:
  rejected by the FA kernel selector, falling back per the upstream design
  (see FATTN_INSTANCE_AUDIT.md for the reachable matrix).
