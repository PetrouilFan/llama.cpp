# Limitations

This document lists the known limitations of the clean current-upstream
implementation and of the verification reported in this repository.

## Runtime limitations

- **Uniform `q2_0` KV cache is not supported in the CUDA path.** The paper
  evaluated a uniform-Q2 variant (`allq2_uniform`). The clean branch keeps
  uniform Q2 semantics in the format tables (CPU), but the CUDA KV-cache path
  has no SET_ROWS support for `GGML_TYPE_Q2_0`, so a policy that requests
  `q2_0` for K or V is rejected at context creation with an explicit error.
  Consequently the equal-storage NU-Q2 vs uniform-Q2 PPL comparison cannot be
  reproduced at runtime; the frozen paper numbers are used instead
  (EQUIVALENCE_REPORT.md).
- **Quantized V requires FlashAttention.** Context creation fails if a policy
  selects a quantized V type while FA is disabled. This matches the legacy
  behavior and the paper path.
- **FlashAttention mixed K/V pairs** are supported only for the audited
  dispatch grid (FATTN_INSTANCE_AUDIT.md). Combinations without a dispatch
  case (e.g. q3_0_64 K with q4_0 V) do not have FA support and are outside the
  production policies.
- **MMQ kernels are not provided** for the new types. MUL_MAT on the
  non-FlashAttention path uses MMVQ (vector dot) kernels, which are provided.
- **No CPU flash attention for quantized KV** beyond what upstream provides;
  policy evaluation on CPU-only setups with quantized V fails by design.

## Verification limitations

- **Context truncation, not shifting.** The tested upstream runtime stops
  generation at the context limit; the smoke test exercised cache prefix
  reuse (multi-request), not mid-generation shifting.
- **Sanitizer runs disable CUDA graphs** (`GGML_CUDA_DISABLE_GRAPHS=1`);
  normal runtime keeps the standard graph setting. The report documents this.
- **Single GPU, single cc (8.6), CUDA 13.3.** All GPU verification was on an
  RTX 3070 Laptop. Other architectures/builds were not executed.
- **Absolute NLL/PPL drift vs the paper.** The clean branch sits on a newer
  upstream base (e031d9567); absolute logits may move. The acceptance gates
  are defined on policy-effect differences relative to each implementation's
  own Q4K-Q4V baseline, not on absolute values.
- **GSM8K smoke subset** is the first 200 examples of the openai/gsm8k test
  split (frozen in the paper revision4 records), not a random subset.
- **HumanEval/MBPP were not rerun** on the clean branch. They are not
  required by the acceptance gates; the paper numbers remain in
  `reference/06_PAPER_IMPLEMENTATION_TARGETS.json`.

## Waived gates

- **Uniform-Q2 negative control (WAIVED).** The paper's negative control
  compares NU-Q2 against uniform-Q2 at equal storage. The clean CUDA runtime
  cannot express uniform `q2_0` KV (no SET_ROWS support, see above), so this
  comparison is not reproducible at runtime. The release waives the gate and
  relies on the frozen paper measurements instead: GSM8K strict 58.53 (NU) vs
  31.01 (uniform), MBPP 48.00 vs 30.80, TruthfulQA bleu 11.1249 vs 7.7857,
  and the llama-3.2-1b held-out PPL 13.0153/19.0996 (uniform all-V) vs
  11.8001/17.3042 (Q4xQ4 baseline). The waiver is recorded in
  RELEASE_MANIFEST.json (`waived_gates`).

## Repository limitations

- Model files, frozen corpora, build directories, raw logs, profiler traces,
  and legacy bundles are not committed (see RELEASE_MANIFEST.json for what
  is).
- The examples under `examples/kv-cache-policies/` are the verification
  policies (n_kv_heads 8); the frozen legacy reference policies in
  `tests/beyond-perplexity/reference/policies/` carry the historical
  `n_kv_heads: 4` metadata bug and are used for parse-level golden tests
  only.
