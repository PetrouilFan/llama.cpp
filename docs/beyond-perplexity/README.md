# beyond-perplexity

Clean current-upstream implementation of static mixed-precision KV caching for
llama.cpp, reproducing the "Beyond Perplexity" paper (static mixed-precision
KV cache for llama.cpp; Q4K-Q4V, Q4K-Q3V, and C2 operating points).

## What this branch adds

- Three new ggml formats: NU-Q2 (`q2_nu`, 2-bit non-uniform, 18 B/block of
  64), Q3_0 (3-bit, 14 B/block of 32), Q3_0_64 (3-bit, 26 B/block of 64);
- a versioned, hashable, fail-closed policy-file interface
  (`--kv-cache-policy`) that selects per-layer K/V cache types;
- CUDA support for the new formats: quantization, dequantization, row
  read/write, copy, vector dot (MMVQ), and FlashAttention for the audited
  mixed K/V pairs;
- golden-vector tests, backend parity tests, runtime policy verification,
  equivalence smoke tests against the frozen legacy implementation, and
  systems measurements.

## Quick start

```sh
# build
cmake -S . -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build -j 8

# run with a policy (C2: Q4 keys, mixed Q4/Q3_0_64/NU-Q2 values)
./build/bin/llama-cli -m MODEL -ngl 99 -c 2048 \
  --kv-cache-policy examples/kv-cache-policies/c2.json -p "Hello"

# perplexity
./build/bin/llama-perplexity -m MODEL -ngl 99 -c 2047 --chunks 64 \
  --ppl-output-type 2 --kv-cache-policy examples/kv-cache-policies/q4xq4.json -f corpus.txt
```

Example policies live in `examples/kv-cache-policies/` (Q4K-Q4V, Q4K-Q3V,
C2, all-NU-Q2, and a q2_nu-K ablation).

## Documentation

- DESIGN.md - architecture of the branch
- POLICY_SCHEMA.md - policy file format and validation
- POLICY_RUNTIME_VERIFICATION.md - runtime verification of the 4 policies + 9 negative tests
- REPRODUCING_THE_PAPER.md - how to reproduce the paper measurements
- CORRECTNESS_REPORT.md - format/parity/policy correctness evidence
- EQUIVALENCE_SMOKE_REPORT.md - clean-vs-legacy smoke equivalence (PPL gate + GSM8K-200)
- EQUIVALENCE_REPORT.md - full release equivalence (PPL, GSM8K-1319, NU vs uniform)
- SYSTEMS_EQUIVALENCE_REPORT.md - memory and throughput at 4K-64K contexts
- COMPUTE_SANITIZER_REPORT.md - memcheck/initcheck results
- RUNTIME_SMOKE_REPORT.md - end-to-end llama-cli/llama-server smoke
- NUQ2_LEGACY_DEVIATION.md, Q3_COPY_FIX.md - documented deviations from legacy
- FATTN_INSTANCE_AUDIT.md - FlashAttention instance matrix and pruning
- LIMITATIONS.md - known limitations
- BUILD_ENVIRONMENT.md - build/GPU environment
- reference/ - frozen paper targets, contracts, and plans (historical evidence)

## Paper targets

| Policy | K | V | bytes/token |
|--------|---|----|-------------|
| Q4K-Q4V | q4_0 | q4_0 x36 | 41,472 |
| Q4K-Q3V | q4_0 | q3_0_64 x36 | 35,712 |
| C2 | q4_0 | q4_0 x4, q3_0_64 x4, q2_nu x28 | 32,768 |
| Q4K-all-NU-Q2V | q4_0 | q2_nu x36 | 31,104 |

The frozen paper values are in `docs/beyond-perplexity/reference/`.

## Repository layout

- `src/llama-kv-cache-policy.{h,cpp}` - policy parser and canonical hashing
- `ggml/src/ggml-cuda/` - CUDA kernels for the new formats
- `tests/test-beyond-perplexity-quants.cpp`, `tests/test-kv-cache-policy.cpp`
  - format and policy tests
- `tests/beyond-perplexity/reference/` - golden vectors and reference policies
- `examples/kv-cache-policies/` - example policy files
- `artifacts/beyond-perplexity/` - verification outputs (runtime audit, smoke
  logs, equivalence data, systems measurements)
- `docs/beyond-perplexity/` - this documentation

## License

This branch is a derivative of llama.cpp and inherits its MIT license.
