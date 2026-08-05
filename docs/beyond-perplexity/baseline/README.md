# Clean upstream baseline (before any beyond-perplexity changes)

Base commit: e031d956797f9a10f704fa523fe5fd9b72e9015c (master, pulled 2026-08-05)
Build:      build-bp-cpu-debug (Debug, LLAMA_BUILD_TESTS=ON, CPU backend only)

## Test results

`ctest --test-dir build-bp-cpu-debug --output-on-failure -j8`: 56/58 passed.

| Test | Result | Note |
|------|--------|------|
| test-quantize-fns   | PASS | |
| test-quantize-stats | FAIL | env-gated: needs models/7B/ggml-model-f16.gguf (not downloaded, ~13 GB) |
| test-backend-ops    | PASS | "OK, 1/1 backends passed" |
| test-tokenizers-ggml-vocabs | FAIL | env-gated: ggml-org/vocabs requires git-lfs; machine lacks git-lfs, LFS pointer files only |
| test-model-resolution | SEGFAULT | env-gated: same missing-vocabs cause |

All other tests passed (53/58 total pass including quant, sampler, chat, jinja, arch tests).

## CUDA build

build-bp-cuda-release (Release, GGML_CUDA=ON, LLAMA_BUILD_TESTS=ON) building in
background; results appended when complete. First attempt failed with
"No space left on device" during nvcc object emission (transient disk pressure),
rebuild started with -j8.

## Environment summary (see BUILD_ENVIRONMENT.md)

- EndeavourOS (Arch), kernel 7.1.4, i7-12700H (14c/20t), 15 GiB RAM
- RTX 3070 Laptop 8 GiB, driver 610.43.03, CUDA 13.3, cmake 4.4.0, gcc 16.1.1
- Model: Qwen3-4B-Q4_K_M.gguf sha256 7485fe6f... (verified, matches paper targets)
