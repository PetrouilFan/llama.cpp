# Release Checklist - Completed

Tag: beyond-perplexity-v0.1.0-paper
Feature head: 3b32e6af5b39da1539338abe09cd2a9dc743b3a3
Upstream base: e031d956797f9a10f704fa523fe5fd9b72e9015c
Date: 2026-08-06

Every release gate is resolved to one of PASS / WAIVED / NOT APPLICABLE /
FAIL. Evidence paths are relative to the repository root.

## Build and identity

- [x] PASS Tag points to the feature head commit (tag = HEAD = branch).
- [x] PASS Clean tagged build in a fresh directory (build-bp-release-verify)
      with the documented release options:
      cmake -S . -B build-bp-release-verify -DGGML_CUDA=ON -DGGML_CUDA_FA=ON
      -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
      Compiler: nvcc 13.3.73 (CUDA 13.3), gcc 16.1.1.
- [x] PASS Binary hashes recorded (RELEASE_MANIFEST.json, SHA256SUMS):
      llama-cli d32436bd..., llama-server 7f5cda41...,
      llama-perplexity 2d5a7217..., libggml-cuda 0ee149e3...,
      test-backend-ops 6e5f2de4..., test-beyond-perplexity-quants
      ccbd3b32..., test-kv-cache-policy 61497acb....
- [x] PASS Model sha256 recorded (7485fe6f...4fdf5, Qwen3-4B-Instruct Q4_K_M).

## Quantization correctness

- [x] PASS test-beyond-perplexity-quants on the tagged build (q2_nu, q3_0,
      q3_0_64, uniform q2 vectors, all records).
- [x] PASS test-kv-cache-policy on the tagged build (inline policies,
      negative cases).
- [ ] PENDING CUDA backend ops for the new types (q3_0, q3_0_64, q2_nu) on
      the tagged build. Blocked on a driver-level GPU hang (nvidia driver
      "Unknown Error", no CUDA device visible); requires a machine reboot.
      Note: passed previously on the pre-tag verification build
      (14109/14109 backend tests, q3_0/q3_0_64/q2_nu filtered run).
- [x] PASS Quantized V requires FA by design; context creation fails with an
      explicit error otherwise (policy runtime verification).

## Policy runtime verification

- [x] PASS B/tok matches for all four production policies (q4xq4 41472,
      q4xq3_64 35712, c2 32768, allq2_nu 31104).
- [x] PASS Nine negative policy cases fail at context creation with explicit
      errors (architecture, n_kv_heads, head_dim, model sha256, codebook
      sha256, q2_0 rejection, etc.).
- [x] PASS Policy identity hashes (RELEASE_MANIFEST.json, examples/
      kv-cache-policies/).

## Runtime smoke

- [ ] PENDING llama-cli end-to-end generation for all four policies on the
      tagged build, FA on, 1891-token prompt, 128 greedy tokens (bit-stable;
      cache prefix reuse via llama-server). Blocked on the GPU reboot noted
      above. Previously verified on the pre-tag verification build.
- [x] PASS Compute sanitizer: 7 runs (memcheck + initcheck x3, server reuse),
      0 errors. Note: runs used GGML_CUDA_DISABLE_GRAPHS=1.

## Equivalence

- [x] PASS Clean vs legacy policy-effect difference 0.000000 <= 0.002 on the
      768-chunk smoke matrix; GSM8K-200 identical (0.645/0.685/0.61).
- [x] PASS Full held-out PPL (ctx 4096, wiki/owt) exact vs paper:
      q4xq4 12.2919/18.4667, q4xq3_64 12.3452/18.5276, c2 12.2435/18.3585.
- [x] PASS GSM8K-1319 strict exact vs paper: 0.6543/0.6710/0.6262.
- [x] WAIVED Uniform-Q2 negative control (no SET_ROWS for q2_0 in CUDA);
      frozen paper numbers used. RELEASE_MANIFEST.json `waived_gates`,
      LIMITATIONS.md.

## Systems

- [x] PASS VRAM savings 4K/16K/32K match paper exactly (34/136/272 MiB).
- [x] PASS 64K analytical KV saving exactly 544 MiB (542.94 MiB at 65,408
      occupied slots); observed total-process VRAM difference 688 MiB is
      documented as process-level reading, not claimed as KV saving.
- [x] PASS C2 decode overhead within the <= 4% gate at the measurable
      contexts (2.38%/3.39%/2.82% at 4K/16K/32K; C2 slightly slower than
      q4xq4).
- [x] NOT APPLICABLE 64K decode gate: upstream slot cap (n_ctx_train 40960,
      tools/server/server-context.cpp, commit ab49f094d2 in the pinned base)
      makes 64K decode unmeasurable; documented in SYSTEMS_EQUIVALENCE_
      REPORT.md.

## Repository hygiene

- [x] PASS No PR submitted; fork push is user-authorized and documented.
- [x] PASS Model files, corpora, build directories, raw logs, profiler
      traces, legacy bundles not committed (RELEASE_MANIFEST.json
      `not_committed`).
- [x] PASS Legacy repository untouched (read-only reference).
- [x] PASS git status clean at the tagged commit.
