# FlashAttention instance file audit

Date: 2026-08-05
Build: `build-bp-cuda-release`, GGML_CUDA=ON, GGML_CUDA_FA=ON, GGML_CUDA_FA_ALL_QUANTS=ON, CUDA 13.3, RTX 3070 Laptop (cc 8.6)

## What these files are

`ggml/src/ggml-cuda/template-instances/fattn-vec-instance-<K>-<V>.cu` instantiates
`ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_K, GGML_TYPE_V>` for D in
{64, 128, 256}. They are pulled in by the CMake `GLOB` when
`GGML_CUDA_FA_ALL_QUANTS=ON` (see `ggml/src/ggml-cuda/CMakeLists.txt`).

The only consumer is the dispatch switch in `ggml_cuda_flash_attn_ext_vec`
(`ggml/src/ggml-cuda/fattn.cu`, `FATTN_VEC_CASES_ALL_D` macro calls at
lines 265-342). A (K, V) pair is reachable at runtime if and only if it has a
case in that switch.

## Instance inventory

| set | count |
|-----|-------|
| instance files in template-instances/ | 100 |
| unique (K, V) pairs with a dispatch case | 67 |
| dispatch pairs without an instance file (link failure) | 0 |
| instance files without a dispatch case (unreachable) | 33 |

The 100 files are the 10x10 cross product of
{F16, Q3_0, Q3_0_64, Q2_NU, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16}.
Upstream base e031d9567 shipped 49 (the 7x7 old-type grid
{F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16}); the beyond-perplexity branch added
the 51 files covering the three new KV types. 18 of those 51 also gained a
dispatch case, mirroring the legacy FA matrix; the other 33 have no dispatch
case and are dead code.

## Classification columns

- required_by_production_policy: paper 4B/17b policy files
  (q4xq4, q4xq3_64, c2, b2, 28q2_8q4, 32q2_4q4, allq2_nu, q8xq8, random_23,
  *_17b): K in {q4_0, q8_0}, V in {q4_0, q3_0_64, q2_nu}.
- required_by_ablation: key_precision (K in {q8_0, q4_0, q3_0_64}, V=q4_0) and
  random V-layer policies; V in {q4_0, q3_0_64, q2_nu}.
- required_by_test_suite: pairs exercised by test-backend-ops
  `test_flash_attn_ext`: same-type {F16, BF16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0}
  plus mixed (Q8_0,Q4_0), (Q4_0,F16), (Q4_0,Q8_0), (F16,Q4_0).
- reachable_at_runtime: has a `FATTN_VEC_CASES_ALL_D` dispatch case.
- action: KEEP (dispatch case present) or PRUNE (dead).

Note: Q1_0, Q2_0, IQ4_NL, F32 test cases exist in test-backend-ops but map to
no instance type in the 10-type grid (Q1_0/Q2_0 have no CUDA FA path; F32 KV
uses the plain fa kernels), so they impose no instance requirement.

## Decision

Prune the 33 instance files that have no dispatch case. They cannot be
reached from any code path: the dispatch switch is the sole call site, and it
is compiled per (K, V) pair.

The 67 kept instances are all required for the build to link: each is
referenced by an explicit `FATTN_VEC_CASES_ALL_D` case. Of those, 26 are also
required by production policies and/or the ablation, 7 by the test suite
alone (same-type old grid pairs not used by any paper policy), and the rest
are the upstream 7x7 old-type grid inherited from upstream's
FA_ALL_QUANTS=ON build (removing them would deviate from the pinned upstream
build/link architecture).

| category | count |
|----------|-------|
| KEEP - production policy | 4 |
| KEEP - ablation (key_precision / random V) | 3 |
| KEEP - test suite | 7 |
| KEEP - upstream old-type grid + new-type grid parity | 53 |
| PRUNE - no dispatch case (dead) | 33 |

Production pairs: (q4_0,q4_0), (q4_0,q3_0_64), (q4_0,q2_nu), (q8_0,q8_0).
Ablation pairs beyond production: (q8_0,q4_0), (q3_0_64,q4_0), (q3_0_64,q3_0_64).
Note: (q3_0_64,q4_0) has no dispatch case in clean (mirrors legacy, which also
aborted on that pair), so the key_precision K=q3_0_64 cell is documented as
not reproducible in clean.

## Binary size and build time

Pristine baseline build (upstream e031d9567 with 49 instances) is not
available on this machine, so the before/after comparison is the current
100-file build vs the pruned 67-file build.

| state | instance files | libggml-cuda.so.0.18.1 size | build time (incremental, -j 8) |
|-------|----------------|-----------------------------|--------------------------------|
| before | 100 | 73,657,152 B (70.2 MiB) | n/a (full rebuild not timed) |
| after | 67 | 63,043,120 B (60.1 MiB) | relink only: 4.4 s (reconfigure 0.5 s) |

Size delta: -10,614,032 B (-14.4 %). A full clean build was not timed in
either state, so the build-time delta is reported as the relink cost only.
Stale .o files from pruned sources remain in the CMake object dir but are not
part of the build (CMake re-globbed the source list); the linked library
contains only the 67 kept instances.

## Procedure

1. CSV matrix generated from fattn.cu dispatch switch and template-instances/
   directory listing (script: /tmp/opencode/build_matrix.py).
2. Delete the 33 PRUNE instance files (listed in FATTN_INSTANCE_AUDIT.csv).
3. Reconfigure (CMake GLOB must re-run) and rebuild.
4. Re-run test-backend-ops CUDA0 suite: 14109/14109 passed (FA_ALL_QUANTS=ON
   exercises more FA combos than the earlier 13096/13096 OFF build).
5. Re-measured libggml-cuda.so and relink time (see table above).
6. Runtime check: llama-perplexity with the C2 policy (`--kv-cache-policy
   c2.json`) reproduces byte-identical per-chunk NLL (`NLL[1] sum=2462.586418
   count=1023 mean=2.407220`) after pruning, confirming the kept K/V FA
   instances (q4_0-q4_0, q4_0-q3_0_64, q4_0-q2_nu) resolve exactly as before.

See FATTN_INSTANCE_AUDIT.csv for the per-file matrix.
