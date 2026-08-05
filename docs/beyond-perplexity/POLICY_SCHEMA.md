# KV Cache Policy Schema

Version 1. This document defines the public policy-file format consumed by the
`--kv-cache-policy` command-line argument (llama-cli, llama-perplexity,
llama-server, and every tool that accepts `-ctk`/`-ctv`).

## Purpose

A policy file fully determines the per-layer K and V cache types for a model.
It replaces the pair of global `--cache-type-k`/`--cache-type-v` flags with an
explicit, hashable, per-layer specification.

## Canonical hash

The `policy_sha256` field is the SHA-256 of the JSON document serialized with:

- keys sorted (equivalent to Python `sort_keys=True`);
- two-space indentation (equivalent to Python `indent=2`);
- `ensure_ascii=True`;

with the keys `policy_sha256`, `created_at`, and `git_commit` removed before
hashing. Any policy file that carries a `policy_sha256` must match this hash;
otherwise parsing fails.

## Fields

| Field              | Type                 | Required | Description |
|--------------------|----------------------|----------|-------------|
| `schema_version`   | integer              | yes      | must be `1` |
| `policy_id`        | string               | no       | human-readable identifier, kept for runtime logs and validation |
| `method`           | string               | no       | free-form provenance, e.g. `nu-q2-mixed-v` |
| `method_version`   | integer              | no       | version of the method that produced the policy |
| `model_sha256`     | string (hex)         | no       | SHA-256 of the GGUF model file; if present it must match the loaded model file at apply time |
| `architecture`     | string               | no       | model architecture name, e.g. `qwen3`; if present it must match the loaded model |
| `n_layers`         | integer              | no       | must equal the length of `v_layer_types` if present |
| `n_kv_heads`       | integer              | no       | KV head count; must be nonzero, and if present must match the model at apply time |
| `head_dim`         | integer              | no       | head dimension; must be nonzero, and if present must match the model at apply time |
| `codebook`         | object               | no       | codebook levels for NU-Q2, e.g. `{"levels": [-0.619326, ...]}`; informational |
| `codebook_sha256`  | string (hex)         | no       | SHA-256 of the codebook; if present it must match the implemented NU-Q2 codebook hash `af450f1aec81b9a23e346640e27ab3150f5a9827b424750eda61d4104dea6713` |
| `k_type`           | string               | yes      | ggml type name used for K in every layer, e.g. `q4_0` |
| `v_layer_types`    | array of strings     | yes      | per-layer ggml type name for V; position `i` is layer `i` |
| `policy_sha256`    | string (hex)         | no       | canonical hash, see above |
| `created_at`       | string               | no       | excluded from the canonical hash |
| `git_commit`       | string               | no       | excluded from the canonical hash |

Supported type names are the ggml type names: `f16`, `bf16`, `q8_0`, `q4_0`,
`q4_1`, `q5_0`, `q5_1`, `q3_0`, `q3_0_64`, `q2_nu`, and any other name
recognized by ggml. KV cache storage is supported for a subset at runtime
(see CORRECTNESS_REPORT.md); policy parsing accepts the full ggml type name
space, and apply-time validation rejects unsupported types such as the legacy
uniform `q2_0` with an explicit error.

## Semantics

- `v_layer_types` is a flat positional array: index `i` is the V type of layer
  `i`. There are no ranges and no default-fill syntax, so overlapping
  assignments, missing layers, and out-of-range indices are not expressible;
  such inputs fail to parse.
- K is uniform across layers (`k_type`). Per-layer K maps are not part of
  schema version 1.
- The policy is applied after model load. The following are validated at
  apply time and abort with an explicit error (no silent fallback):

  - layer count mismatch;
  - architecture mismatch;
  - `n_kv_heads`/`head_dim` mismatch with the model;
  - `model_sha256` mismatch with the actual model file;
  - `codebook_sha256` mismatch with the implemented NU-Q2 codebook;
  - `q2_0` (uniform Q2) cache types, which are not supported by the CUDA
    KV-cache path;
  - quantized V with FlashAttention disabled (quantized V requires FA).

## Byte accounting

Per-token cache bytes are computed from the format sizes:

| Type      | bytes/block | values/block | bytes/element |
|-----------|-------------|--------------|---------------|
| q4_0      | 18          | 32           | 0.5625        |
| q8_0      | 34          | 32           | 1.0625        |
| q3_0      | 14          | 32           | 0.4375        |
| q3_0_64   | 26          | 64           | 0.40625       |
| q2_nu     | 18          | 64           | 0.28125       |
| q2_0      | 18          | 64           | 0.28125       |

For Qwen3-4B (36 layers, 8 KV heads, 128 head dim), the production policies
resolve to:

| Policy         | K bytes/token | V bytes/token | total | vs Q4K-Q4V |
|----------------|---------------|---------------|-------|------------|
| Q4K-Q4V        | 20,736        | 20,736        | 41,472| baseline   |
| Q4K-Q3V        | 20,736        | 14,976        | 35,712| -13.8889%  |
| C2             | 20,736        | 12,032        | 32,768| -20.9877%  |
| Q4K-all-NU-Q2V | 20,736        | 10,368        | 31,104| -25%       |

## Example

See `examples/kv-cache-policies/` for the frozen production policies:

- `q4xq4.json` (Q4K-Q4V);
- `q4xq3_64.json` (Q4K-Q3V);
- `c2.json` (C2, exact per-layer V map: q4_0 at layers 0,7,20,23; q3_0_64 at
  9,14,16,34; q2_nu everywhere else);
- `allq2_nu.json` (Q4K-all-NU-Q2V);
- `q2k_f16v.json` (ablation example: q2_nu K, f16 V).

## Errors

All validation failures print a descriptive message to stderr and abort
context creation; the process exits nonzero. There is no silent fallback to a
different cache configuration. See POLICY_RUNTIME_VERIFICATION.md for the
recorded negative tests.
