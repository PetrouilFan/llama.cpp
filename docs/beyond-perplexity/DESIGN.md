# Design

This document describes the clean current-upstream implementation of static
mixed-precision KV caching for the Beyond Perplexity paper. It is written for
reviewers and future maintainers.

## Overview

llama.cpp caches one K and one V tensor per layer. Upstream supports a single
global K type and a single global V type. This branch adds:

- three new ggml formats (NU-Q2, Q3_0, Q3_0_64);
- a per-layer V type via an explicit policy file;
- direct packed storage in the KV cache (no mirror F16 cache);
- CUDA support for the new types across quantization, copy, get/set rows,
  vector dot, and FlashAttention;
- fail-closed validation of the policy against the loaded model.

The three production operating points from the paper are: Q4K-Q4V (baseline),
Q4K-Q3V, and C2 (4 layers Q4_0, 4 layers Q3_0_64, 28 layers NU-Q2 on V).

## Formats

| Type | values/block | payload | scale | total bytes | effective bits/value |
|------|--------------|---------|-------|-------------|----------------------|
| q2_nu | 64 | 16 B, 2-bit | FP16 | 18 | 2.25 |
| q3_0 | 32 | 12 B, 3-bit | FP16 | 14 | 3.5 |
| q3_0_64 | 64 | 24 B, 3-bit | FP16 | 26 | 3.25 |

NU-Q2 is a distinct runtime type (`GGML_TYPE_Q2_NU`) carrying its own
codebook with levels `[-0.619326, -0.186363, 0.186363, 0.619301]`. The
historical paper used `GGML_TYPE_Q2_0` (uniform 2-bit) for both its uniform
and NU variants; the clean branch separates the two so the semantics are
explicit and self-consistent. The uniform `q2_0` type is preserved with its
upstream semantics but is not supported as a KV cache format in the CUDA path.

Q3_0 is the upstream 32-value 3-bit format. Q3_0_64 is the paper's 64-value
3-bit variant (26 bytes). Both use the same 3-bit packing with a centered
codebook `{-4,-3,-2,-1,0,1,2,3}`.

## CPU layer

New types are registered in the ggml type tables (ggml.c) with:

- `blck_size`, `type_size`, `name`;
- dequantize row functions;
- quantize functions with exact threshold assignment;
- vector dot functions against Q8_0;
- traits (`vec_dot_type = GGML_TYPE_Q8_0`).

Packed bytes are byte-identical to the legacy implementation for fixed input
vectors (golden vectors, see CORRECTNESS_REPORT.md).

## CUDA layer

- `common.cuh`: type traits for the new types (qk/qr/qi).
- `dequantize.cuh`: dequantize kernels (pairs of consecutive elements).
- `vecdotq.cuh`: MMQ-style plain vector dot kernels for q3_0, q3_0_64, q2_nu
  against q8_1; VDR (values per register) = 1 for all three.
- `mmvq.cu`: dispatch for MUL_MAT on the non-FlashAttention path.
- `convert.cu`, `getrows.cu`, `set-rows.cu`, `cpy.cu`, `cpy-utils.cuh`:
  conversion, row read/write, and copy for the new types.
- `fattn.cu` + generated instances: FlashAttention support for the mixed
  K/V pairs used by the policies (see FATTN_INSTANCE_AUDIT.md).
- `ggml-cuda.cu`: op support flags for MUL_MAT, GET_ROWS, SET_ROWS, CPY.

FlashAttention is mandatory for quantized V (enforced at context creation),
and it is the path the paper used for all production measurements.

## Policy system

The public interface is `--kv-cache-policy FILE`. The file is a versioned
JSON document (schema version 1) described in POLICY_SCHEMA.md. Key design
points:

- `k_type` is uniform per policy; `v_layer_types` is a flat per-layer array;
- the policy carries a canonical SHA-256, plus optional metadata
  (architecture, layer count, KV geometry, model hash, codebook hash);
- parse rejects malformed documents (unknown types, hash mismatch, wrong
  layer count);
- apply rejects policies that contradict the loaded model (architecture,
  geometry, model file hash, codebook hash, uniform q2_0, quantized V without
  FlashAttention);
- every rejection is an explicit error and abort; there is no silent fallback.

The legacy `LLAMA_KV_V_LAYERS` environment variable (range syntax) is not part
of the public interface; range/overlap constructs are not expressible in the
flat schema and fail at parse.

## Cache lifecycle

Per-layer types survive the full upstream cache lifecycle: allocation,
packed writes, reads, clear, copy, remove, keep, shift/defrag, slot reuse,
and state operations. There is no full-FP16 mirror cache; the cache stores
the packed blocks directly and reads go through GET_ROWS / CPY /
FlashAttention kernels for the resolved types.

## Deviations from the legacy implementation

Two intentional deviations are documented in depth:

- NU-Q2 vector dot uses the NU codebook (matches the CPU reference), not the
  legacy uniform `(q-1)` expression (NUQ2_LEGACY_DEVIATION.md). The legacy
  wart was never exercised on any historical paper path.
- q3_0/q3_0_64/q2_nu F32 copies use a consecutive-element pair kernel; the
  legacy generic template produced wrong F32 outputs for these types and was
  never exercised on GPU (Q3_COPY_FIX.md).

## Non-goals

Online routing, learned policy search, external memory, offload systems,
second-backend optimization, broad calibration, and external KV quantization
methods are out of scope, as is the full research harness.
