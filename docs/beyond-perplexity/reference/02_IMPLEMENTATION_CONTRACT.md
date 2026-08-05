# Implementation Contract

This document defines the public feature that the clean branch must implement.
It is normative unless direct inspection of the frozen legacy implementation
proves a detail inaccurate. Any correction must be documented and traced to
source or golden evidence.

## Product scope

The clean branch implements native static mixed-precision KV caching for
llama.cpp with:

- Q4 keys for the two production operating points;
- Q3_0_64, NU-Q2, or Q4 values per layer;
- an explicit policy-file interface;
- direct packed cache storage;
- selected CUDA attention support;
- correctness and equivalence tests;
- paper presets.

It is not a general optimizer, online controller, offload system, external
memory mechanism, or implementation of KIVI/KVQuant.

## Formats

### Q2 uniform

- 64 values/block.
- 16 packed payload bytes.
- one FP16 scale.
- 18 total bytes.
- 2.25 effective bits/value.

### NU-Q2

- same 64-value/18-byte physical footprint;
- distinct runtime type and explicit codebook semantics;
- levels `[-0.619326, -0.186363, 0.186363, 0.619301]`;
- midpoint thresholds `[-0.4028445, 0.0, 0.402832]`.

The exact scale and normalization rules come from legacy source and golden
vectors, not prose alone.

### Q3_0

- 32 values.
- 12 payload bytes.
- FP16 scale.
- 14 total bytes.
- 3.5 effective bits/value.

### Q3_0_64

- 64 values.
- 24 payload bytes.
- FP16 scale.
- 26 total bytes.
- 3.25 effective bits/value.

### Existing references

Q4_0 is 18 bytes/32 values. Q8_0 is 34 bytes/32 values.

## Production policies

### Q4K-Q3V

Every layer uses Q4_0 keys and Q3_0_64 values.

Target: 35,712 bytes/token, 13.8889% below Q4K-Q4V.

### Q4K-MixV-21 / C2

Every layer uses Q4_0 keys.

Values:

```text
q4_0:      [0, 7, 20, 23]
q3_0_64:   [9, 14, 16, 34]
q2_nu:     all remaining layers
```

Target: 32,768 bytes/token, 20.9877% below Q4K-Q4V.

### Q4K-all-NU-Q2V

Every layer uses Q4_0 keys and Q2_NU values.

Target: 31,104 bytes/token, 25% below Q4K-Q4V.

## Public policy schema

Use a versioned JSON policy with:

```text
schema version
policy ID
description
selection provenance
model architecture
layer count
KV head count
head dimension
optional model SHA-256
codebook ID/hash/levels
K default/map
V default/map
```

The implementation may choose a cleaner equivalent schema after documenting it.

## Validation

Fail on:

- unsupported schema;
- unknown type;
- overlapping assignments;
- uncovered layers;
- out-of-range indices;
- geometry mismatch;
- codebook mismatch;
- strict model hash mismatch;
- byte-accounting mismatch.

No silent fallback.

## Runtime identity

Print deterministically:

```text
policy ID and canonical hash
model geometry and model-hash verification
codebook ID/hash
resolved K/V types
type counts
K/V/total bytes per token
saving versus Q4K-Q4V
selected attention paths
```

## Mandatory optimized CUDA pairs

```text
q4_0 K × q4_0 V
q4_0 K × q3_0_64 V
q4_0 K × q2_nu V
```

Ablation-only pairs may use a documented correct fallback.

## Cache lifecycle

Per-layer types must survive allocation, writes/reads, clear, copy, remove,
keep, shift/defrag, slot reuse, and state operations required by current
upstream.

No persistent full-FP16 mirror cache.

## Non-goals

- online routing;
- learned policy search;
- external memory/offload;
- second-backend optimization;
- broad family calibration;
- external KV quantization methods;
- full research harness in Git.
