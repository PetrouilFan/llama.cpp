# Equivalence and Golden-Test Plan

## Two kinds of equivalence

Because the clean branch uses a newer upstream commit, absolute logits may move.
Require:

1. semantic equivalence of formats and policy resolution;
2. paper-effect equivalence relative to each implementation's Q4K-Q4V baseline.

## Exact gates

These must match exactly:

- struct sizes;
- CPU packed bytes for fixed vectors;
- code assignment at thresholds;
- C2 layer map and counts;
- analytical bytes/token;
- parser validation behavior.

## Numerical gates

- CPU dequantization: exact when feasible, otherwise max error <= 1e-6.
- CUDA vs CPU dequantization: legacy tolerance or <= 2e-3, whichever is stricter.
- Mixed attention: use the legacy accepted tolerance and record dtype/scale.
- No NaN or Inf.

## Golden corpus

For every type generate:

- zero block;
- constant positive/negative;
- ramp;
- alternating signs;
- reproduction levels;
- threshold ± epsilon;
- extrema;
- fixed-seed normal and heavy-tail;
- all-code/all-position packed patterns.

Q2 covers all 64 positions and all four codes. Q3 covers all cross-byte bit
positions.

Store input hashes, packed bytes, reconstructed outputs, binary hash, command,
source commit, and tolerances.

## Policy gates

Verify:

- Q4K-Q3V: all 36 V layers Q3_0_64;
- C2 exact 4/4/28 assignment;
- all-NU-Q2: all 36 V layers;
- complete non-overlapping coverage;
- exact totals 41,472 / 35,712 / 32,768 / 31,104.

## Smoke equivalence

Run 64 Wiki chunks, 64 OWT chunks, and fixed GSM8K-200 for legacy and clean:

```text
Q4K-Q4V
Q4K-Q3V
C2
```

Report PPL, per-chunk NLL, ΔNLL to local Q4, per-example agreement, policy
identity, memory, and runtime.

Primary gate:

```text
abs(clean policy ΔNLL - legacy policy ΔNLL) <= 0.002
```

## Final release matrix

After smoke passes:

- full held-out PPL for Q4/Q3/C2;
- full GSM8K 1,319 for Q4/Q3/C2;
- 32K and 64K populated memory/throughput;
- all-NU-Q2 versus uniform-Q2 small PPL ablation;
- HumanEval/MBPP only when necessary for direct clean-branch claims.

## Failure classification

Classify mismatches as:

```text
format
policy
upstream baseline drift
backend numerical drift
harness
hash mismatch
performance
unknown
```

Never loosen gates after observing results without a written rationale.
