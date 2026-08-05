# Legacy golden vector generation (beyond-perplexity)

This directory regenerates the golden reference vectors committed under
tests/beyond-perplexity/reference/. The vectors are produced by the frozen
legacy experimental implementation and are the byte-exact equivalence target
for the clean reimplementation.

## Provenance

- Source repository (read-only): /home/petrouil/Projects/llama.cpp_tests/kv_compressions/llama.cpp
- Source commit: bc80cdb03636f462208d18ea2d43213cd1c161f2 (tag paper-evidence-freeze-v1)
- Source branch: paper/full-evaluation
- Legacy codebook sha256: af450f1aec81b9a23e346640e27ab3150f5a9827b424750eda61d4104dea6713
- Generator binaries: gcc and nvcc of the host machine (versions recorded in
  legacy_reference_manifest.json)

## What is compiled

- golden_host.cc links the legacy ggml/src/ggml-quants.c compiled verbatim
  (with stubs.cc supplying ggml_abort and type-size helpers). It produces the
  q2_0 (uniform Q2), q3_0 and q3_0_64 vectors with the legacy CPU
  quantize/dequantize functions.
- golden_q2nu.cu is compiled with nvcc and includes the legacy
  q2-nu-codebook.cuh and cpy-utils.cuh verbatim. It produces the q2_nu
  (non-uniform Q2) vectors from the legacy CUDA quantize/dequantize device
  functions.

## Coverage (8 cases x 4 variants per type)

- zeros: all-zero block (exercises the degenerate scale path)
- const_pos / const_neg: constant +-0.75 block
- extrema: block with extrema values and 1e-4 floor
- thresh: values at the codebook quantization thresholds, epsilon on both sides
- all_codes: every reachable code value in one block
- all_positions: every packed bit position in the block
- random: fixed-seed random blocks (seed 12345 + sub * 7919, uniform in [-1, 1])

## Regenerate

    ./build_harness.sh /path/to/legacy/llama.cpp
    python3 generate.py /path/to/legacy/llama.cpp [--out DIR]

generate.py refuses to run unless the legacy repository HEAD matches
bc80cdb03636f462208d18ea2d43213cd1c161f2. Output defaults to
tests/beyond-perplexity/reference/ (5 JSON files: 4 vector files +
legacy_reference_manifest.json).

## Tolerances

- CPU dequantization compare: absolute 1e-6
- CUDA dequantization compare: absolute 2e-3
- Packed bytes: exact match required

## Notes

- No model weights or private data are included in the outputs.
- Input dtype is float32; scale and dequant values are recorded in fp16/fp32
  hex plus decimal.
- The q2_0 vectors double as the "Q2 uniform" control. Upstream q2_0 semantics
  are identical to the legacy ones, so these vectors are valid against both
  codebases.
