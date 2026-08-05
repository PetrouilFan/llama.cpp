#!/usr/bin/env bash
# Build the golden-vector harnesses from the frozen legacy repo.
#
# Usage: build_harness.sh /path/to/legacy/llama.cpp
#
# The legacy repo is only READ. It must be at (or before) the frozen
# evidence commit bc80cdb03636f462208d18ea2d43213cd1c161f2
# (tag paper-evidence-freeze-v1), which contains the q3_0/q3_0_64/q2_0
# formats in ggml/src/ggml-quants.c and q2_nu in ggml/src/ggml-cuda/.
set -euo pipefail

LEGACY="${1:?usage: build_harness.sh /path/to/legacy/llama.cpp}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${HERE}/build"

mkdir -p "$OUT"

gcc -O2 -I "$LEGACY/ggml/include" -I "$LEGACY/ggml/src" \
    -c "$LEGACY/ggml/src/ggml-quants.c" -o "$OUT/ggml-quants.o"

g++ -O2 -I "$LEGACY/ggml/include" -I "$LEGACY/ggml/src" \
    "$HERE/golden_host.cc" "$OUT/ggml-quants.o" "$HERE/stubs.cc" \
    -o "$OUT/golden_host" -lm

if command -v nvcc > /dev/null 2>&1; then
    nvcc -O2 -std=c++17 \
        -I "$LEGACY/ggml/include" -I "$LEGACY/ggml/src" \
        -I "$LEGACY/ggml/src/ggml-cuda" \
        "$HERE/golden_q2nu.cu" -o "$OUT/golden_q2nu"
    echo "built: $OUT/golden_host, $OUT/golden_q2nu"
else
    echo "built: $OUT/golden_host (nvcc not found, q2_nu golden skipped)"
fi
