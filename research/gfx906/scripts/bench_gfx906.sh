#!/bin/bash
# gfx906 benchmark script: llama-bench PP/TG matrix
# Usage: bench_gfx906.sh <output.csv> [tag]
set -e

BIN=${BIN:-build-gfx906/bin/llama-bench}
MODELS_DIR=${MODELS_DIR:-/home/petrouil/Projects/test_models}
OUT=${1:-reports/gfx906/results/bench.csv}
TAG=${2:-baseline}

MODELS=(
  "LFM2.5-2.6B-Q8_0.gguf"
  "LFM2.5-2.6B-Q4_0.gguf"
  "LFM2.5-2.6B-Q4_K_M.gguf"
)

PP_SIZES="128 256 512 1024 2048"
TG_SIZES="128"

REPS=5
: > "$OUT"
for m in "${MODELS[@]}"; do
  for pp in $PP_SIZES; do
    "$BIN" -m "$MODELS_DIR/$m" -p "$pp" -r "$REPS" -o csv >> "$OUT"
  done
  for tg in $TG_SIZES; do
    "$BIN" -m "$MODELS_DIR/$m" -n "$tg" -r "$REPS" -o csv >> "$OUT"
  done
done

python3 - "$OUT" "$TAG" <<'EOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['n_prompt'] != 'n_prompt']
print(f"tag={sys.argv[2]} rows={len(rows)}")
for r in rows:
    np_, ng = int(r['n_prompt']), int(r['n_gen'])
    kind = f"pp{np_}" if np_ else f"tg{ng}"
    print(f"{r['model_type'].split()[-1]:6s} {kind:>6s} {float(r['avg_ts']):10.1f} t/s  sd {float(r['stddev_ts']):.1f}")
EOF