#!/usr/bin/env python3
# Assemble the raw golden records into the committed reference JSONs.
#
# Usage: generate.py /path/to/legacy/llama.cpp [--out DIR]
#
# Reads build/golden_cpu.raw and build/golden_q2nu.raw produced by
# build_harness.sh + running the binaries, and writes:
#   tests/beyond-perplexity/reference/q2_uniform_vectors.json
#   tests/beyond-perplexity/reference/q2_nu_vectors.json
#   tests/beyond-perplexity/reference/q3_0_vectors.json
#   tests/beyond-perplexity/reference/q3_0_64_vectors.json
#   tests/beyond-perplexity/reference/legacy_reference_manifest.json
import argparse
import hashlib
import json
import os
import subprocess
import sys

TYPES = {
    "q2_0":    "q2_uniform_vectors.json",
    "q2_nu":   "q2_nu_vectors.json",
    "q3_0":    "q3_0_vectors.json",
    "q3_0_64": "q3_0_64_vectors.json",
}

LEGACY_SOURCE_COMMIT = "bc80cdb03636f462208d18ea2d43213cd1c161f2"
CODEBOOK_SHA256 = "af450f1aec81b9a23e346640e27ab3150f5a9827b424750eda61d4104dea6713"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy", help="path to the frozen legacy llama.cpp repo")
    ap.add_argument("--out", default=None, help="output directory (default: tests/beyond-perplexity/reference relative to repo root)")
    args = ap.parse_args()

    legacy = os.path.realpath(args.legacy)
    here = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    out_dir = args.out or os.path.join(repo_root, "tests", "beyond-perplexity", "reference")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(legacy, "ggml", "src")):
        sys.exit(f"not a llama.cpp checkout: {legacy}")

    head = subprocess.run(["git", "-C", legacy, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != LEGACY_SOURCE_COMMIT:
        print(f"WARNING: legacy HEAD is {head}, expected {LEGACY_SOURCE_COMMIT}", file=sys.stderr)

    raw_dir = os.path.join(here, "build")
    records = []
    for raw in ("golden_cpu.raw", "golden_q2nu.raw"):
        path = os.path.join(raw_dir, raw)
        if not os.path.exists(path):
            sys.exit(f"missing {path} - run build_harness.sh and the harnesses first")
        for line in open(path):
            line = line.strip()
            if line:
                records.append(json.loads(line))

    by_type = {}
    for r in records:
        by_type.setdefault(r["type"], []).append(r)

    for t, fname in TYPES.items():
        if t not in by_type:
            sys.exit(f"no records for {t}")
        # drop hex/f duplicates: keep compact [hex, f] pairs
        out = []
        for r in by_type[t]:
            out.append({
                "case": r["case"],
                "sub": r["sub"],
                "input": [[x["hex"], x["f"]] for x in r["input"]],
                "scale": r["scale"],
                "packed": r["packed"],
                "dequant": [[x["hex"], x["f"]] for x in r["dequant"]],
            })
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            json.dump({"format_version": 1, "type": t, "records": out}, f, indent=1)
            f.write("\n")
        print(f"wrote {path} ({len(out)} records)")

    # input hashes: sha256 of the concatenated input hex values, per type
    input_hashes = {}
    for t, recs in by_type.items():
        h = hashlib.sha256()
        for r in recs:
            for x in r["input"]:
                h.update(x["hex"].encode())
        input_hashes[t] = h.hexdigest()

    gen_cmd = (
        "scripts/beyond-perplexity/legacy_golden/build_harness.sh <legacy-repo>; "
        "scripts/beyond-perplexity/legacy_golden/build/golden_host > "
        "scripts/beyond-perplexity/legacy_golden/build/golden_cpu.raw; "
        "scripts/beyond-perplexity/legacy_golden/build/golden_q2nu > "
        "scripts/beyond-perplexity/legacy_golden/build/golden_q2nu.raw; "
        "python3 scripts/beyond-perplexity/legacy_golden/generate.py <legacy-repo>"
    )

    binaries = {}
    for b in ("golden_host", "golden_q2nu"):
        path = os.path.join(raw_dir, b)
        if os.path.exists(path):
            binaries[b] = sha256_file(path)

    manifest = {
        "format_version": 1,
        "title": "beyond-perplexity legacy golden reference",
        "source": {
            "repo": "legacy experimental llama.cpp (see legacy_reference_manifest provenance)",
            "commit": LEGACY_SOURCE_COMMIT,
            "tag": "paper-evidence-freeze-v1",
            "q2_nu_codebook_sha256": CODEBOOK_SHA256,
        },
        "generation": {
            "command": gen_cmd,
            "host_compiler": subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
            "cuda_compiler": subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.splitlines()[-1] if shutil_which("nvcc") else "n/a",
            "binaries_sha256": binaries,
        },
        "input_dtype": "float32",
        "input_hashes_sha256": input_hashes,
        "tolerances": {
            "cpu_dequant_abs": 1e-6,
            "cuda_dequant_abs": 2e-3,
        },
        "coverage": {
            "per_type": ["zeros", "const_pos", "const_neg", "extrema", "thresh", "all_codes", "all_positions", "random"],
            "q2_0_note": "uniform q2_0: d = amax, codes 0..2 used (code 3 unreachable), levels {-amax, 0, +amax}",
            "q2_nu_note": "non-uniform q2_0: d = amax, thresholds -0.403/0.0/+0.403, levels from codebook sha",
            "q3_note": "q3_0/q3_0_64: d = max/-4, 8 codes, levels (c-4)*d",
            "packed_note": "q2 families: byte j/4, bit (j%4)*2; q3 families: byte (j*3)/8, bit (j*3)%8",
        },
        "no_model_data": True,
        "no_private_data": True,
    }
    path = os.path.join(out_dir, "legacy_reference_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")


import shutil


def shutil_which(name):
    return shutil.which(name)


if __name__ == "__main__":
    main()
