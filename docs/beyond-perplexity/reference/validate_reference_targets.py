#!/usr/bin/env python3
"""Validate analytical targets in 06_PAPER_IMPLEMENTATION_TARGETS.json."""
from __future__ import annotations
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
candidates = [
    HERE.parent / "06_PAPER_IMPLEMENTATION_TARGETS.json",
    HERE.parent.parent / "06_PAPER_IMPLEMENTATION_TARGETS.json",
    Path.cwd() / "docs/beyond-perplexity/reference/06_PAPER_IMPLEMENTATION_TARGETS.json",
]
path = next((p for p in candidates if p.exists()), None)
if path is None:
    raise SystemExit("Could not locate 06_PAPER_IMPLEMENTATION_TARGETS.json")

data = json.loads(path.read_text())
formats = data["formats"]
model = data["primary_model"]
policies = data["policies"]
n = model["elements_per_k_or_v_per_layer_token"]
L = model["n_layers"]

def bpt(type_name: str) -> int:
    value = n * formats[type_name]["bytes_per_element"]
    if abs(value - round(value)) > 1e-9:
        raise AssertionError(f"non-integral per-layer bytes for {type_name}: {value}")
    return int(round(value))

base = 2 * L * bpt("q4_0")
assert base == 41472, base

for key, policy in policies.items():
    k_bytes = L * bpt(policy["k_type"])
    v_bytes = sum(count * bpt(type_name)
                  for type_name, count in policy["v_type_counts"].items())
    total = k_bytes + v_bytes
    expected = policy["expected_total_kv_bytes_per_token"]
    if total != expected:
        raise AssertionError(f"{key}: computed {total}, expected {expected}")
    saving = (base - total) / base * 100.0
    expected_saving = policy["expected_saving_vs_q4k_q4v_percent"]
    if not math.isclose(saving, expected_saving, abs_tol=1e-8):
        raise AssertionError(f"{key}: saving {saving}, expected {expected_saving}")
    print(f"PASS {key:18s} K={k_bytes:5d} V={v_bytes:5d} "
          f"total={total:5d} saving={saving:8.4f}%")

c2 = policies["c2"]["v_layers"]
covered = sorted(c2["q4_0"] + c2["q3_0_64"] + c2["q2_nu"])
assert covered == list(range(36)), "C2 layer coverage mismatch"
assert len(set(covered)) == 36, "C2 duplicate layer assignment"

levels = formats["q2_nu"]["levels"]
thresholds = [(levels[i] + levels[i + 1]) / 2 for i in range(3)]
for got, expected in zip(thresholds, formats["q2_nu"]["midpoint_thresholds"]):
    assert abs(got - expected) < 1e-9, (got, expected)

print("PASS C2 complete/non-overlapping layer map")
print("PASS NU-Q2 midpoint thresholds")
print(f"Validated: {path}")
