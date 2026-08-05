# Release Freeze Checklist

## Repository hygiene
- [ ] Official current-upstream clone and exact base recorded.
- [ ] Focused branch with no unrelated changes.
- [ ] `git diff --check` passes.
- [ ] Clean worktree.
- [ ] No model, dataset, build, raw trace, or private-path artifact tracked.

## Formats
- [ ] Uniform Q2 semantics preserved.
- [ ] NU-Q2 explicit and distinct.
- [ ] Q2/NU-Q2 size 18 bytes.
- [ ] Q3_0 size 14 bytes.
- [ ] Q3_0_64 size 26 bytes.
- [ ] Static assertions and golden packed-byte tests pass.

## Policy system
- [ ] Public policy-file CLI documented.
- [ ] Versioned schema.
- [ ] Invalid policies fail closed.
- [ ] Geometry and optional strict hash validation.
- [ ] Runtime map/hash/bytes logging.
- [ ] Exact C2 map.
- [ ] Exact byte accounting.

## Correctness
- [ ] CPU reference tests.
- [ ] CUDA/CPU parity.
- [ ] Q2 all-code/all-position regression.
- [ ] Q3 cross-byte regression.
- [ ] row read/write.
- [ ] cache lifecycle.
- [ ] selected mixed attention.
- [ ] test-quantize-fns.
- [ ] test-quantize-stats documented.
- [ ] test-backend-ops compared with pristine base.
- [ ] Compute Sanitizer memcheck/initcheck clean.
- [ ] racecheck result documented.

## Equivalence
- [ ] Wiki-64.
- [ ] OWT-64.
- [ ] GSM8K-200.
- [ ] ΔNLL gate.
- [ ] analytical memory exact.
- [ ] 32K/64K memory.
- [ ] decode overhead <=4% or explained.
- [ ] full PPL and GSM8K release confirmation.
- [ ] NU-Q2 beats uniform Q2 at equal bytes.

## Documentation
- [ ] Design.
- [ ] Policy schema.
- [ ] Build/reproduction.
- [ ] Correctness.
- [ ] Equivalence.
- [ ] Limitations.
- [ ] Base commit.
- [ ] Release manifest.
- [ ] CITATION.cff.
- [ ] License/attribution audit.
- [ ] Historical evidence vs clean implementation clearly separated.

## Freeze
- [ ] Focused commits reviewed.
- [ ] Branch head and binary hashes recorded.
- [ ] SHA256SUMS generated.
- [ ] Local annotated tag `beyond-perplexity-v0.1.0-paper`.
- [ ] Push commands generated but not executed without authorization.
