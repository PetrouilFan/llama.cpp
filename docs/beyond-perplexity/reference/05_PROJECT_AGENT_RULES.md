# Project Agent Rules

These supplement but never override upstream `AGENTS.md` and `CONTRIBUTING.md`.

1. Old repository is read-only.
2. Current upstream APIs/style win over legacy structure.
3. Implement paper-required functionality only.
4. No blind cherry-picks.
5. No environment-only public configuration.
6. No silent policy fallback.
7. Never redefine standard Q2 semantics as NU-Q2.
8. No persistent full-FP16 mirror cache.
9. No combinatorial CUDA template explosion.
10. Every packed type gets CPU reference, tests, and size assertions.
11. Every CUDA path gets CPU-reference comparison.
12. “Upstream failure” requires a pristine-base comparison.
13. Never relax a threshold after failure without evidence.
14. No model/data/build artifacts in Git.
15. No number without command, hash, and source.
16. Never mutate final2/final3/revision4.
17. Do not rewrite historical paper evidence with clean-branch numbers.
18. Equivalence is a new artifact and must be labelled.
19. Do not push to official upstream.
20. Do not tag until mandatory gates pass.
