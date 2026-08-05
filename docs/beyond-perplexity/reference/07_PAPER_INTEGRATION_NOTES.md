# Paper and Repository Integration Notes

## Safe repository claim after gates pass

> This branch is a clean reimplementation of the native mixed-precision KV
> cache formats and policies evaluated in “Beyond Perplexity,” rebased onto
> current upstream llama.cpp. The original paper numbers remain tied to the
> frozen historical evidence package; this branch includes a separate
> equivalence report against that implementation.

## Unsafe claims

- clean branch generated historical tables;
- exact reproduction when only byte accounting matches;
- universal Q4 key floor;
- universal quality preservation;
- external-baseline superiority;
- broad family/backend generalization;
- zero overhead;
- lossless compression.

## Paper update after release

Add:

```text
Clean implementation repository: <release URL>
Clean upstream base commit: <commit>
Clean release tag: beyond-perplexity-v0.1.0-paper
Historical evidence archive: <DOI/stable URL>
```

Distinguish historical evidence, clean implementation, and equivalence report.

## Recommended release assets

- GitHub source archive/tag;
- release manifest;
- SHA256SUMS;
- policy preset bundle;
- compact golden vectors;
- equivalence summary JSON.

Large raw outputs belong in an external artifact archive.
