# Changelog

## 0.1.0

First packaged version. Behaviour matches the notebooks at the `legacy-notebooks` tag; see
`docs/legacy-map.md` for the few places where the old code could not be followed literally.

- One pipeline (`rogii predict`) replaces the 57-cell base notebook and its nine appended-cell variants.
- Profiles are TOML files; the `globals().get(...)` flags and `ROGII_GOLD_*` environment bridge are gone.
- Stages pass frames instead of rewriting `submission.csv`; report files keep their names.
- The 2 MB base64 checkpoint embedded in `CV3.ipynb` is a package file.
- About 1.2 GB of datasets and model binaries left the repository; scripts re-download them.
- Added unit tests, a synthetic-data smoke run, a parity harness and CI.

Removed as dead or misleading code: the python beam search (shadowed by the compiled one before its
first call), the unused single-scale PF ensemble, the notebook figure with hard-coded scores, and the
full-stack ablation cell (recoverable from the tag).
