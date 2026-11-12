# From the notebooks to the package

The pre-refactor tree is the git tag `legacy-notebooks` (`git show legacy-notebooks:CV1.ipynb`).
The parity tests read their reference code from it.

## Where the base notebook went

| `high-scoring-reproducible.ipynb` cells | now |
|---|---|
| 6, 7, 10 control panel, environment bridge, `CFG` | `config.py`, `configs/profiles/*.toml` |
| 8 koolbox install and fallback | `models/trainer.py` |
| 11 selector, particle filter, contacts, bimodal scan | `tracking/`, `geology/` |
| 14-24 ridge features and stack | `features/`, `models/ridge_stack.py` |
| 26-28 selector loop, 0.3 / 0.7 anchor | `stages.py` |
| 30 projection | `postprocess/projection.py` |
| 33-43 learned trajectory | `models/learned.py`, `tracking/kernels.py`, `tracking/trackers.py` |
| 45 anchor x learned blend | `blend.py` |
| 47-48 overlap probe, contact override | `postprocess/contact_override.py` |
| 50 visible-prefix calibration | `postprocess/visible_prefix.py` |
| 51 model-package correction | `models/package.py` |
| 52 seed-branch hedge | `postprocess/branch_hedge.py` |
| 53 audit | `audit.py` |
| 12 CV reports | `diagnostics/cv.py` |
| 34, 38 figures | `diagnostics/plots.py` |

The appended cells of the variants (`blend_cell*.py`, CV1-4, `tvt-v*-threeway`) became `structural/`.

## Things the notebooks did that the port had to resolve

**Shadowed definitions.** Several names are defined in more than one cell and a later definition wins
at call time. The one that matters: `run_beam_ensemble` (cell 11) calls `beam_search`, which cell 14
redefines as the compiled version before cell 26 first uses it. So the notebooks never ran the
python beam search from cell 11; the compiled one (rolling-mean smoothing, float32 paths) is what
produced every submission, and it is what `tracking/beam.py` keeps. The remaining duplicates
(`_pf_ancc`, `_pf_z`, `multi_scale_ncc`, the imputers, `build_well`) differ only cosmetically and are
merged. `load_well` genuinely differs between cells 11 and 33 (typewell sorted or not); the package
exposes `sort_typewell` and each caller passes what the notebook effectively used.

**pandas 3.** Several cells mutate arrays obtained from `.to_numpy()` in place, which raises on
pandas 3 (read-only views) and is silently swallowed by the surrounding `try/except`. On pandas 3 the
notebook therefore skips the prefix-trust gate and the seed-branch hedge. The package copies before
mutating, so it matches the notebook on pandas below 3, the version the notebooks were written against. The parity loader
applies the same copies to the legacy source.

**CSV round trips.** The notebooks rewrite and re-read `submission.csv` between stages; pandas'
default float parser is not exactly round-trip, so legacy values carry noise of a few ulp (about
1e-11 ft). The package keeps frames in memory. Parity tolerances are 1e-8.

**Numba RNG.** The single-path compiled particle filters use numba's thread-local, unseeded
generator, so features built from them differ between runs and between threads. Parity tests seed the
generator on the calling thread; the likelihood-weighted PF seeds per seed index and is exact.

**Report reasons.** The guarded override reported `no_ref` for a missing formation in cell 48 and
`valid_rows=0` in cell 50; both paths now say `no_ref`.

## What the parity suite covers

`pytest -m parity` (needs `data/raw`):

- seeded particle filter, likelihood-weighted ensemble and branch statistics: bit for bit
- beam ensemble, selector bucketing, variants with and without the bimodal hedge, bimodal scan: bit for bit
- compiled trackers, NCC and the tabular feature builder: bit for bit, including column order and dtypes
- every structural variant against its notebook cell on all test wells: within 1e-8 (`cv3` carries float32 nets)
- the whole post-ridge chain (selector, projection, blend, override, visible prefix, hedge, audit) for
  three profiles: every CSV and JSON report within 1e-8

Not covered, because they need artifacts that are not in the repository: the ridge stack and the
learned-trajectory models (both sides read the same stub files), and the saved model package, whose
HistGradientBoosting pickle does not load on scikit-learn 1.9. Neither a full leaderboard-equivalent
run nor training from scratch was attempted.

## Not ported

- the full-stack bimodal ablation and heel ablation grid (cell 54): read-only diagnostics, off by default
- the notebook figure with hard-coded scores (cell 44), moved to `docs/experiments.md`
- the single-scale `run_pf_lik_ensemble` and the python `beam_search`: unreachable in the notebook flow
