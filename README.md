# rogii-wellbore

Predicting true vertical thickness (TVT) along horizontal wells for the
[ROGII wellbore geology prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
Kaggle competition.

Each test well comes with a visible heel (rows with `TVT_input`) and a hidden lateral.
The pipeline tracks the stratigraphic position through the lateral from gamma-ray (GR)
matching against a typewell, prior knowledge from neighbouring wells, and boosted models,
then blends and audits the result into `submission.csv`.

This repository started as a dump of eleven near-identical Kaggle notebooks. It is now an
installable package with one pipeline, TOML profiles, tests, and a parity harness that checks
the port against the original notebooks.

## Quick start

```bash
make install                      # venv, editable install, pre-commit hooks
make data                         # competition files -> data/raw (Kaggle CLI)
make artifacts                    # third-party model datasets -> artifacts/

.venv/bin/rogii list              # profiles and structural variants
.venv/bin/rogii predict --profile vp_balanced_modelpkg_005 --structural v5
```

`rogii predict --fast` shrinks every Monte-Carlo budget for a smoke run. Outputs (the submission
plus one report per stage) land in `outputs/`, or `/kaggle/working` on Kaggle.

On macOS LightGBM needs `brew install libomp`.

## How it fits together

```
src/rogii/
  config.py            frozen dataclasses; profiles in configs/profiles/*.toml
  pipeline.py          the prediction flow, stage by stage
  stages.py            selector stage and the ridge/selector anchor blend
  geology/             contact-derived TVT, heel calibration of the GR log
  tracking/            compiled particle filters and beam search, bimodal datum scan, selector variants
  features/            tabular features, formation and ANCC imputers
  models/              ridge stack, learned-trajectory inference, saved model package, Trainer shim
  postprocess/         projection, same-well contact override, visible-prefix calibration, branch hedge
  structural/          neighbour-well structural model and its blend variants
  diagnostics/         leave-well-out reports and plots
```

See [docs/pipeline.md](docs/pipeline.md) for the stage order and the files each stage writes,
[docs/method.md](docs/method.md) for the maths, and [docs/experiments.md](docs/experiments.md)
for the leaderboard notes that used to live in code comments.

### Profiles and variants

A *profile* (`configs/profiles/*.toml`) switches pipeline stages: which visible-prefix profile
decides the final trajectory, whether the saved model package may nudge it, whether the bimodal
hedge is active. A *structural variant* (`--structural`) is the extra neighbour-well blend that the
notebooks appended after the audit:

| variant | what it is |
|---|---|
| `v2`, `cv1` | distance-binned weights over eleven structural paths, blended 0.35 / 0.40 |
| `cv2` | refit + ElasticNet + PLS members averaged in drift space |
| `cv3` | `cv2` plus GRU and TCN corrections (checkpoint ships in `structural/weights`) |
| `cv4` | `cv2` with a per-well blend weight gated by neighbour distance |
| `v3`, `v4`, `v5` | three-way blends with the saved package; `v5` bands the weights by depth past the anchor |
| `v2_dl` | `v2` followed by an optional BiGRU/HGB component (weight 0 by default) |

## Data and artifacts

Nothing large is tracked. `data/raw` holds the competition files and `artifacts/` the third-party
datasets listed in [docs/artifacts.md](docs/artifacts.md); both are git-ignored and re-created by
`scripts/fetch_data.sh` and `scripts/fetch_artifacts.sh`.

## Tests

```bash
make lint
make test        # unit and smoke tests on a synthetic competition folder, about a minute
make parity      # compares against the notebooks at the legacy-notebooks tag; needs data/raw
```

The parity suite executes the original cell definitions from `git show legacy-notebooks:...`
and asserts that tracking, features, every structural variant and the full post-ridge chain
agree with the new code (see [docs/legacy-map.md](docs/legacy-map.md) for what that did and did not cover).

## Running on Kaggle

`make wheel` builds `dist/kaggle/rogii_wellbore-*.whl`. Upload it as a dataset, attach it with the
competition and the artifact datasets, and run `notebooks/kaggle_inference.ipynb`.

## License

MIT, see [LICENSE](LICENSE). Third-party datasets keep their own terms
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
