# Pipeline

`rogii.pipeline.run` executes the stages below in order. Every stage takes and returns an
`id,tvt` frame; the report files each one writes next to `submission.csv` keep the names the
notebooks used.

| # | stage | module | writes |
|---|---|---|---|
| 1 | ridge stack: boosted fold models over tabular features, stacked by a positive ridge | `models/ridge_stack.py` | |
| 2 | selector: 128-seed likelihood-weighted PF plus a 14-config beam ensemble, mixed per well; wells that also exist in train use their contact path | `stages.py`, `tracking/` | `bimodal_selector_report.csv`, `bimodal_active_wells.csv` |
| 3 | anchor: `0.30 * ridge + 0.70 * selector` | `stages.anchor_blend` | |
| 4 | projection: robust low-order fit of the stratigraphic level against normalised MD | `postprocess/projection.py` | `sp45_projection_submission.csv` |
| 5 | learned trajectory: saved boosters over PF/beam/NCC/surface features, blended with a likelihood-weighted PF | `models/learned.py` | `learned_trajectory_submission.csv` |
| 6 | anchor x learned blend (weight `sp45_weight`, all candidates saved) | `blend.py` | `submission_sp45_learned_w*.csv`, `sp45_learned_blend_report.csv` |
| 7 | same-well contact override, accepted only when the visible prefix reproduces to 1 ft | `postprocess/contact_override.py` | `overlap_layer_probe.csv`, `guarded_overlap_override_report.csv` |
| 8 | visible-prefix calibration: back-test candidate generators on hidden slices of each prefix, then make a bounded move | `postprocess/visible_prefix.py` | `gold_prefix_*.csv`, `submission_gold_prefix_<profile>.csv`, `submission_self_verified_anchor.csv`, `gold_prefix_submission_audit.json` |
| 9 | saved model package, gated by how far it disagrees with the current trajectory | `models/package.py` | `model_package_*.csv`, `submission_model_package_gated_*.csv` |
| 10 | PF seed-branch hedge for wells whose seeds split in two | `postprocess/branch_hedge.py` | `pf_seed_branch_hedge_report.csv`, `submission_before_branch_hedge.csv` |
| 11 | audit | `audit.py` | `submission.csv`, `submission_audit.json`, `latest_valid_submission.*` |
| 12 | optional structural blend (`--structural`), then audited again | `structural/` | |

Stages 7-9 are skipped or neutralised by profile where noted in `configs/profiles/*.toml`.

## Inputs

`paths.competition_dir` looks for the competition folder in this order: `--data`, `$ROGII_DATA`,
the Kaggle input mounts, `data/raw`. Artifact datasets are looked up under `--artifacts`,
`$ROGII_<DATASET>` (upper-cased, dashes to underscores), `artifacts/<name>`, and the Kaggle mounts.

## Offline diagnostics

`rogii cv oracle|selector|heel|struct` scores pieces of the pipeline on train wells:

- `oracle`: pooled RMSE of flat / constant / line / smooth fits of the truth, a floor for any model
- `selector`: the selector on wells it never saw (24 PF seeds by default)
- `heel`: whether heel calibration moves the GR datum scan towards the true datum
- `struct`: leave-one-out score of a structural variant against carry-forward and the neighbour path
