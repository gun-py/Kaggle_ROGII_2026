# Artifacts

Everything below lives in git-ignored folders and is fetched with the scripts in `scripts/`.

| folder under `artifacts/` | source | size | used by |
|---|---|---|---|
| `wellbore-geology-prediction-artifacts` | `ravaghi/wellbore-geology-prediction-artifacts` | large (single models exceed 100 MB) | ridge stack: `data/train.csv` and `models/{lightgbm,catboost}-N/*.pkl` |
| `rogii-claude-models-pub` | `fleongg/rogii-claude-models-pub` | 25 MB | learned trajectory: `lgb*.pkl`, `features.json` |
| `rogii-model-package` | `pilkwang/rogii-model-package` | 431 MB | model-package correction and the `v3`-`v5` package option: manifest, feature builder, XGBoost/CatBoost/HGB/LightGBM boosters, TCN |
| `koolbox-offline` | `phongnguyn23021656/koolbox-offline` | 53 MB | offline wheels for the `koolbox` Trainer that the saved ridge-stack pickles refer to |

Present in the original repository but not read by any current code:
`rogii-stack-v10-tabicl`, `rogii-tabicl-mirror`, `rogii-tabicl-mirror-duplicate`. They stay on disk
for reference and are not part of the fetch script.

## Notes

- Without `koolbox`, `rogii.models.trainer.ensure_koolbox` registers a compatible `Trainer` under that
  module name so the pickles still load.
- The package's HistGradientBoosting pickle was written by an older scikit-learn; it loads on
  Kaggle's image but not on scikit-learn 1.9 (`No module named '_loss'`).
- With no ridge artifacts, `rogii predict` stops and says so; `--train-missing` trains the stack from
  scratch, which takes hours and a GPU.
