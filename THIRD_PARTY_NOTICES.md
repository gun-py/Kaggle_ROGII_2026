# Third-party material

The pipeline reads, but does not redistribute, public Kaggle datasets. Each is governed by the
licence on its Kaggle page; fetch them yourself with `scripts/fetch_artifacts.sh`.

| dataset | used for |
|---|---|
| `ravaghi/wellbore-geology-prediction-artifacts` | cached train features and the fitted LightGBM/CatBoost fold models of the ridge stack |
| `fleongg/rogii-claude-models-pub` | pretrained trajectory boosters (`lgb*.pkl`) and their feature list |
| `pilkwang/rogii-model-package` | saved model package: feature builder, boosters, TCN, blend config |
| `phongnguyn23021656/koolbox-offline` | offline wheels, including the `koolbox` Trainer the saved pickles refer to |

The competition data belongs to ROGII and is available under the competition rules.
