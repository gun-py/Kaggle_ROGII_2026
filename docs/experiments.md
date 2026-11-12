# Experiment log

Numbers that were scattered through notebook comments, kept here so the code does not have to
carry them. They are reported as the notebooks recorded them; nothing was re-run.

## Components

| component | score | note |
|---|---|---|
| last-known carry-forward | 15.91 CV | pooled RMSE over 773 train wells |
| LightGBM, original features | 10.85 CV | |
| stack with likelihood-weighted PF features | 9.69 CV | the ridge stack |
| baseline recipe | 9.75 CV | |
| final learned stack | 9.21 CV | |
| pipeline above, after post-processing | 9.349 LB | "stack" in the blend notes |
| structural model, v4 weights | 10.963 LB | "struct" |
| saved model package, labelled | 7.335 LB | not reproducible, see below |

## Structural blend history

- `v2` (`tvt-v2-blended`): `0.65 * stack + 0.35 * struct`, 7.319 LB.
- `cv1`: weight 0.40, the leaderboard-solved robust optimum, predicted about 7.26.
- `cv2`: refit + ElasticNet + PLS ensemble; honest 5-fold OOF 9.814 against 10.090 for the fixed weights of `v2`.
- `cv3`: adds GRU and TCN members trained offline on the same eleven columns.
- `cv4`: struct weight 0.40 / 0.45 / 0.30 by neighbour distance (below 400 ft, 400-700 ft, beyond), derived from the global error correlation 0.041 and per-bin error variances.
- `v3`: three-way blend with the saved package's trees, package off.
- `v4`: package on at 0.50 with weights 0.30 / 0.20 / 0.50. It scored 9.162. A convex blend of the three labelled scores cannot be worse than 8.665 (8.784 with the tighter Jensen bound), so at least one component was far worse in that run than its label. Taking stack and struct as labelled puts the package at RMSE 16.8 or more. The likely mechanism: 44 `tw_path_boundary_*` features are all-NaN at test time because test typewells carry no `Geology` column, which shifts the trees as well as the TCN.
- `v5`: package off again; struct weights refit per neighbour-distance bin and four MD bands. Over 773 leave-one-out train wells, 5-fold GroupKFold, 3.78M rows: carry-forward 15.910, `v4` weights 10.082, refit with the same structure 9.961, refit with four MD bands 9.849. Free least squares overfit (13.27) as did six extra neighbour-gating columns (9.970). Two-way optimum 0.57 / 0.43, expected about 7.19.
- `v2_dl` (`tvt-v2-blended1`): a BiGRU over strided neighbour-prior, GR-emission and trajectory features, with an HGB fallback. Local diagnostics on 2026-08-02 found it redundant (correlation 0.7-0.8 with struct and stack, NNLS weight 0), so its weight defaults to 0.

## Notebook to variant

| notebook | variant |
|---|---|
| `high-scoring-reproducible` | base pipeline, `rogii predict` without `--structural` |
| `tvt-structural-alignment` | origin of the structural model; `rogii cv struct` |
| `tvt-v2-blended`, `blend_cell.py` | `v2` |
| `CV1` | `cv1` |
| `CV2` | `cv2` |
| `CV3` | `cv3` |
| `CV4` | `cv4` |
| `tvt-v3-threeway`, `blend_cell_v3.py` | `v3` |
| `tvt-v4-threeway`, `blend_cell_v4.py` | `v4` |
| `tvt-v5-threeway`, `blend_cell_v5.py` | `v5` |
| `tvt-v2-blended1` | `v2_dl` |
