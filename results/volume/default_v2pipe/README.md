# Default volume-grid repeat using the current V2 pipeline files (#36)

Completed after PR #149 merged the files generated with the #136 recipe. Implementation base: `8fd69ba`.
Input commit: `e49137394886c8ea6ab6e383100c625f2516b629`. Run: `default_20261120-161532`.
Model seed 42, synthetic weight 1.0, fixed 60/20/20 splits and frozen original-arm validation-F1 thresholds.
Each cell selects its model on validation PR-AUC. Cleaned reference: **validation 0.5318 / test 0.5576**.

**VolumeDecision: CTGAN 0%; SMOTENC 0%.** No candidate has a validation lift with a strictly positive 95% CI lower bound.
All ten synthetic cells receive an **inconclusive** paired-bootstrap verdict against cleaned. No cell has a reliable gain.

| Arm | Fill | Selected model | Validation PR-AUC | Test PR-AUC | Validation lift 95% CI | Verdict |
|---|---:|---|---:|---:|---|---|
| SMOTENC | 1.0% | CatBoost | 0.5281 | 0.5569 | [-0.013786, +0.006463] | inconclusive |
| SMOTENC | 2.5% | CatBoost | 0.5313 | 0.5621 | [-0.012177, +0.010954] | inconclusive |
| SMOTENC | 5.0% | CatBoost | 0.5267 | 0.5611 | [-0.016248, +0.006248] | inconclusive |
| SMOTENC | 10.0% | CatBoost | 0.5342 | 0.5657 | [-0.009831, +0.014155] | inconclusive |
| SMOTENC | 25.0% | CatBoost | 0.5285 | 0.5515 | [-0.015863, +0.009873] | inconclusive |
| CTGAN | 1.0% | CatBoost | 0.5285 | 0.5530 | [-0.012909, +0.006011] | inconclusive |
| CTGAN | 2.5% | CatBoost | 0.5329 | 0.5681 | [-0.009686, +0.011925] | inconclusive |
| CTGAN | 5.0% | CatBoost | 0.5281 | 0.5559 | [-0.015509, +0.008025] | inconclusive |
| CTGAN | 10.0% | LightGBM | 0.5290 | 0.5553 | [-0.014929, +0.009819] | inconclusive |
| CTGAN | 25.0% | LightGBM | 0.5224 | 0.5540 | [-0.021956, +0.003939] | inconclusive |

## Protocol and checks

- 48 successful fits in 47 seconds, with no synthetic rows in validation or test.
- 30 focused volume, training-grid and chart tests passed.
- Paired class-stratified bootstrap: 1,000 resamples, bootstrap seed 0. The volume choice uses validation only. The #138 gain/loss verdict uses both splits.
- Both zero-fill decisions have `lift=lo=hi=0.0`; `decisions.json` lists every candidate lift and confidence interval.
- The five committed CTGAN files have correct row counts, positive labels and synthetic flags, and are nested prefixes of the 25% file. `input_audit.json` records SHA-256 hashes.
- The filename `default_ctgan_v2pipe_fill2.csv` denotes 2.5%, as PR #149 documents. The run explicitly uses the 25% pool and takes the correctly sized prefix for every level.
- This is an offline model-training comparison using the supplied files. It does not assert that the CTGAN rows pass the quality gate. #136 reports a fidelity failure on payment amounts; that generator-quality work is separate.

## Difference from the numbers posted in #136

All five SMOTENC cells reproduce Kyle's published validation/test PR-AUC at four decimals. The five CTGAN cells using the newly committed files differ from that table. We report the observed current-run numbers above without replacing them with the earlier values.
As a cross-check, the independent `tools.augmented` runner on the same 10% CTGAN rows returns validation 0.529000 / test 0.555282, matching `training_grid`. The earlier #136 comment reports 0.5296 / 0.5558. The cause of the difference between the published CTGAN table and the supplied files is not established. Saved original run predictions/settings would be needed to reconcile it.
`published_result_comparison.csv` records every difference; `runner_crosscheck.csv` records the second-runner check.

## Reproduce

```sh
uv run python -m agentic_gan.tools.training_grid --dataset default --seeds 42 \
  --fill-levels 0.01 0.025 0.05 0.10 0.25 \
  --ctgan-pool data/interim/default/synthetic/default_ctgan_v2pipe_fill25.csv \
  --results-dir results/volume/default_v2pipe
uv run python -m agentic_gan.tools.volume --dataset default --run runs/<id> \
  --out results/volume/default_v2pipe/decisions.json
```

The existing scikit-learn OpenMP library was supplied through `DYLD_FALLBACK_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/sklearn/.dylibs`; no environment packages were changed.

Artifacts: `grid.csv`, `comparisons.csv`, `decisions.json`, `fine_grid.png`, full model metrics, run times, input hashes and verification metadata. The untracked run folder preserves model/prediction artifacts.

Sources: [pool provenance, PR #149](https://github.com/Columbia-DS-Capstone/Data-Science-Capstone-Project/pull/149), [Kyle's #136 results](https://github.com/Columbia-DS-Capstone/Data-Science-Capstone-Project/issues/136#issuecomment-6968152535).
