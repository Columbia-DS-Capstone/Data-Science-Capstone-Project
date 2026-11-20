# Bank synthetic-weight sweep (#135)

Run after PR #146 merged the #36 volume-rule implementation into GitHub main (`f244a3e`).
Seed 42, fixed 60/20/20 splits, original-arm validation-F1 thresholds; models selected on validation PR-AUC.
Cleaned reference: validation **0.4500**, test **0.4811**.

**No cell reliably beats cleaned.** Sixteen cells are inconclusive and two are reliable losses.
The loss cells are SMOTENC at 25% fill with weights 0.5 and 1.0.
Some CTGAN cells exceed cleaned on test alone, but none meets the #138 gain criterion on both validation and test.

| Arm | Fill | Weight | Validation PR-AUC | Test PR-AUC | Verdict |
|---|---:|---:|---:|---:|---|
| ctgan | 0.05 | 0.25 | 0.4438 | 0.4722 | inconclusive |
| ctgan | 0.05 | 0.5 | 0.4374 | 0.4766 | inconclusive |
| ctgan | 0.05 | 1 | 0.4372 | 0.4828 | inconclusive |
| ctgan | 0.1 | 0.25 | 0.4424 | 0.4860 | inconclusive |
| ctgan | 0.1 | 0.5 | 0.4396 | 0.4872 | inconclusive |
| ctgan | 0.1 | 1 | 0.4372 | 0.4771 | inconclusive |
| ctgan | 0.25 | 0.25 | 0.4401 | 0.4818 | inconclusive |
| ctgan | 0.25 | 0.5 | 0.4333 | 0.4833 | inconclusive |
| ctgan | 0.25 | 1 | 0.4264 | 0.4671 | inconclusive |
| smotenc | 0.05 | 0.25 | 0.4504 | 0.4694 | inconclusive |
| smotenc | 0.05 | 0.5 | 0.4463 | 0.4649 | inconclusive |
| smotenc | 0.05 | 1 | 0.4471 | 0.4780 | inconclusive |
| smotenc | 0.1 | 0.25 | 0.4400 | 0.4679 | inconclusive |
| smotenc | 0.1 | 0.5 | 0.4417 | 0.4658 | inconclusive |
| smotenc | 0.1 | 1 | 0.4352 | 0.4677 | inconclusive |
| smotenc | 0.25 | 0.25 | 0.4376 | 0.4661 | inconclusive |
| smotenc | 0.25 | 0.5 | 0.4316 | 0.4608 | reliable loss |
| smotenc | 0.25 | 1 | 0.4222 | 0.4465 | reliable loss |

## Evidence

- 96 successful model fits across the three weights; all report `weight_applied=True`.
- No synthetic rows in validation or test in any run.
- Weight 1.0 exactly reproduces all 12 selected-model validation/test prediction files from the earlier volume grid at the overlapping fills.
- CTGAN V2 uses nested prefixes of `bank_ctgan_v2_fill25.csv`; SMOTENC uses one seed-42 pool sized for 25% fill, as in the fine volume grid.
- `grid.csv` includes each selected model, validation/test PR-AUC, lifts, confidence intervals, verdict, and protocol id.
- `comparisons.csv` saves 95% paired-bootstrap intervals over cleaned: 1,000 stratified resamples, bootstrap seed 0, using the merged #138 helpers.
- `runs.json` records run ids. Prediction/model artifacts remain in untracked `runs/`.
- The three weight directories hold full model metrics and run timings.
- `bank_weights.png` shows both arms, all weights and fills, both evaluation splits, and the cleaned reference.

Reproduce with the three-weight `training_grid` loop in `docs/training.md`. Use separate results directories per weight. The experiment is an offline comparison with existing CTGAN pools, not a new quality-gate acceptance run.

Default remains separate: the committed Default V2 pool predates the merged #136 recipe; a current-recipe repeat needs its new pool/run artifacts or fresh generation. No Default result is claimed here.

No issue comments or shared-deck changes were published.
