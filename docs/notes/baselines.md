# Baseline results

All three datasets were run at full size with seed 42 and stratified 60/20/20 splits, including three-fold training CV. The eight notebook models were retained and a neural net was added (nine models). The MLP uses one 100-unit hidden layer, max 300 iterations, and early stopping on an internal 10% training subset. The original eight keep their notebook settings.

Selection uses validation **trapezoidal PR-AUC**, matching the notebook, rather than average precision. Threshold is fixed at 0.5. Only the validation winner is evaluated on test; empty test cells for other models in `results/baselines.csv` are intentional. The CSV contains 54 rows: nine models × two arms × three datasets. JSON files beside it retain full metrics, warnings, settings, source/split hashes, and versions. These are single-seed comparisons; CV SD is across training folds.

## Results

| Dataset / arm | Selected model | Test PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| bank_marketing / cleaned | Stacking ensemble | 0.489 | 0.808 | 0.360 | 0.674 | 0.246 | 0.898 |
| bank_marketing / original | Stacking ensemble | 0.476 | 0.794 | 0.356 | 0.690 | 0.240 | 0.898 |
| default / cleaned | Gradient boosting | 0.568 | 0.788 | 0.468 | 0.676 | 0.358 | 0.820 |
| default / original | Gradient boosting | 0.568 | 0.788 | 0.468 | 0.676 | 0.358 | 0.820 |
| polish / cleaned | LightGBM | 0.835 | 0.974 | 0.718 | 0.938 | 0.581 | 0.978 |
| polish / original | LightGBM | 0.835 | 0.974 | 0.718 | 0.938 | 0.581 | 0.978 |

`original` applies minimum required numeric imputation/scaling and categorical imputation/encoding. `cleaned` uses the shared preprocessing rules. Both use training-only fitting, with preprocessing refitted in each outer CV fold. Stacking shares that preprocessing across its inner folds, as in the notebook. Both Bank arms exclude duration.

## Class imbalance and observations

- **Bank:** 3,173/27,126 training positives (11.70%). Cleaning increases test PR-AUC from 0.476 to 0.489; recall remains low at the fixed 0.5 threshold.
- **Default:** 3,982/18,000 training positives (22.12%), no missing cells. Both arms select gradient boosting. Small differences reflect category representation/order and tree tie-breaking, not evidence of a substantive cleaning benefit. Undocumented codes remain distinct categories.
- **Polish:** 1,255/26,043 training positives (4.82%). There are 24,708 missing training cells; A37 is 43.29% missing. A27 is missing in 34.50% of bankrupt training rows versus 4.90% of non-bankrupt rows. Both arms use median imputation and retain extreme ratios (maximum absolute training value 11,903,000). Imputation sensitivity remains a future comparison. The neural net has near-zero validation recall and predicts no positives in some CV folds; accuracy near 95% is therefore misleading. Its warnings are retained in the run JSON.

## Reuse

```python
from agentic_gan.tools.baselines import run_models

results = run_models(
    X_train, y_train, X_val, y_val, X_test, y_test,
    seed=42, preprocessor=None, cv_folds=3,
)
```

Returns a pandas DataFrame. Pass binary 0/1 labels and encoded features with matching column order; exclude row IDs. For raw data, pass an unfitted sklearn-compatible preprocessor for outer-fold-local fitting. With already encoded inputs, CV uses those supplied encodings. Add synthetic rows to training only, using the real-training-fitted encoder, and keep real holdouts/settings unchanged. `agents/training.py` exposes `run_training_arms` for repeated calls with shared holdouts and settings.

Use Python 3.12 and `uv sync --frozen`, then the README commands. Keep the same raw inputs, dataset configuration, seed (42), and CV folds (3) to reproduce these runs. Fraud and GAN-augmented/SMOTE experiments are not run here.
