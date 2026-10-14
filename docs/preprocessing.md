# Shared preprocessing

Edit `src/agentic_gan/utils/datasets.json` to set each dataset's input file, target, categorical columns, imputation method, duration option, and train/validation/test ratios and seed. Defaults are stratified 60/20/20, seed 42, median imputation, and no Bank duration.

Run from the repository root:

```bash
uv run python -m agentic_gan.tools.preprocessing --dataset bank_marketing
uv run python -m agentic_gan.tools.preprocessing --dataset default
uv run python -m agentic_gan.tools.preprocessing --dataset polish
```

The script loads the raw data, splits it according to the config, learns preprocessing from training rows only, and writes results to `data/interim/<dataset>/`. No `splits/` directory or separate row-ID files are needed. The same input rows in the same order, config, and software versions reproduce the same split. These are generated splits, not recovered EDA assignments.

Main outputs:

- `train_cleaned.csv`, `validation_cleaned.csv`, `test_cleaned.csv`: cleaned tables for GAN use, before encoding/scaling.
- Corresponding `*_encoded.csv`: numeric tables for predictive models.
- `preprocessor.pkl`: fitted imputation, category encoding, and scaling for reuse on synthetic rows.
- `treatment_log.csv`: column treatments and reasons.

The output folder also contains the training profile, missingness masks, and a manifest recording settings, input hash, software versions, and split sizes. Default includes a category-code audit. Rerunning overwrites generated outputs. `source_row` identifies original row positions and must be excluded from model/GAN features.

Use `--config /path/to/datasets.json` for another configuration, `--input /path/to/source` for an input override, `--imputation mean` for a different imputation method, or `--include-duration` for a post-call Bank diagnostic. Paths in config are relative to the repo root; `--root` overrides that root.

Bank accepts the UCI API CSV or semicolon-delimited bank-full CSV. It renames the day column, preserves unknown categories, retains `pdays=-1`, and derives the contact flag. Default expects named UCI columns as in `pH7_eda.ipynb` (PAY_0 is accepted as PAY_1); undocumented codes are preserved as categories. Polish reads all five ARFF cohorts from the original ZIP in notebook loading order. Missing numeric values use the training median/mean, entirely missing numeric columns use zero, and categorical gaps use the training mode.

Raghav's original `eda/uci_banking/scripts/uci_preprocessing.py` is unchanged and independent. Its treatments differ from this pipeline. For comparisons using the new preprocessing, train the baseline on the shared encoded outputs.

```python
import pickle
from agentic_gan.tools.preprocessing import run_preprocessing

output = run_preprocessing('bank_marketing')
# Only load trusted project pickle files with compatible package versions.
with (output / 'preprocessor.pkl').open('rb') as f:
    fitted = pickle.load(f)
encoded_synthetic = fitted.transform_cleaned(synthetic_rows)
```

Fraud is deferred.
