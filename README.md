# Data-Science-Capstone-Project

DS Capstone — Agentic GAN Framework for Synthetic Financial Data Generation, Data Repair, and Bank Marketing Response Modeling

Columbia University — M.S. in Data Science
Course: Data Science Capstone
Industry Partner: TD Bank
Domain: Finance

Focus Areas: Generative AI · Machine Learning · Predictive Modeling · Reinforcement Learning

Overview

This project was developed as part of the Data Science Capstone for the M.S. in Data Science at Columbia University in collaboration with TD Bank.

The goal is to build an agentic generative modeling framework for financial tabular data that can:

generate high-fidelity synthetic financial/customer data,

identify and repair incomplete or low-quality records,

evaluate whether synthetic data preserves the statistical structure of real data, and

improve downstream bank marketing response prediction under realistic data-quality and class-imbalance constraints.

The project combines Generative Adversarial Networks (GANs), statistical data-quality evaluation, supervised machine learning, and agent-based orchestration into a reproducible end-to-end workflow.

Problem Statement

Financial institutions often work with datasets that contain missing values, imbalanced outcomes, limited samples, and privacy-sensitive customer attributes. These constraints can reduce the effectiveness of predictive models and make data sharing or experimentation difficult.

This capstone investigates whether synthetic-data generation and automated data-repair workflows can improve model development while preserving important properties of the original dataset.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
   - **Mac users:** also run `brew install libomp` (needed by XGBoost and LightGBM).
2. From the repo root, run `uv sync`. This creates `.venv` with the exact package versions in `uv.lock`.
3. Run code with `uv run ...`. For notebooks, run `uv run jupyter lab`, or select `.venv` as the kernel in VS Code.

To add a package, run `uv add <package>`, then commit both `pyproject.toml` and `uv.lock`. Don't use pip, conda or a requirements.txt.

Run the placeholder orchestrator with `uv run python -m agentic_gan.orchestrator --dataset bank_marketing`. It creates a folder under `runs/` with typed stub outputs, a manifest, `decisions.jsonl`, and a report. The stubs do not train models or evaluate the dataset.
Print its LangGraph Mermaid diagram with `uv run python -m agentic_gan.orchestrator --diagram`.

## Repository layout

- `src/agentic_gan/`: the system (orchestrator, agents, tools, schemas)
- `docs/`: architecture and design docs
- `eda/`: exploratory notebooks and baseline models
- `tests/`: automated tests
- `runs/`: pipeline run outputs, created at runtime and not tracked

Academic Context

This repository contains work completed for the Data Science Capstone within the M.S. in Data Science program at Columbia University.

The project is an academic collaboration with TD Bank and is intended for research, experimentation, and educational purposes. Any proprietary, confidential, or restricted data should not be committed to this public repository.

Contributors

Columbia University — M.S. in Data Science
Capstone Team — TD Bank Industry Project

Add team member names, faculty advisor, and industry mentor here.

License / Usage

This project was created for academic purposes. Usage of project code, data, or derived artifacts should comply with Columbia University requirements, applicable licenses, and any confidentiality or data-use agreements associated with the industry-sponsored capstone.

## Shared preprocessing

One script prepares Bank Marketing, Default, and Polish Bankruptcy data:
`src/agentic_gan/tools/preprocessing.py`.

### Configuration

Edit [datasets.json](src/agentic_gan/utils/datasets.json) to set each dataset's input path, target column, categorical columns, imputation method, and split settings. Defaults are stratified **60% training / 20% validation / 20% test**, **seed 42**, and **median imputation**. Bank's post-call `duration` feature is excluded by default.

Place the raw inputs at the configured paths (relative to the repo root):

| Dataset | Default input |
|---|---|
| Bank Marketing | `data/raw/bank_marketing.csv` |
| Default | `data/raw/default.csv` |
| Polish Bankruptcy | `data/raw/polish_companies_bankruptcy_uci365.zip` |

The script reads local inputs; it does not download them. See the [input format details](docs/preprocessing.md) before preparing the files.

### Run

After `uv sync`, run from the repo root:

```bash
uv run python -m agentic_gan.tools.preprocessing --dataset bank_marketing
uv run python -m agentic_gan.tools.preprocessing --dataset default
uv run python -m agentic_gan.tools.preprocessing --dataset polish
```

The script generates splits directly from the config. The same input rows in the same order, settings, and software versions reproduce the same splits. Imputation, category encoding, and scaling are learned **only from training rows**.

Optional overrides include `--config /path/to/datasets.json`, `--input /path/to/source`, `--imputation mean`, and `--include-duration` for a post-call Bank diagnostic.

### Outputs

Results are written to `data/interim/<dataset>/`:

Each dataset folder contains only:

- `train_cleaned.csv`
- `validation_cleaned.csv`
- `test_cleaned.csv`

These tables contain corrected and imputed features before encoding/scaling. `source_row` identifies source rows and must be excluded from model/GAN features. The baseline runner fits encoding and scaling in memory using training rows only. Rerunning preprocessing replaces the cleaned files and removes legacy generated artifacts. Raw inputs and generated outputs are ignored by Git.

Fraud preprocessing is deferred.

See [preprocessing setup and tool API](docs/preprocessing.md) for dataset treatments and preprocessing tool reuse.

## Baseline models

Compare original and cleaned data with five fixed models: gradient boosting, LightGBM,
CatBoost, stacking, and a neural net. Use Python 3.12 and `uv sync --frozen` first:

```bash
uv run python -m agentic_gan.tools.baselines --dataset default --arm both
uv run python -m agentic_gan.tools.baselines --dataset polish --arm both
uv run python -m agentic_gan.tools.baselines --dataset bank_marketing --arm both
```

Run these commands sequentially because they share `results/baselines.csv`. On the first
original-arm run, each model's threshold maximizes **validation F1**. Those numerical
thresholds are saved in `results/baseline_protocol_<dataset>_42_cv3.json` and reused for
cleaned and augmented arms. An existing protocol is never silently overwritten or retuned.
Run `--arm both` or `--arm original` before running cleaned alone. Changed input data,
splits, model settings, or dependency versions require a separate experiment directory.

Both arms read raw inputs. Preprocessing fits inside training CV folds and stacking's
inner folds; the final fit uses training rows only. Models are ranked by validation
trapezoidal PR-AUC. Only the validation winner gets test metrics. CSV metrics prioritize
PR-AUC and F1, with accuracy last. Non-winner test cells are intentionally empty. The
per-run JSON files retain diagnostics, model settings, source/split hashes, and versions.
Defaults remain seed 42 and three-fold training CV; CV reports threshold-independent
PR-AUC and ROC-AUC. Matching dataset/arm/seed/CV rows are replaced on rerun.

The Training Agent consumes the frozen protocol:

```python
from agentic_gan.agents.training import run_training_arms

results = run_training_arms(
    training_sets, X_val, y_val, X_test, y_test,
    protocol_path="results/baseline_protocol_bank_marketing_42_cv3.json",
    seed=42, cv_folds=3,
)
```

Supply the same real holdouts and feature schema for every arm. Synthetic rows belong
only in the training inputs and use the real-training-fitted encoder. For raw data,
supply an unfitted `preprocessor=`; for encoded data, omit it. The lower-level
`run_models(..., thresholds=protocol["thresholds"])` API is also available. Omitting
thresholds is reserved for original-arm calibration, not for comparison arms.

The augmented runner also requires frozen protocols and the full five-model suite.
It now defaults to seed 42, the seed frozen here. Additional seeds require reference
protocols calibrated on the same real holdouts; changed holdout labels/order are rejected.
It writes `results/augmented_bank_marketing_frozen.csv`, keeping the historical 0.5
results separate. This task does not rerun the historical augmentation experiments.

See [baseline results and rationale](docs/notes/baselines.md) for the model choices,
thresholds, measured results, and dataset-specific explanations.
