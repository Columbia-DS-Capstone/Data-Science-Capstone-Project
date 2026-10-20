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

Compare original and cleaned data with nine models. Use Python 3.12 and `uv sync --frozen` first:

```bash
uv run python -m agentic_gan.tools.baselines --dataset default --arm both
uv run python -m agentic_gan.tools.baselines --dataset polish --arm both
uv run python -m agentic_gan.tools.baselines --dataset bank_marketing --arm both
```

Omit `--arm both` to run only the cleaned arm. Both arms read raw inputs. The original arm uses minimum preparation; the cleaned arm applies shared cleaning rules. Imputation, encoding, and scaling are fitted within each training CV fold. Stacking base models also fit preprocessing within their inner folds. Final fitting uses training rows only; validation and test only receive the fitted transformations. Selection uses validation PR-AUC; the selected model receives test PR-AUC, ROC-AUC, F1, precision, recall, and accuracy. The compact `results/baselines.csv` contains validation PR-AUC, training CV PR-AUC mean/SD, and selected-model test metrics. Per-run JSON files retain full metrics, warnings, model settings, source/split hashes, and package versions. Seed defaults to 42 and CV folds to 3. Results accumulate in the CSV; matching dataset/arm/seed/CV runs are replaced on rerun. Run commands sharing this file sequentially.

The Training Agent and synthetic-data experiments can reuse:

```python
from agentic_gan.tools.baselines import run_models

results = run_models(X_train, y_train, X_val, y_val, X_test, y_test, seed=42)
```

Pass encoded features, or supply an unfitted `preprocessor=` for raw data. Synthetic rows belong only in the training inputs. See [baseline results and comparison notes](docs/notes/baselines.md) for measured results and comparison conventions.
