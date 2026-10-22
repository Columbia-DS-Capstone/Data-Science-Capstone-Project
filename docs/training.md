# Model Training Agent

`agents.training.run_training(request, run)` implements the orchestrator's
`TrainingRequest -> TrainingOutput` handoff after the quality gate. The graph uses
this callable by default. Other stages can be supplied through `build_graph(agents=...)`;
the placeholder preprocessing/generator stages do not produce production training inputs.

## Inputs and frozen settings

The `raw` and `baseline` `ArmFiles` contain train, validation and test CSVs, **before
encoding**, with the target and a `source_row` column. Raw tables match `load_dataset`
output; baseline tables match the shared `Preprocessor` fitted on real training rows.
Bank labels can be `yes`/`no` in raw files and must be 0/1 in cleaned files. Numeric
category codes are normalized on reading CSVs. `source_row` is never a feature.

The agent validates disjoint, unique row IDs; source-file and split hashes; labels
and row order; dataset settings; package versions; model parameters; and the protocol
checksum. It also compares supplied raw/cleaned table contents to the source and
training-only cleaning. Any synthetic flag in a real file must be exactly zero;
missing or ambiguous flags are rejected. This prevents an unflagged replacement
row or a changed holdout from silently entering the comparison.

The default reference is
`results/baseline_protocol_<dataset>_<split_seed>_cv3.json` from issue #52. Override
its location with `RunConfig.training_protocol_path` or the `protocol_path=` keyword.
The models are gradient boosting, LightGBM, CatBoost, stacking, and the neural net.
`candidate_models` must match that complete suite. No tuning or probability
calibration occurs here; feature exclusions, changed splits, or changed settings
require a new reference experiment.

The protocol's seed fixes the data split and original-validation F1 thresholds.
`RunConfig.seeds` varies estimator/SMOTENC randomness **without resplitting or
recalibrating**. Thus each model uses the same numerical threshold across every
arm, fill level, and training seed. The stored protocol is copied into the run.
This is a fixed-split seed experiment, distinct from rerunning the baseline CLI
with a different split seed.

## Arms and saved results

| Schema arm | Results label | Training rows |
|---|---|---|
| `raw` | `original` | Raw real train, with the frozen minimum imputation/encoding/scaling |
| `baseline` | `cleaned` | Cleaned real train |
| `smote` | `smotenc` | Cleaned real train plus SMOTENC rows at each configured fill level |
| `gan` | `ctgan` | Cleaned real train plus each quality-accepted synthetic set |

SMOTENC is regenerated for each seed. Within a seed, the fill levels use prefixes
of one generated pool. Counts use `fill_rows(level, positives, negatives)` from
the real cleaned training split. A level that adds zero rows has explicit skipped
records. CTGAN inputs are already generated and quality-accepted; the Training Agent
does not refit the main generator. The same accepted pool is used across model seeds,
so that seed spread does not measure CTGAN training randomness. No accepted sets means
the other three arms still run. Synthetic tables must contain exactly the cleaned
columns plus `is_synthetic=1`, have the declared count, and have positive-class labels.

`ArmEncoder` learns encoding and scaling from real training rows only, including
when cloned inside stacking folds. Synthetic rows do not set category vocabularies
or numeric scaling. Cleaned tables retain the shared training-fitted imputations;
they are not repaired again using synthetic rows. For repeated outer CV, cleaning
is relearned from the raw rows inside each outer training fold.

Each `ModelRecord` carries status, hyperparameters, threshold, protocol ID, row
counts, validation/test metrics, a model artifact, and both prediction paths.
Failures have a reason and make the overall `TrainingOutput` fail; successful fits
are retained for diagnosis. Warnings such as non-convergence are recorded per model.

Artifacts live under the run's `models/` directory and are returned for manifest
registration:

- One `.joblib` dictionary per arm/level/model/seed: `estimator` (including its
  encoder), `features` (input column order), `threshold`, and `protocol_id`.
  Inference inputs omit target/row IDs and include `is_synthetic=0` for real rows.
- Validation and test prediction CSVs: `source_row`, `y_true`, `probability`,
  `prediction`, `threshold`, and `is_synthetic=0`. The decision rule is `p >= threshold`.
- `metrics.csv`: one row per split and model combination, with PR-AUC, F1,
  precision, recall, ROC-AUC, and accuracy last. PR-AUC uses the same trapezoidal
  precision-recall integration as #52, not average precision.
- `validation_summary.csv`: mean, sample standard deviation, and count over seeds,
  grouped by arm/fill/model. A single seed has an undefined standard deviation,
  not an artificial zero. This is the validation-only input for the future #36
  volume rule; this agent does not choose a volume.
- `training_protocol.json`: the exact frozen reference.

The selected model per arm/fill/seed maximizes real validation PR-AUC; ties retain
suite order. Test predictions and metrics are saved for **all** fitted models as
requested by the agent contract, unlike the winner-only baseline CLI. Test metrics
are descriptive and never choose models, thresholds, fill levels, or seeds.

## Optional repeated cross-validation

Set `training_cv_folds` to at least 2 and `training_cv_repeats` to a positive integer
(default 3). Zero folds, the default, disables it. Repeated stratified folds combine
real train and validation rows and never include test. Row partitions are shared
across arms and model seeds. Each outer fold refits cleaning from raw fold-training
rows, refits encoding on real rows, and regenerates SMOTENC only from fold training.
This option applies to the currently supported stratified datasets; a future
chronological fraud protocol needs a corresponding time-aware evaluation design.

The agent saves `cv_metrics.csv` and `cv_predictions.csv` with repeat/fold identifiers
and real row IDs. These report threshold-independent PR-AUC and ROC-AUC. They do not
report CV F1: the frozen threshold already used original validation labels, so F1
on pooled train+validation would reuse calibration labels. CV is a supplementary
stability analysis, not a replacement test set or a set of independent replicates.

The accepted full-training CTGAN rows **cannot** be reused inside CV because some
of their generator's training rows would lie in the CV holdout. For CTGAN CV, supply
`cv_gan_factory(cleaned_fold_train, row_count, seed)`. The factory must fit a fresh
generator, apply its quality checks using only the supplied fold training data,
and return a flagged synthetic DataFrame. It receives no holdout/test rows. Without
this factory, CTGAN CV gets explicit `SKIPPED` rows and an output warning; the main
quality-accepted CTGAN experiment still runs normally.

```python
from agentic_gan.agents.training import run_training

# request is a validated TrainingRequest; run is the orchestrator's RunDir.
request = request.updated(config=request.config.updated(
    seeds=[42, 43, 44],
    training_cv_folds=3,
    training_cv_repeats=3,
))
output = run_training(request, run)
```

Tests in `tests/test_training_agent.py` cover all arm/level/seed combinations,
frozen thresholds, serialization and manifest registration, provenance rejection,
failure propagation, deterministic reruns, fold-local generation, a real five-model
smoke run, and the orchestrator's filtering of quality-rejected fill levels.
