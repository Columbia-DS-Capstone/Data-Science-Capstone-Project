# Issue #99: TimeGAN generator for Default — change log

Issue: https://github.com/Columbia-DS-Capstone/Data-Science-Capstone-Project/issues/99
Branch: `arrico/timegan-generator-for-default-99`, cut from `main` at `fedde3a`.

## How to read this file

One entry per step, appended in order. Each entry has the same five parts: built, worked,
failed, decisions, open questions. Read the whole file before starting a step.

## The issue in one paragraph

Build `src/agentic_gan/tools/timegan.py`: a TimeGAN that trains on the Default monthly block
and samples synthetic customers in the same shape as the other generators. Each customer is a
sequence of 6 steps with 9 features per step (3 monthly features, 5 static columns, the
target). Then run it on the full Default train split, write a 10% fill file in the CTGAN
layout, and post the wall time, loss curves, fidelity scorecard and PR-AUC next to CTGAN and
cleaned. Wiring into selection is issue #137 and is out of scope.

## Standing decisions

These hold for every step unless an entry below changes them.

- `epochs` means epochs per phase, so the loss log has 3 x `epochs` rows.
- Steps are chronological: month 6 (April) first, month 1 (September) last.
- The tool is dataset-agnostic. It learns valid codes, ranges and dtypes from the training
  frame and never imports Default constants. Default names live in the runner and the tests.
- The defaulters-only filter (`y == 1`) is a `row_filter` passed by the runner, not part of
  `sample_timegan`.
- `selection.py`, `agents/*`, `wgan_gp.py`, `ctgan*.py`, `training_grid.py`,
  `quality_fidelity.py` and `results/` are not edited.

## Entry 0 — 2026-11-20 — Step 0 preflight

### Built

Nothing in `src/` or `tests/`. This file only.

### Worked

Baseline, run outside the sandbox:

```
uv run pytest tests/tools/test_wgan_gp.py tests/test_quality_fidelity.py tests/tools/test_training_grid.py -q
106 passed, 1 warning in 27.48s
```

Facts confirmed, with the lines they come from:

- **Month order.** `docs/dataset_summaries.md` lines 86 to 103: `PAY_1`, `BILL_AMT1` and
  `PAY_AMT1` are September 2005; `PAY_6`, `BILL_AMT6` and `PAY_AMT6` are April 2005.
  `DEFAULT_WIDE_TEMPORAL` (`selection.py` lines 27 to 31) lists months 1 to 6, newest first.
  Chronological `step_columns` is therefore month 6 down to month 1.
- **Train split.** `data/interim/default/train_cleaned.csv`: 18,000 rows, 25 columns, all
  `int64`, no missing values. Columns are `source_row`, the 5 static columns, the 18 monthly
  columns, `y`. Positives 3,982, negatives 14,018, gap 10,036, rate 22.12%.
- **10% fill count.** `fill_counts` (`utils/synthetic.py` lines 14 to 20) and `fill_rows`
  (`schemas.py` lines 67 to 69) both give 1,004.
- **Rules on real rows.** No real train row breaks any of the three `DEFAULT_RULES`
  (`quality_fidelity.py` lines 128 to 146).
- **Codes.** The union of `PAY_1..6` values in train is -2 to 8, equal to `DEFAULT_PAY_CODES`
  (`quality_fidelity.py` line 111). Distinct values: `SEX` 2, `EDUCATION` 7, `MARRIAGE` 4,
  `y` 2, `LIMIT_BAL` 77, `AGE` 55. With `CODED_MAX_VALUES = 12` (`ctgan_sampling.py` line 41)
  the coded features are `SEX`, `EDUCATION`, `MARRIAGE`, `y` and repayment status; `LIMIT_BAL`,
  `AGE`, bill amount and payment amount are continuous.
- **Ranges over all six months in train.** Bill amount -209,051 to 983,931. Payment amount
  0 to 1,215,471.
- **Synthetic file checks.** `_synthetic` in `agents/training.py` lines 595 to 616, called
  from lines 1034 to 1040:
  1. row count equals `fill_rows` for the level (lines 596 to 597, and 1035 to 1037);
  2. the column set equals the cleaned train columns plus `is_synthetic` (lines 598 to 599).
     `_read_real` moves `source_row` into the index (lines 426 to 433), so the file must not
     have a `source_row` column;
  3. every `is_synthetic` is 1 and every target is 1 (lines 600 to 601);
  4. no missing values (lines 602 to 603 and 614 to 615);
  5. every numeric feature is finite (lines 612 to 613).
- **`training_grid`.** The file name is `ROOT / f"{ctgan_prefix}{round(level * 100)}.csv"`
  (line 96). Without `--no-save`, `save_results` (lines 118 to 139) rewrites
  `results/training_all_arms.csv` and `results/training_run_times.csv`, replacing every row
  of the dataset (line 138). The GAN arm is always labelled `ctgan`
  (`agents/training.py` line 62).
- **`wgan_gp.py` patterns to mirror.**
  - Seeding: `torch.manual_seed` and `np.random.seed` before any fitting (lines 351 to 352),
    then a dedicated `torch.Generator` for batch indices and noise (lines 382 to 383).
    Batches are drawn with replacement, `steps_per_epoch = max(n_rows // batch_size, 1)`
    (lines 385 and 401).
  - Divergence: a non-finite epoch mean raises `RuntimeError` (lines 434 to 435).
  - Sampler: a fresh `torch.Generator` from `seed` (lines 519 to 521); each round requests
    `min(int(need / max(keep_rate, 0.02) * 1.2) + 10, 100_000)` rows (line 537); applies
    `row_filter` (lines 566 to 568); drops rows equal to a `drop_duplicates_of` row or to an
    earlier kept row, only when `drop_duplicates_of` is given (lines 525 to 527 and 572 to
    581); keeps `head(need)` (line 583); updates `keep_rate` from rows that passed the filter
    (line 585); warns and returns the shortfall after `max_rounds` (lines 589 to 593).
  - Save and load: plain pickle, networks moved to CPU first (lines 291 to 307).

### Failed

- **Sandbox only.** Inside the sandbox two `TestLearnsSomething` tests error with
  `PermissionError` from joblib (`os.sysconf("SC_SEM_NSEMS_MAX")`). They pass outside it.
  Test runs for this issue need to run outside the sandbox.
- **Ruff is not clean on `main`.** `uv run ruff check src tests` reports 51 errors, all in
  `src/agentic_gan/web.py` (48 E501, 1 E741, 1 E731, 1 I001). That file is out of scope.
  The check for this issue is: the count stays 51 and no error is in a TimeGAN file.
- **`gh` is not logged in.** `gh auth status` says "You are not logged into any GitHub
  hosts". Posting to the issue and opening the PR need `gh auth login` or a manual paste.
  Reading the issue works through the public API.

### Decisions

- Run tests outside the sandbox.
- Treat the 51 `web.py` ruff errors as the baseline and do not fix them here.

### Open questions

- **Heavy-tailed amounts under min-max scaling.** The issue fixes min-max scaling per
  feature over all steps. Payment amount spans 0 to 1,215,471, so most rows scale to values
  very close to 0. The bill and payment amount columns are the likeliest fidelity failures.
  No change planned; this is what the scorecard in Step 6 will show.
- **Exact duplicates in train.** 14 real train rows are exact duplicates of another train
  row. This does not affect training; the sampler's `drop_duplicates_of` handles copies.
- **Schedule.** The issue's checkpoints (Nov 16, 18, 19) and the Thu Nov 19 21:00 merge
  freeze have passed. Whether the PR may merge during the freeze is Kyle's call.

## Entry 1 — 2026-11-20 — Step 1 data layer

### Built

`src/agentic_gan/tools/timegan.py` (new), with no networks yet:

- `DEFAULTS` (`num_layers` 3, `gamma` 1.0, `lr` 1e-3, `noise_dim` None, meaning the width of
  the feature vector), `HIDDEN_DIM_RANGE = (24, 64)`, `DEVICES`.
- `step_columns_from_wide(wide, n_steps, *, newest_first)`: feature-major names to one list
  per step, oldest first.
- `SequenceLayout` (frozen dataclass) and `fit_layout`. The per-feature fields follow the
  feature vector: step features (`step_feature_0..`), static columns, target.
- `to_sequences` and `from_sequences`.
- `_validate_frame` (called by `fit_layout`) and `_validate_args`.
- `TrainedTimeGAN` dataclass, `save_model`, `load_model`.

`tests/tools/test_timegan.py` (new): 69 tests.

### Worked

```
uv run pytest tests/tools/test_timegan.py -q
69 passed in 3.37s

uv run ruff check src/agentic_gan/tools/timegan.py tests/tools/test_timegan.py
All checks passed!

uv run ruff check src tests
Found 51 errors.   (unchanged baseline, all in src/agentic_gan/web.py)

uv run pytest tests/tools/test_timegan.py tests/tools/test_wgan_gp.py -q
135 passed, 1 warning in 23.57s
```

- The toy table and the real 18,000-row train split both round-trip exactly through
  `to_sequences` then `from_sequences` (values, column order and dtypes).
- On the real train split the layout is: tensor (18000, 6, 9), float32, range 0 to 1. Coded
  features are repayment status, `SEX`, `EDUCATION`, `MARRIAGE` and `y`. The learned status
  codes equal `DEFAULT_PAY_CODES`.
- Uniform random sequences, and sequences far outside 0 to 1, decode to rows that break no
  `DEFAULT_RULES`.

### Failed

- One test of mine had a wrong expectation: it assumed the toy table's smallest payment
  amount was 0, but the generated minimum was 9. The decoder was right. The toy table now
  sets about a fifth of the payment amounts to 0, as the real table does, and the fixture
  asserts the minimum is 0. No code change.

### Decisions

- Checks added beyond the step's list, each a `ValueError`: the target listed as a feature;
  duplicate column names; infinite values; fewer than 2 rows. `from_sequences` also refuses a
  wrong shape and non-finite values, so a diverged generator fails loudly instead of decoding
  to a valid-looking row.
- Snapping uses a sorted search, not a distance matrix, because the target is always coded
  and may have many values. A tie goes to the lower code.
- `noise_dim` is `None` in `DEFAULTS` and resolved to the feature width at training time.
- The settings checks are tested through `_validate_args` for now. Step 2 will exercise them
  through `train_timegan`.

### Open questions

- **Float32 precision.** An integer-valued feature decodes exactly only while its range is
  below 2**24 (about 16.7 million). Default's widest range is 1,215,471, so it is safe. The
  module docstring states the limit.
- **Amount scaling, now measured.** On the real train split the median scaled payment amount
  is 0.0016 and the median scaled bill amount is 0.19. This confirms the Step 0 concern that
  min-max scaling leaves the payment amounts crowded near 0.
