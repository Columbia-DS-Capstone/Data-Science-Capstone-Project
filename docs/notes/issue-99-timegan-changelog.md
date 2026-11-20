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

## Entry 2 — 2026-11-20 — Step 2 networks and training

### Built

In `src/agentic_gan/tools/timegan.py`:

- Five networks on one shared base (`_SequenceNet`: a batch-first GRU, then a linear layer at
  every step): `Embedder`, `Recovery`, `Supervisor` (one GRU layer fewer), `SequenceGenerator`,
  `Discriminator` (one logit per step, no sigmoid).
- Loss helpers `_reconstruction_loss`, `_supervised_loss`, `_moment_loss`, `_adversarial_loss`.
- `_autoencoder_phase`, `_supervised_phase`, `_joint_phase`, and `train_timegan` with the
  issue's exact signature.
- Constants `PHASES`, `LOSS_COLUMNS`, `GENERATOR_STEPS = 2`, `DISCRIMINATOR_THRESHOLD = 0.15`.

In `tests/tools/test_timegan.py`: 34 new tests, 103 in total.

### Worked

```
uv run pytest tests/tools/test_timegan.py -q
103 passed in 11.80s

uv run ruff check src/agentic_gan/tools/timegan.py tests/tools/test_timegan.py
All checks passed!

uv run ruff check src tests
Found 51 errors.   (unchanged baseline, all in src/agentic_gan/web.py)
```

- 2 epochs on the toy table gives 6 loss rows in phase order; the log file equals
  `loss_history`; a loss a phase does not compute is NaN.
- The same seed gives identical loss history and bit-identical weights in all five networks;
  a different seed changes every network.
- Five Adam optimisers are created and hold the right networks. With 2 steps per epoch and
  2 epochs they step 4, 4, 8, 8 and 4 times (autoencoder, supervisor, joint generator, joint
  embedder, discriminator). With the threshold raised, the discriminator steps 0 times.
- A non-finite loss raises `RuntimeError` naming the phase and epoch, tested in each phase.
- One-off check on real rows, not committed: 10 epochs per phase on the first 2,000 rows of
  the Default train split. Every loss fell: autoencoder 3.33 to 2.49, supervised 0.070 to
  0.0025, joint generator 32.6 to 16.2. It took 16.6 s at 15 steps per epoch.

### Failed

Nothing failed in this step.

### Decisions

- **Five optimisers, as in the reference code:** autoencoder (embedder + recovery),
  supervisor, joint generator (generator + supervisor), joint embedder (embedder + recovery),
  discriminator. Each keeps its own Adam state.
- **What the loss columns hold.** `Embedder Loss` is the loss the embedder update minimised:
  `10 * sqrt(MSE)` in the autoencoder phase, and that plus `0.1 *` supervised loss in the
  joint phase. `Supervised Loss` is the plain next-step MSE in both phases that compute it.
  `Discriminator Loss` is recorded at every joint step, whether or not the update ran.
- **`sqrt` floor.** `sqrt(loss + 1e-8)` instead of `sqrt(loss)`, so the gradient stays finite
  if a loss ever reaches exactly zero. It does not change the loss values at the precision
  logged.
- **Standard deviation in the moment loss** is `sqrt(batch variance + 1e-6)` with the
  population variance, as in the reference code.
- **In the generator update the embedder output is computed without gradient.** The embedder
  is not updated in that step, so this changes nothing but the cost.
- **`GENERATOR_STEPS` and `DISCRIMINATOR_THRESHOLD` are module constants**, not `DEFAULTS`
  keys, so the Step 1 `DEFAULTS` test stands unchanged. Both are recorded in `model.params`,
  with the resolved `noise_dim`, `num_layers`, `gamma`, `lr` and `steps_per_epoch`.
- **`seed` must be a whole number**; anything else is a `ValueError`.
- Added a test that 15 epochs lower the autoencoder and supervised losses, as a guard against
  a sign or wiring error.

### Open questions

- **Wall time, rough only.** The full train split has 140 steps per epoch against 15 in the
  2,000-row check. If time scales with steps, 10 epochs per phase is about 2.6 minutes, 100
  about 26 minutes and 300 about 78 minutes. Step 4 measures this on the full split.
- **The autoencoder loss flattened at about 2.47 after 75 steps in the 2,000-row check**, then
  fell again to 1.18 in the joint phase. Whether it breaks through with more steps is
  something the full run's loss curve will show.

## Entry 3 — 2026-11-20 — Step 3 sampler

### Built

- `src/agentic_gan/tools/timegan.py`: `_generate` (uniform noise, then generator, supervisor,
  recovery, `from_sequences`) and `sample_timegan` with the issue's signature. The round loop
  is a copy of `sample_wgan_gp` (`wgan_gp.py` lines 490 to 600): each round asks for
  `need / max(keep rate, 0.02) * 1.2 + 10` rows, capped at 100,000; applies `row_filter`;
  drops duplicates when `drop_duplicates_of` is given; stops at `n`; after `max_rounds` it
  logs a warning and returns the rows it has. `wgan_gp.py` is untouched.
- Output: the training frame's columns in order with their dtypes, then `is_synthetic = 1`.
- `tests/tools/test_timegan.py`: 52 new tests in `TestSampleTimeGAN`, `TestRowFilter`,
  `TestDropDuplicates` and `TestSampleArguments`, and a `varied` fixture (see Failed).

### Worked

- `uv run pytest tests/tools/test_timegan.py`: 155 passed in 18.45 s, run outside the
  sandbox. This is pass criterion 1 of the issue.
- `uv run ruff check src/agentic_gan/tools/timegan.py tests/tools/test_timegan.py`: all checks
  passed. Whole tree: still the 51 baseline errors in `src/agentic_gan/web.py`.
- Every test in the step's list is there: columns and order, dtypes, `n` rows, valid static
  categories, `PAY_*` in `DEFAULT_PAY_CODES`, zero `DEFAULT_RULES` violations, same seed same
  rows, different seed different rows, weights unchanged, save then load gives the same
  sample, a `y == 1` filter returns only `y == 1`, no returned row equals a training row, an
  impossible filter with `max_rounds=2` returns an empty frame with the right columns and
  dtypes, `n <= 0` raises.
- Guards beyond the list: the rows equal a hand-computed noise, generator, supervisor,
  recovery pass; sampling neither reads nor advances torch's global random state; networks
  stay in eval mode; the request sizes are 34 then 1,210 when nothing survives, and never
  above 100,000; duplicates within one sample are dropped.
- The tests were checked by breaking the sampler in seven ways (skip the supervisor, flag 0,
  ignore the filter, no in-sample duplicate check, ignore the seed, seed the global state,
  wrong keep-rate floor). Each break failed a test. The breaks were not committed.

### Failed

- **The 2-epoch toy model returns one row repeated.** 1,000 samples gave 1 distinct row, no
  `y == 1`, and a standard deviation of 0.0000 at the recovery output. The tests for
  "different seed, different rows" and "a `y == 1` filter" cannot pass on that model, and
  they should not be weakened. So those tests use a second fixture, `varied`: the same toy
  table, 12 epochs per phase at batch size 32 (108 steps per phase, about 5.5 s). It gives
  400 distinct rows in 400 and both classes. The fixture asserts this itself, so it fails
  loudly if a later change makes the model collapse again. The validity tests run on both
  models.
- **The collapse is under-training, not a wiring fault.** Variety grows with training:

  | Data | Epochs per phase | Distinct rows in 1,000 | `y == 1` share (real) | Distinct values: SEX, EDUCATION, MARRIAGE, PAY_1, AGE | Std at recovery (real) |
  | --- | --- | --- | --- | --- | --- |
  | toy, 300 rows | 2 | 1 | 0.000 (0.220) | 1, 1, 1, 1, 1 | 0.0000 (0.3464) |
  | toy, 300 rows | 20 | 1,000 | 0.000 (0.220) | 1, 1, 1, 1, 1 | not recorded |
  | toy, 300 rows | 60 | 1,000 | 0.412 (0.220) | 2, 3, 4, 2, 8 | 0.1508 (0.3464) |
  | real, first 2,000 rows | 10 | 1,000 | 0.260 (0.225) | 2, 1, 2, 3, 6 | 0.1024 (0.2028) |
  | real, first 2,000 rows | 40 | 1,000 | 0.187 (0.225) | 2, 1, 2, 4, 29 | 0.1529 (0.2028) |

  Batch size 128 throughout, so the toy rows are 2 steps per epoch and the real rows 15.

### Decisions

- **Differences from `sample_wgan_gp`, all deliberate:**
  - It does not call `torch.manual_seed`. The noise comes from a fresh `torch.Generator`
    seeded with `seed`, so sampling leaves the caller's random state alone.
  - The generator is created on the model's device.
  - `max_rounds < 1` is a `ValueError`, so there is always at least one round.
  - The filter's result goes through `np.asarray(..., dtype=bool)`, so a Series or an array
    both work.
  - `drop_duplicates_of` without one of the model's columns is a `ValueError`. Extra columns
    are ignored.
- **Same as `sample_wgan_gp`:** duplicates, both against `drop_duplicates_of` and within the
  sample, are dropped only when `drop_duplicates_of` is given. Without it, repeated rows are
  returned. The Step 5 runner passes the train frame, so the fill file gets both checks.
- **`row_filter` sees the decoded frame without `is_synthetic`**, the same columns as the
  training frame.
- The test file now takes about 18 s, up from about 12 s, because of the `varied` fixture.

### Open questions

- **Category variety is the main risk for the full run.** After 40 epochs on 2,000 real rows
  the samples still had one EDUCATION value where the real rows have several, and the spread
  at the recovery output was 75% of the real spread. The full run has 140 steps per epoch, so
  it trains far more, but the fidelity scorecard in Step 6 may fail on the categorical
  columns. The issue says the scorecard need not pass the gate; the result is still to be
  reported as measured.
- **Share of `y == 1` rows moves with training** (0.26, then 0.19 on real rows). The fill
  needs 1,004 positive rows, so the runner's `y == 1` filter may discard 75% to 80% of the
  rows drawn. The sampler's overshoot handles that; the cost is sampling time, which Step 5
  will show.
