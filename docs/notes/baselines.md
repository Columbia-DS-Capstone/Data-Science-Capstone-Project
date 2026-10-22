# Sprint 2 baseline freeze — 22 October 2026

## Fixed model suite

The suite is reduced from nine reported candidates to **five**, retaining the five
requested candidates. This is a deliberate exception to the mentor's example of one
model per family: CatBoost, LightGBM and classical gradient boosting are all boosted
trees. They test distinct tree-building choices; we do not claim five independent
algorithm families. Standalone logistic regression, decision tree, random forest and
XGBoost are removed. Logistic regression and random forest remain internal components
of the stack, not separate candidates in the results.

| Model | Why retain it | Frozen settings |
|---|---|---|
| Gradient boosting | A conventional sequential boosted-tree reference; shallow trees can capture nonlinear effects with less flexibility than leaf-wise growth. It was the Default validation winner in the preceding runs. | sklearn defaults: 100 trees, learning rate 0.1, depth 3 |
| LightGBM | Histogram-based, leaf-wise boosting can isolate useful interactions efficiently. It was the Polish validation winner; retaining it tests whether rare bankruptcy patterns benefit from more adaptive partitions. | 150 trees, other library defaults, 2 threads |
| CatBoost | A contrasting boosted-tree implementation using symmetric trees by default. This run tests that learner, not a native categorical-encoding advantage: the shared preprocessing one-hot encodes categories before fitting. | 200 iterations, other library defaults, 2 threads |
| Stacking ensemble | Combines linear, bagged-tree and boosted-tree probabilities with a logistic meta-learner. This tests whether complementary decisions help beyond a single learner; it was the Bank validation winner. | Logistic regression (max_iter=1000), random forest (100 trees), LightGBM (150 trees); 3 inner folds; logistic final estimator |
| Neural net | A non-tree comparison that learns distributed nonlinear relationships and tests sensitivity to scaling, sparse one-hot features and imbalance. It is retained even when it underperforms. | MLP, one 100-unit hidden layer, max_iter=300, internal 10% training early-stopping subset, patience 10 |

Constructor settings and random seeds are identical across arms. No hyperparameter search
was added. Constructor parameters and dependency versions are captured in each frozen
protocol so a library upgrade or settings change cannot silently alter the experiment. The stack fits its base
preprocessors separately within each inner training fold.

## Threshold selection and reuse

For each **dataset × model**, train on the original-arm training split and select the
threshold that maximizes positive-class F1 on its real validation split. Search every
unique predicted probability, use `p >= threshold`, and break an exact F1 tie by taking
the highest threshold (fewer predicted positives). Do not use test labels to choose a
threshold. Model selection remains highest validation trapezoidal PR-AUC.

The five numerical thresholds for a dataset are saved once in
`results/baseline_protocol_<dataset>_42_cv3.json`. Original reruns, cleaned data and future
augmented arms use these exact values. **There is one fixed threshold per dataset/model,
not one universal number for all probability scales, and no separate tuning per arm.**
This isolates the effect of changing training data. An arm's probabilities may shift,
so this fixed operating point need not maximize its own F1; that is part of the comparison.

A cutoff of 0.5 is a default decision rule, not an F1 optimum. With unequal prevalence
and imperfectly calibrated scores it can miss many positives. F1 gives equal emphasis
to precision and recall, which fits this experiment's minority-class detection objective;
it is not a bank-specific cost function. The CSV includes validation F1 at both the frozen
threshold and 0.5 to make the choice inspectable. Validation F1 used for threshold selection
is optimistic; the untouched test split estimates performance after the choice.

PR-AUC is threshold-independent and remains the first metric. We retain trapezoidal
PR-AUC to match prior results (it is not average precision). Training CV now reports
only PR-AUC and ROC-AUC: there is no misleading CV F1 computed at an implicit 0.5 cutoff.
Only the validation winner in each arm receives test metrics; the other test cells stay
empty. This preserves the existing test-selection policy rather than ranking models on test.

The protocol records model parameters, package versions, source/split hashes, configuration,
seed, CV settings, holdout-label order, thresholds and a checksum. Missing or incompatible
protocols fail before comparison training; they are never silently regenerated for another
arm. Source-row/feature alignment remains the caller's responsibility for encoded arrays.

## Measured rerun

All datasets use full-size, stratified 60/20/20 splits, seed 42, and three-fold training CV.
Both Bank arms exclude post-call duration. `original` uses minimum imputation, scaling and
encoding; `cleaned` uses the shared cleaning rules. All preprocessing fits on training rows,
including within outer CV and stacking's inner folds. Generated GAN CSVs are unchanged.

The final CSV contains **30 successful rows: five models × two arms × three datasets**.

| Dataset / arm | Validation winner | Threshold | Test PR-AUC | Test F1 | ROC-AUC | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bank_marketing / original | Stacking ensemble | 0.145609 | 0.4765 | 0.4989 | 0.7941 | 0.4632 | 0.5406 | 0.8729 |
| bank_marketing / cleaned | Stacking ensemble | 0.145609 | 0.4901 | 0.5079 | 0.8084 | 0.4650 | 0.5595 | 0.8732 |
| default / original | Gradient boosting | 0.255419 | 0.5678 | 0.5512 | 0.7877 | 0.5230 | 0.5825 | 0.7902 |
| default / cleaned | Gradient boosting | 0.255419 | 0.5677 | 0.5512 | 0.7877 | 0.5230 | 0.5825 | 0.7902 |
| polish / original | LightGBM | 0.201069 | 0.8354 | 0.7584 | 0.9745 | 0.8295 | 0.6986 | 0.9786 |
| polish / cleaned | LightGBM | 0.201069 | 0.8354 | 0.7584 | 0.9745 | 0.8295 | 0.6986 | 0.9786 |

Frozen thresholds (rounded for display; the protocol JSON retains full precision):

| Model | Bank | Default | Polish |
|---|---:|---:|---:|
| Gradient boosting | 0.151737 | 0.255419 | 0.180186 |
| LightGBM | 0.191609 | 0.279692 | 0.201069 |
| CatBoost | 0.181900 | 0.218698 | 0.132656 |
| Stacking ensemble | 0.145609 | 0.262291 | 0.210562 |
| Neural net | 0.202698 | 0.246438 | 0.075753 |

Validation evidence for the original-arm winners:

| Dataset / model | Validation F1 at 0.5 | Validation F1 at chosen threshold |
|---|---:|---:|
| bank_marketing / Stacking ensemble | 0.3259 | 0.4790 |
| default / Gradient boosting | 0.4481 | 0.5409 |
| polish / LightGBM | 0.7441 | 0.7734 |

Changing the threshold increases recall at the expense of precision. Lower accuracy on
Bank and Default is consistent with detecting more positives; it does not contradict
the improvement in F1. PR-AUC remains unchanged by threshold selection.


## Why model rankings differ

- **Bank Marketing:** 3,173 of 27,126 training rows are positive (11.70%). Categories and
  previous-contact information interact, while many customers have overlapping numeric
  characteristics. The stack can combine a broad linear trend with forest and boosting
  partitions. Both arms select it on validation. Cleaning adds an explicit previous-contact
  indicator and semantic column handling; it is plausible that this helps some learners,
  but a single seed cannot establish the cause of a small PR-AUC difference.
- **Default:** 3,982 of 18,000 training rows are positive (22.12%), and there are no missing
  training cells. Repayment history, balances and payment amounts can support short,
  nonlinear decision rules. Shallow gradient boosting can capture these with a relatively
  constrained learner. Its validation lead over stacking is very small, so describe it as
  this run's winner, not a proven general advantage. Cleaning has little to repair; category
  ordering and tree split tie-breaking can explain small numerical arm differences.
- **Polish Bankruptcy:** 1,255 of 26,043 training rows are positive (4.82%). There are
  24,708 missing training cells; A37 is 43.29% missing. A27 is missing in 34.50% of bankrupt
  training rows versus 4.90% of other rows. Ratios have extreme tails (maximum absolute
  training value 11,903,000). LightGBM's histogram splits and best-first leaf growth can
  isolate localized, nonlinear patterns without relying on smooth global relationships.
  This is a plausible explanation for its strong ranking, not a causal attribution.
  Both arms median-impute and retain extreme ratios, so neither tests native missing-value
  routing or missingness indicators. The scaled MLP remains vulnerable to heavy tails,
  weak minority signal and its accuracy-based internal early stopping. Accuracy near 95%
  is therefore insufficient evidence of useful bankruptcy detection.

All three boosting methods partition feature space using thresholds and combine trees,
but differ in tree shape, split search and regularization. The neural net adjusts shared
weights through smooth nonlinear transformations; stacking learns how to combine several
learners' out-of-fold probabilities. Which approach works best depends on prevalence,
feature representation, missingness, interactions and sample size. PR-AUC itself is sensitive
to prevalence, so compare models **within each dataset**; do not interpret a larger PR-AUC
on one dataset as proof that its problem is intrinsically easier. These single-seed results
do not establish statistical significance; CV SD is variation across training folds.

## Reproduction and agent contract

Use Python 3.12 with `uv sync --frozen`, plus `brew install libomp` on macOS if needed.
Run the three README baseline commands sequentially. `--arm both` calibrates original
first, then reuses the protocol for cleaned. If a protocol already exists it is reused
for both arms. Keep raw inputs, splits, dependency versions and model settings unchanged.
For an intentional new protocol, use a separate experiment root; do not edit a frozen file.

```python
from agentic_gan.agents.training import run_training_arms

results = run_training_arms(
    training_sets, X_val, y_val, X_test, y_test,
    protocol_path="results/baseline_protocol_bank_marketing_42_cv3.json",
    seed=42, cv_folds=3,
)
```

`training_sets` maps each arm to `(X_train, y_train)`. Supply identical real holdouts and
matching feature order. For augmented arms, encode synthetic rows using the encoder fitted
on real training rows, add them only to training, and pass encoded inputs. The helper
loads all five models and thresholds from the checked protocol. It does not automatically
run the separate LangGraph training stub, which is still a placeholder.

The newly pulled `tools/augmented.py` uses the same protocol and complete suite. Its
default seed is now 42 because that is the seed frozen here; additional seeds require
compatible original reference protocols on the same holdouts. Its output defaults to
`results/augmented_bank_marketing_frozen.csv`. Existing augmented CSV/plot files are
historical 0.5-threshold experiments and were **not** rerun or relabeled by this task.
They must not be pooled with the frozen-protocol results.

IEEE-CIS preprocessing is deferred pending the settings/splits from issue #50 and local
source files, as agreed. No provisional IEEE settings or generated fraud outputs are
included in this change.

## References

- [scikit-learn: tuning decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html).
- [LightGBM: histogram splitting and leaf-wise growth](https://lightgbm.readthedocs.io/en/stable/Features.html).
- [CatBoost: how training is performed](https://catboost.ai/docs/en/concepts/algorithm-main-stages).
