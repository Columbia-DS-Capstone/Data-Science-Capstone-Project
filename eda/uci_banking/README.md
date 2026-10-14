# UCI Bank Marketing modeling

This directory compares binary classifiers for predicting whether a customer subscribed to a term deposit using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing). Both notebooks include the recorded call duration and classify outcomes of completed calls.

- `uci_baseline.ipynb` compares eight classifiers with stratified cross-validation, selects by validation PR-AUC, and evaluates the selected model on a held-out test split.
- `uci_smotenc.ipynb` repeats the comparison with categorical-aware SMOTENC oversampling of the training class. Set `N_SYNTHETIC` in its first code cell to choose the number of synthetic subscriber rows (currently 5,000). Validation and test data remain at their original class balance.
- `scripts/uci_preprocessing.py` fetches, splits, imputes, encodes, and scales the data. `scripts/uci_preprocessing_smotenc.py` adds the configurable training-only SMOTENC step.

Both notebooks use the same default 60/20/20 train/validation/test split and show metric tables, graphs, and short interpretations. Run them from this directory using the `bead` Python environment with `ucimlrepo`, scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, pandas, NumPy, and Matplotlib installed. The UCI fetch requires internet access.

## Preprocessing decisions

| Step | What we do and why |
| --- | --- |
| Target and column names | Map `y` from `no`/`yes` to 0/1 for binary classification, and rename the API's `day_of_week` field to `day` for clarity. |
| Unknown categories | Fill missing values in `job`, `education`, `contact`, and `poutcome` with `unknown`. These represent source “unknown” categories; retaining them avoids inventing customer attributes and lets the model use that information. |
| Call duration | Keep `duration` because the objective is to classify subscription outcomes for completed calls. Full duration is unavailable before a call ends, so these results cannot be used to decide whom to call beforehand. |
| Previous contact | Turn `pdays = -1` into a `previously_contacted` flag plus a missing `pdays` value. The sentinel means no previous contact, not “minus one days ago.” |
| Split before fitting | Make a stratified train/validation/test split (60/20/20 by default) so each set has a similar subscriber rate. Fit learned preprocessing on training rows only to avoid leaking information from validation or test rows. |
| Numeric features | Fill missing numeric values with the training median and standardize them. The median handles the missing `pdays` values; scaling keeps numeric magnitudes comparable for logistic regression and neighbor-based sampling. |
| Categorical features | One-hot encode categories, ignoring unseen categories at validation or test time, so the classifiers can accept the fields without assuming an artificial category order. |
| SMOTENC experiment | Generate the requested number of subscriber rows from training data only. SMOTENC handles numeric and categorical features together before final one-hot encoding; each cross-validation training fold fits its own sampler. Validation and test data keep their original class balance. |

## Evaluation and implications

Both notebooks use a stratified **60/20/20 train/validation/test split** and three-fold cross-validation on training data. The selected model is the one with the highest validation PR-AUC; it is then evaluated once on the untouched test split. Displayed metric values are rounded to three decimals.

Each notebook plots validation PR-AUC, precision, recall, F1, and accuracy across models. The accuracy chart marks the always-predict-no baseline. Test plots show the selected model's precision–recall curve and confusion matrix, with counts and within-class percentages.

The SMOTENC notebook also compares the same stacking model before and after synthetic sampling: training class counts, all five test metrics, and side-by-side test confusion matrices.

| Experiment | Selected model | F1 | Accuracy | Precision | Recall | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | Stacking | 0.555 | 0.911 | 0.668 | 0.474 | 0.650 |
| SMOTENC | Stacking | 0.552 | 0.908 | 0.637 | 0.487 | 0.636 |

At the 0.5 cutoff, SMOTENC found **515** subscribers versus **502** for the baseline, but also flagged **293** non-subscribers versus **249**. Synthetic rows improved recall slightly while reducing precision and PR-AUC. Because full duration is available only after a call, these models are useful for analysis of completed interactions or review workflows; the reported metrics do not by themselves establish campaign profit or the effect of contacting a customer.

A conditional tabular GAN is a possible next synthetic-data experiment. Unlike SMOTENC's neighbor-based generation, it may learn richer relationships among numeric and categorical features and generate more varied subscriber examples. It may also reproduce noise or overfit; any benefit must be demonstrated against the baseline on the same untouched validation and test rows. [CTGAN research](https://papers.neurips.cc/paper_files/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html)

## Metrics

The positive class is a customer who subscribes (`y = yes`). Precision, recall, F1, and accuracy use a predicted-probability threshold of 0.5. Higher values are better for every metric below.

| Metric | Meaning |
| --- | --- |
| PR-AUC | Area under the precision–recall curve across probability cutoffs, calculated by trapezoidal integration. It is the model-selection metric; the positive-class rate is about 0.117 here. |
| Precision | Of customers predicted to subscribe, the fraction who actually subscribe: `TP / (TP + FP)`. |
| Recall | Of actual subscribers, the fraction correctly identified: `TP / (TP + FN)`. |
| F1 | Harmonic mean of precision and recall: `2 × precision × recall / (precision + recall)`. |
| Accuracy | Fraction of all predictions that are correct. It can look high even when many subscribers are missed. |

`TP`, `FP`, and `FN` mean true positives, false positives, and false negatives. Cross-validation tables show the mean and standard deviation (`SD`) across three stratified folds. Validation metrics select the model; test metrics report its final held-out performance.

