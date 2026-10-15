# SMOTENC investigation

[The notebook](smote_investigation.ipynb) compares baseline and SMOTENC models on the same UCI Bank Marketing train/validation/test split.

## Why is the accuracy dropping?

SMOTENC raises training subscriber odds from `3,173 / 23,953 = 0.1325` to `8,173 / 23,953 = 0.3412` (2.576 times higher). On the 9,043 test customers, the baseline stack has **502 true positives and 250 false positives**; SMOTENC has **515 and 293**. Thus:

$$
\Delta\text{accuracy} = \frac{\Delta TP-\Delta FP}{N}
= \frac{13-43}{9043} = -0.0033.
$$

- Accuracy falls from **0.9109 to 0.9076** because 43 extra false positives outweigh 13 recovered subscribers.
- Recall rises (0.4745 → 0.4868),
- precision falls (0.6676 → 0.6374),
- and F1 changes little (0.5547 → 0.5520).

The decision boundary plot supports the following direction:
- The averaged **predict yes** area grows from **32.9% to 34.4%** of the displayed age × duration rectangle.
- The fraction of *actual test customers* predicted yes grows from **8.3% to 8.9%**.
- Plot area depends on the chosen range and 30 reference profiles; it is not a customer share or each customer's full-feature decision boundary.

The **Brier score**, `mean((predicted probability − outcome)²)`, rises from **0.0626 to 0.0638** (worse probability error). It supports the diagnosis, while TP/FP counts directly explain accuracy. PR-AUC, computed across cutoffs, falls from **0.6501 to 0.6360**; probability ranking changed too. One split cannot isolate SMOTENC's causal mechanism.

## Comparison for a future GAN

The notebook also compares class weights, ADASYN, BorderlineSMOTE, undersampling, and a validation-tuned cutoff. The validation-selected ranking benchmark is the **baseline stack (PR-AUC 0.6291)**; tuning its cutoff to **0.1706** gives validation **F1 0.6350**. Compare any GAN against the strongest relevant non-GAN control with the same downstream model, select on validation, then evaluate on a **fresh holdout**. The current test results have already been inspected and should not be used to tune the GAN.
