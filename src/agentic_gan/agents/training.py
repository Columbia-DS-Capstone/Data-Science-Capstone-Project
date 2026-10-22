"""Model Training Agent tools; callers supply real holdouts and each arm's training rows."""

import pandas as pd

from agentic_gan.tools.baselines import load_protocol, run_models

__all__ = ["run_models", "run_training_arms"]


def run_training_arms(
    training_sets,
    X_val,
    y_val,
    X_test,
    y_test,
    seed=42,
    *,
    protocol_path,
    preprocessor=None,
    cv_folds=3,
):
    """Run each {arm: (X_train, y_train)} with identical holdouts and settings.

    Accepts original, cleaned, GAN-augmented, SMOTE, or other caller-named arms.
    Synthetic rows must already be generated and belong only in training_sets.
    For encoded inputs, encode synthetic rows with the real-training-fitted setup.
    protocol_path is the frozen original-arm protocol for this dataset and seed.
    Every arm uses its full model suite and identical numerical thresholds.
    """
    if not training_sets:
        raise ValueError("Supply at least one training arm")
    protocol = load_protocol(
        protocol_path, seed=seed, cv_folds=cv_folds, y_val=y_val, y_test=y_test
    )
    tables = []
    for arm, (X_train, y_train) in training_sets.items():
        table = run_models(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            seed=seed,
            preprocessor=preprocessor,
            cv_folds=cv_folds,
            thresholds=protocol["thresholds"],
        )
        table.insert(0, "arm", arm)
        table["protocol_id"] = protocol["protocol_id"]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
