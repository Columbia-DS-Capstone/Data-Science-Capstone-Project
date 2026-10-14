"""Dataset-independent binary baselines, using Raghav's model settings and PR-AUC."""

import argparse
import hashlib
import json
import platform
import time
import warnings
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from agentic_gan.tools.preprocessing import Preprocessor, load_dataset, normalize, split_dataset
from agentic_gan.utils.datasets import load_configs

METRICS = ("pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy")


RESULT_COLUMNS = [
    "dataset",
    "arm",
    "model",
    "seed",
    "cv_folds",
    "status",
    "selected",
    "cv_pr_auc_mean",
    "cv_pr_auc_sd",
    "val_pr_auc",
    "test_pr_auc",
    "test_roc_auc",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_accuracy",
]


def build_models(seed=42):
    """Original eight unchanged; ninth model is an MLP with training-only early stopping."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    models = {
        "Logistic regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Decision tree": DecisionTreeClassifier(random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=seed),
        "Gradient boosting": GradientBoostingClassifier(random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=150, n_jobs=2, random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=150, verbosity=-1, n_jobs=2, random_state=seed),
        "CatBoost": CatBoostClassifier(
            iterations=200, thread_count=2, verbose=0, random_seed=seed, allow_writing_files=False
        ),
    }
    models["Stacking ensemble"] = StackingClassifier(
        estimators=[
            ("logistic", clone(models["Logistic regression"])),
            ("forest", clone(models["Random forest"])),
            ("lightgbm", clone(models["LightGBM"])),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        stack_method="predict_proba",
        cv=3,
        n_jobs=1,
    )
    models["Neural net"] = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
    )
    return models


def pr_auc_score(y, probabilities):
    """Trapezoidal PR-AUC, matching the notebook (not average precision)."""
    precision, recall, _ = precision_recall_curve(y, probabilities)
    return float(auc(recall, precision))


def metric_row(y, probabilities):
    probabilities = np.asarray(probabilities)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predicted probabilities")
    prediction = (probabilities >= 0.5).astype(int)
    return {
        "pr_auc": pr_auc_score(y, probabilities),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def _pr_scorer(estimator, X, y):
    return pr_auc_score(y, estimator.predict_proba(X)[:, 1])


def _check_inputs(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, cv_folds):
    if cv_folds != 0 and cv_folds < 2:
        raise ValueError("cv_folds must be zero (disabled) or at least two")
    if len({X.shape[1] for X in (X_train, X_val, X_test)}) != 1:
        raise ValueError("Feature counts differ across splits")
    frames = [X for X in (X_train, X_val, X_test) if isinstance(X, pd.DataFrame)]
    reference_columns = list(frames[0].columns) if frames else None
    for X, y in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        if len(X) != len(y) or not len(X):
            raise ValueError("Each feature table must have matching nonempty labels")
        if np.asarray(y).ndim != 1 or set(np.unique(y)) != {0, 1}:
            raise ValueError("Each split must contain both binary labels 0 and 1")
        if isinstance(X, pd.DataFrame):
            if not X.columns.is_unique or list(X.columns) != reference_columns:
                raise ValueError("Feature columns/order differ across splits")
            if "source_row" in X.columns:
                raise ValueError("Remove source_row from model features")
            if isinstance(y, pd.Series) and not X.index.equals(y.index):
                raise ValueError("Feature and label indices are misaligned")
        if preprocessor is None and not np.isfinite(np.asarray(X, dtype=float)).all():
            raise ValueError("Encoded inputs must be finite; supply a preprocessor for raw data")
    if np.bincount(np.asarray(y_train, dtype=int)).min() < max(3, cv_folds):
        raise ValueError("Too few training examples per class for stacking/CV")


def run_models(
    X_train, y_train, X_val, y_val, X_test, y_test, seed=42, *, preprocessor=None, cv_folds=3
) -> pd.DataFrame:
    """Return one row per model; select on validation PR-AUC, test only the winner.

    Six positional inputs may be encoded arrays/DataFrames. For raw data, pass
    an unfitted sklearn-compatible preprocessor, cloned inside each CV fold.
    Synthetic rows belong in X_train/y_train only. Use identical seed, feature
    schema, preprocessing, CV settings and real holdouts for every arm.
    With already-encoded inputs, CV describes training-fitted preprocessing;
    pass raw data + preprocessor for outer-fold-local preprocessing.
    Stacking shares that preprocessing across its inner folds, as in the notebook.
    Test metric cells are empty for models not selected. Failures and convergence
    warnings are recorded; failed models cannot win. No estimator is fitted on test.
    """
    _check_inputs(X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, cv_folds)
    models = build_models(seed)
    rows, fitted = [], {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed) if cv_folds else None
    scoring = {
        "pr_auc": _pr_scorer,
        "roc_auc": "roc_auc",
        "f1": "f1",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
    }
    with threadpool_limits(limits=2):
        for name, estimator in models.items():
            started = time.perf_counter()
            row = {
                "model": name,
                "seed": seed,
                "cv_folds": cv_folds,
                "status": "success",
                "selected": False,
                "error": "",
                **{f"test_{m}": np.nan for m in METRICS},
            }
            model = (
                make_pipeline(clone(preprocessor), estimator)
                if preprocessor is not None
                else estimator
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    if cv is not None:
                        scores = cross_validate(
                            model,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring=scoring,
                            n_jobs=1,
                            error_score="raise",
                        )
                        for metric in METRICS:
                            row[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
                            row[f"cv_{metric}_sd"] = float(scores[f"test_{metric}"].std(ddof=1))
                    model.fit(X_train, y_train)
                    row.update(
                        {
                            f"val_{k}": v
                            for k, v in metric_row(y_val, model.predict_proba(X_val)[:, 1]).items()
                        }
                    )
                    fitted[name] = model
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                row["warnings"] = " | ".join(sorted({str(w.message) for w in caught}))
            row["fit_seconds"] = time.perf_counter() - started
            rows.append(row)
            print(
                f"{name}: {row['status']}; "
                f"validation PR-AUC={row.get('val_pr_auc', float('nan')):.6f}",
                flush=True,
            )
        successful = [r for r in rows if r["status"] == "success"]
        if not successful:
            raise RuntimeError("All models failed: " + json.dumps(rows, default=str))
        winner = max(successful, key=lambda r: r["val_pr_auc"])
        winner["selected"] = True
        winner.update(
            {
                f"test_{k}": v
                for k, v in metric_row(
                    y_test, fitted[winner["model"]].predict_proba(X_test)[:, 1]
                ).items()
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("val_pr_auc", ascending=False, kind="stable", na_position="last")
        .reset_index(drop=True)
    )


class SharedPreprocessor(TransformerMixin, BaseEstimator):
    """Make the shared cleaner cloneable, so each training CV fold learns its own fills."""

    def __init__(self, config):
        self.config = config

    def fit(self, X, y):
        train = X.copy()
        train[self.config.target] = np.asarray(y)
        self.fitted_ = Preprocessor(
            self.config, self.config.imputation, self.config.include_duration
        ).fit(train)
        return self

    def transform(self, X):
        return self.fitted_.transform(X).to_numpy()


def _minimal_transformer(X, discrete, strategy="median"):
    categorical = [c for c in X if c in discrete or not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X if c not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy=strategy, keep_empty_features=True), StandardScaler()
                ),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ]
    )


def prepare_inputs(dataset, arm, root, seed, config_path=None):
    """Load raw sources and return folds plus unfitted train-only preprocessing."""
    if arm not in ("original", "cleaned"):
        raise ValueError(f"Unknown arm: {arm}; choose original or cleaned")
    config = load_configs(config_path)[dataset]
    config = replace(config, split={**config.split, "seed": seed})
    raw = load_dataset(config, root / config.file)
    normalized = normalize(raw, config, config.include_duration)
    data = raw.copy()
    data[config.target] = normalized[config.target]
    if arm == "cleaned":
        transformer = SharedPreprocessor(config)
    elif arm == "original":
        # Minimum required to fit all estimators. No derived contact flag or recoding.
        if dataset == "bank_marketing" and not config.include_duration:
            data = data.drop(columns="duration")
        transformer = _minimal_transformer(
            data.drop(columns=config.target), config.discrete, config.imputation
        )
    else:
        raise ValueError(f"Unknown arm: {arm}")
    ids = split_dataset(data, config)
    frames = [data.loc[ids[s]] for s in ("train", "validation", "test")]
    inputs = tuple(
        item
        for frame in frames
        for item in (frame.drop(columns=config.target), frame[config.target])
    )
    metadata = {
        "source_sha256": hashlib.sha256((root / config.file).read_bytes()).hexdigest(),
        "config": asdict(config),
        "include_duration": config.include_duration,
        "split_hashes": {
            s: hashlib.sha256(json.dumps(v).encode()).hexdigest() for s, v in ids.items()
        },
        "versions": {
            p: version(p)
            for p in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm", "catboost")
        },
        "python": platform.python_version(),
    }
    return inputs, transformer, metadata


def save_results(table, path):
    """Upsert by dataset/arm/seed/CV, accepting any arm name and retaining other runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "arm", "seed", "cv_folds"]
    if path.exists():
        old = pd.read_csv(path)
        incoming = set(map(tuple, table[keys].drop_duplicates().to_numpy()))
        keep = [tuple(row) not in incoming for row in old[keys].to_numpy()]
        table = pd.concat([old.loc[keep], table], ignore_index=True)
    temporary = path.with_suffix(".csv.tmp")
    table.reindex(columns=RESULT_COLUMNS).to_csv(temporary, index=False)
    temporary.replace(path)


def run_dataset(dataset, *, root=None, arm="cleaned", seed=42, cv_folds=3, config_path=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    inputs, transformer, metadata = prepare_inputs(dataset, arm, root, seed, config_path)
    table = run_models(*inputs, seed=seed, preprocessor=transformer, cv_folds=cv_folds)
    for key, value in [
        ("dataset", dataset),
        ("arm", arm),
        ("threshold", 0.5),
        ("pr_auc_method", "trapezoidal"),
        ("include_duration", metadata["include_duration"]),
        ("source_sha256", metadata["source_sha256"]),
    ]:
        table[key] = value
    for split, y in zip(("train", "validation", "test"), inputs[1::2], strict=True):
        table[f"{split}_rows"] = len(y)
        table[f"{split}_positive_rate"] = float(np.mean(y))
    result_path = root / "results" / "baselines.csv"
    save_results(table, result_path)
    metadata.update(
        dataset=dataset,
        arm=arm,
        seed=seed,
        cv_folds=cv_folds,
        results=json.loads(table.to_json(orient="records")),
        models={name: model.get_params(deep=False) for name, model in build_models(seed).items()},
        selection="highest validation trapezoidal PR-AUC; threshold 0.5; test winner only",
    )
    (result_path.parent / f"{dataset}_{arm}_{seed}_cv{cv_folds}.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    print(
        table.loc[table.selected, ["model", *[f"test_{m}" for m in METRICS]]].to_string(
            index=False
        ),
        flush=True,
    )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Key in shared datasets.json")
    parser.add_argument("--arm", choices=["original", "cleaned", "both"], default="cleaned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", dest="config_path", type=Path)
    args = vars(parser.parse_args())
    arms = ["original", "cleaned"] if args["arm"] == "both" else [args["arm"]]
    for arm in arms:
        table = run_dataset(**{**args, "arm": arm})
        if (table.status != "success").any():
            raise SystemExit(
                "Some models failed; see errors in the run JSON beside results/baselines.csv"
            )


if __name__ == "__main__":
    main()
