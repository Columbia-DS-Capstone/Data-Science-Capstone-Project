"""Dataset-agnostic SMOTENC tool: oversample the minority class of a cleaned training table.

The simple-oversampler control for the GAN comparison. It takes the same cleaned (not encoded
or scaled) training split as ``tools/ctgan.py`` and returns synthetic rows in the same format,
so both generators go through the same encoding and training path. No dataset rules are
enforced, unlike the CTGAN rows: interpolation can produce rows a real record would not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler

IS_SYNTHETIC = "is_synthetic"


def smotenc_rows(
    train_df: pd.DataFrame,
    target_col: str,
    n: int,
    discrete_columns: Sequence[str] | None = None,
    positive_value: Any = 1,
    k_neighbors: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Return exactly ``n`` synthetic minority rows (``target_col == positive_value``).

    ``discrete_columns=None`` treats every non-numeric column as categorical; the target is
    always excluded from the features. Numeric columns are scaled for the neighbour search
    and mapped back afterwards, then clipped to the training range, and whole-number columns
    are rounded, as in ``tools/ctgan.py``. Rows carry ``is_synthetic = 1``.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if target_col not in train_df.columns:
        raise ValueError(f"target_col {target_col!r} not in train_df")
    if IS_SYNTHETIC in train_df.columns:
        raise ValueError(f"train_df must not already contain {IS_SYNTHETIC!r}")
    if train_df.isna().any().any():
        raise ValueError("SMOTENC needs a table without missing values")

    features = train_df.drop(columns=target_col)
    target = train_df[target_col]
    if discrete_columns is None:
        categorical = [c for c in features if not pd.api.types.is_numeric_dtype(features[c])]
    else:
        categorical = [c for c in features if c in set(discrete_columns)]
    numeric = [c for c in features if c not in categorical]
    n_positive = int((target == positive_value).sum())
    if n_positive <= k_neighbors:
        raise ValueError("too few minority rows for the requested k_neighbors")

    scaler = StandardScaler().fit(features[numeric])
    scaled = features.copy()
    scaled[numeric] = scaler.transform(features[numeric])

    sampler = SMOTENC(
        categorical_features=[features.columns.get_loc(c) for c in categorical],
        sampling_strategy={positive_value: n_positive + n},
        k_neighbors=k_neighbors,
        random_state=seed,
    )
    resampled, _ = sampler.fit_resample(scaled, target)
    synthetic = pd.DataFrame(resampled).iloc[len(features):].reset_index(drop=True)
    synthetic.columns = features.columns

    synthetic[numeric] = scaler.inverse_transform(synthetic[numeric].astype(float))
    for col in numeric:
        low, high = float(features[col].min()), float(features[col].max())
        synthetic[col] = synthetic[col].clip(low, high)
        if bool((features[col] == features[col].round()).all()):
            synthetic[col] = synthetic[col].round()
    for col, dtype in features.dtypes.items():
        synthetic[col] = synthetic[col].astype(dtype)

    synthetic[target_col] = positive_value
    synthetic = synthetic[list(train_df.columns)]
    synthetic[IS_SYNTHETIC] = 1
    return synthetic
