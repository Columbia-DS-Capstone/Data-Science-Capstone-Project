"""Helpers shared by the synthetic-data generators and the training experiments.

Kept free of heavy imports: importing torch (CTGAN) before LightGBM crashes the process on
macOS, so code that only needs these helpers must not have to load ``tools/ctgan.py``.
"""

from collections.abc import Sequence

import pandas as pd

IS_SYNTHETIC = "is_synthetic"


def fill_counts(y: pd.Series, fractions: Sequence[float] = (0.10, 0.25, 0.50)) -> dict[float, int]:
    """Rows to generate for each fill level, as a share of the minority-to-majority gap."""
    counts = y.value_counts()
    if len(counts) != 2:
        raise ValueError("fill_counts expects a binary target")
    gap = int(counts.max() - counts.min())
    return {f: round(gap * f) for f in fractions}
