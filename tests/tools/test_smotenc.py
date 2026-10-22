import numpy as np
import pandas as pd
import pytest

from agentic_gan.tools.smotenc import smotenc_rows


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.15).astype(int)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.normal(50, 10, n) + 15 * y,
        "job": rng.choice(["a", "b", "c"], n),
        "flag": rng.integers(0, 2, n),
        "y": y,
    })


def test_returns_n_flagged_minority_rows(df):
    out = smotenc_rows(df, "y", 60, discrete_columns=["job", "flag"], seed=1)
    assert len(out) == 60
    assert (out["y"] == 1).all() and (out["is_synthetic"] == 1).all()
    assert list(out.columns) == [*df.columns, "is_synthetic"]


def test_values_stay_valid(df):
    out = smotenc_rows(df, "y", 60, discrete_columns=["job", "flag"], seed=1)
    assert set(out["job"]) <= set(df["job"])
    assert set(out["flag"]) <= {0, 1}
    assert out["age"].dtype == df["age"].dtype and out["flag"].dtype == df["flag"].dtype
    assert out["age"].between(df["age"].min(), df["age"].max()).all()
    assert out["income"].between(df["income"].min(), df["income"].max()).all()


def test_does_not_modify_input_and_is_reproducible(df):
    before = df.copy()
    first = smotenc_rows(df, "y", 30, discrete_columns=["job", "flag"], seed=7)
    second = smotenc_rows(df, "y", 30, discrete_columns=["job", "flag"], seed=7)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(first, second)


def test_rejects_bad_input(df):
    with pytest.raises(ValueError, match="positive integer"):
        smotenc_rows(df, "y", 0)
    bad = df.copy()
    bad.loc[0, "income"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        smotenc_rows(bad, "y", 10, discrete_columns=["job", "flag"])
