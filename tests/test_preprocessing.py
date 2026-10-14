import json
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from agentic_gan.tools.preprocessing import Preprocessor, run_preprocessing, split_dataset
from agentic_gan.utils.datasets import DATASETS


def bank():
    return pd.DataFrame(
        {
            "day_of_week": [1, 2, 31, 3],
            "job": ["a", None, "b", "new"],
            "education": ["unknown"] * 4,
            "contact": [None] * 4,
            "poutcome": ["unknown"] * 4,
            "pdays": [-1, 5, -1, 10],
            "balance": [10.0, 20.0, np.nan, 10000.0],
            "duration": [1] * 4,
            "y": ["no", "yes", "no", "yes"],
        }
    )


def test_train_only_and_reload():
    data = bank()
    processor = Preprocessor(DATASETS["bank_marketing"]).fit(data.iloc[:2])
    cleaned = processor.clean(data)
    assert cleaned.loc[2, "balance"] == 15
    assert cleaned.loc[0, "pdays"] == -1
    assert "duration" not in cleaned and "day_of_month" in cleaned
    assert cleaned["previously_contacted"].tolist() == ["0", "1", "0", "1"]
    assert cleaned.loc[1, "job"] == "unknown"
    assert not cleaned.isna().any().any()
    assert "new" not in processor.encoder.named_transformers_["categorical"].categories_[0]
    reloaded = pickle.loads(pickle.dumps(processor))
    pd.testing.assert_frame_equal(processor.transform(data), reloaded.transform_cleaned(cleaned))
    assert processor.encoder.named_transformers_["numeric"].mean_[2] == 15
    assert "duration" in Preprocessor(DATASETS["bank_marketing"], include_duration=True).fit(
        data.iloc[:2]
    ).clean(data)


def test_default_codes_and_polish_imputation():
    data = pd.DataFrame(
        {
            "SEX": [1, 2, 1],
            "EDUCATION": [0, 5, 6],
            "MARRIAGE": [0, 1, 2],
            "PAY_1": [-2, 0, -1],
            "LIMIT_BAL": [10, 20, 30],
            "y": [0, 1, 0],
        }
    )
    p = Preprocessor(DATASETS["default"]).fit(data)
    assert p.clean(data)["EDUCATION"].tolist() == ["0", "5", "6"]
    assert "PAY_1" in p.discrete and "SEX" not in p.numeric
    data = pd.DataFrame(
        {
            "A1": [1.0, 2.0, 9.0, np.nan],
            "A2": [np.nan] * 4,
            "cohort": ["1year"] * 4,
            "bankrupt": [0, 1, 0, 1],
        }
    )
    for method, expected in [("mean", 4), ("median", 2)]:
        p = Preprocessor(DATASETS["polish"], method).fit(data.iloc[:3])
        assert p.clean(data).loc[3, "A1"] == expected
        assert p.clean(data)["A2"].eq(0).all()
        assert np.isfinite(p.transform(data).to_numpy()).all()


def test_split_validation_and_reproducibility():
    data = pd.DataFrame({"bankrupt": [0, 1] * 50})
    config = DATASETS["polish"]
    first = split_dataset(data, config)
    assert {k: len(v) for k, v in first.items()} == {"train": 60, "validation": 20, "test": 20}
    assert first == split_dataset(data, config)
    ids = sum(first.values(), [])
    assert len(ids) == len(set(ids)) == len(data)
    assert set(ids) == set(data.index)
    changed = replace(config, split={"train": 0.5, "validation": 0.25, "test": 0.25, "seed": 7})
    assert len(split_dataset(data, changed)["train"]) == 50
    with pytest.raises(ValueError, match="sum to 1"):
        split_dataset(data, replace(config, split={**config.split, "train": 0.8}))


def test_full_output_contract(tmp_path):
    data = pd.concat([bank()] * 20, ignore_index=True)
    source = tmp_path / "bank.csv"
    data.to_csv(source, index=False)
    out = run_preprocessing("bank_marketing", tmp_path, source)
    with (out / "preprocessor.pkl").open("rb") as stream:
        processor = pickle.load(stream)
    clean = pd.read_csv(out / "test_cleaned.csv", index_col="source_row")
    encoded = pd.read_csv(out / "test_encoded.csv", index_col="source_row")
    pd.testing.assert_frame_equal(
        processor.transform_cleaned(clean), encoded, check_dtype=False, atol=1e-10
    )
    assert "source_row" not in processor.feature_names
    log = pd.read_csv(out / "treatment_log.csv")
    assert set(data.columns) <= set(log.column)


def test_custom_config(tmp_path):
    from dataclasses import asdict

    data = pd.concat([bank()] * 20, ignore_index=True)
    data.to_csv(tmp_path / "input.csv", index=False)
    config = asdict(DATASETS["bank_marketing"])
    config.pop("name")
    config.update(
        file="input.csv",
        imputation="mean",
        include_duration=True,
        split={"train": 0.5, "validation": 0.25, "test": 0.25, "seed": 7},
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"bank_marketing": config}))
    out = run_preprocessing("bank_marketing", root=tmp_path, config_path=path)
    assert len(pd.read_csv(out / "train_cleaned.csv")) == 40
    assert "duration" in pd.read_csv(out / "train_cleaned.csv")
    assert not (tmp_path / "splits").exists()
    assert json.loads((out / "manifest.json").read_text())["imputation"] == "mean"
