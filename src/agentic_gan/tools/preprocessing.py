"""Reproducible preprocessing tools, based on Raghav's UCI baseline.

Write cleaned splits; model tools encode and scale in memory using training rows.
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agentic_gan.utils.datasets import DATASETS, DatasetConfig, load_configs


def load_dataset(config: DatasetConfig, path: Path) -> pd.DataFrame:
    """Load cached data without fetching or silently changing source order."""
    if config.name == "polish":

        def unpack(blob, depth=0):
            if depth > 4:
                raise ValueError("Unexpected archive nesting")
            found = {}
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                for name in archive.namelist():
                    if name.endswith("/"):
                        continue
                    if name.lower().endswith(".zip"):
                        found.update(unpack(archive.read(name), depth + 1))
                    elif name.lower().endswith(".arff"):
                        key = Path(name).name.lower()
                        if key in found:
                            raise ValueError(f"Duplicate cohort: {key}")
                        found[key] = archive.read(name)
            return found

        files = unpack(path.read_bytes())
        frames = []
        for name in config.cohorts:
            records, _ = arff.loadarff(io.StringIO(files[name].decode("utf-8-sig")))
            frame = pd.DataFrame(records)
            frame.columns = [str(c).lower() for c in frame]
            frame = frame.rename(
                columns={"class": config.target, **{f"attr{i}": f"A{i}" for i in range(1, 65)}}
            )
            required = [f"A{i}" for i in range(1, 65)] + [config.target]
            frame = frame[required].apply(pd.to_numeric, errors="raise")
            frame.insert(0, "cohort", name.removesuffix(".arff"))
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
    else:
        data = pd.read_csv(path, sep=None, engine="python")
        if config.name == "default":
            data = data.drop(columns=["ID"], errors="ignore").rename(
                columns={"default payment next month": "y", "Y": "y", "PAY_0": "PAY_1"}
            )
    if not data.columns.is_unique:
        raise ValueError("Duplicate source columns")
    data.index.name = "source_row"
    return data


def normalize(
    data: pd.DataFrame, config: DatasetConfig, include_duration: bool = False
) -> pd.DataFrame:
    """Apply fixed semantic rules; never learn statistics here."""
    data = data.copy()
    if config.name == "bank_marketing":
        data = data.rename(columns={"day_of_week": "day_of_month", "day": "day_of_month"})
        if not data.columns.is_unique:
            raise ValueError("Ambiguous day column aliases")
        if not data["day_of_month"].between(1, 31).all():
            raise ValueError("Expected Bank day-of-month values 1..31")
        for c in ("job", "education", "contact", "poutcome"):
            data[c] = data[c].fillna("unknown")
        if not include_duration:
            data = data.drop(columns="duration", errors="ignore")
        data["previously_contacted"] = (data["pdays"] != -1).astype(int)
        if data["pdays"].isna().any():
            raise ValueError("Unexpected missing pdays; expected -1 for no previous contact")
        if config.target in data and not pd.api.types.is_numeric_dtype(data[config.target]):
            data[config.target] = data[config.target].map({"no": 0, "yes": 1})
    if config.target in data:
        if data[config.target].isna().any() or not data[config.target].isin([0, 1]).all():
            raise ValueError("Target must be observed and binary")
        data[config.target] = data[config.target].astype(int)
    return data.replace([np.inf, -np.inf], np.nan)


def split_dataset(data: pd.DataFrame, config: DatasetConfig) -> dict[str, list[int]]:
    """Recreate stratified splits from config; no separate row-ID files."""
    settings = config.split
    ratios = [settings[s] for s in ("train", "validation", "test")]
    if not all(np.isfinite(v) and 0 < v < 1 for v in ratios) or not np.isclose(
        sum(ratios), 1.0, rtol=0, atol=1e-9
    ):
        raise ValueError("Split ratios must be positive and sum to 1")
    holdout = settings["validation"] + settings["test"]
    train, hold = train_test_split(
        data.index,
        test_size=holdout,
        stratify=data[config.target],
        random_state=settings["seed"],
    )
    validation, test = train_test_split(
        hold,
        test_size=settings["test"] / holdout,
        stratify=data.loc[hold, config.target],
        random_state=settings["seed"],
    )
    return {
        name: ids.tolist()
        for name, ids in zip(
            ("train", "validation", "test"), (train, validation, test), strict=True
        )
    }


def profile_dataframe(data: pd.DataFrame) -> dict:
    """Profile only the supplied training rows."""
    return {
        "rows": len(data),
        "duplicate_rows": int(data.duplicated().sum()),
        "columns": {
            c: {
                "dtype": str(data[c].dtype),
                "missing": int(data[c].isna().sum()),
                "unique": int(data[c].nunique()),
            }
            for c in data
        },
    }


class Preprocessor:
    """Training-fitted cleaning and encoding, reusable after serialization."""

    def __init__(self, config: DatasetConfig, imputation="median", include_duration=False):
        if imputation not in ("median", "mean"):
            raise ValueError("imputation must be median or mean")
        self.config = config
        self.imputation = imputation
        self.include_duration = include_duration

    def fit(self, raw_train: pd.DataFrame):
        train = normalize(raw_train, self.config, self.include_duration)
        self.columns = train.columns.drop(self.config.target).tolist()
        self.discrete = [c for c in self.config.discrete if c in self.columns]
        self.numeric = [c for c in self.columns if c not in self.discrete]
        self.fill_values = {}
        for c in self.columns:
            values = train[c]
            if c in self.discrete:
                modes = values.mode()
                self.fill_values[c] = modes.iloc[0] if len(modes) else "__missing__"
            else:
                values = pd.to_numeric(values, errors="raise")
                value = values.median() if self.imputation == "median" else values.mean()
                self.fill_values[c] = 0.0 if pd.isna(value) else float(value)
        self.training_rows = len(train)
        self.encoder = ColumnTransformer(
            [
                ("numeric", StandardScaler(), self.numeric),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.discrete,
                ),
            ],
            remainder="drop",
        )
        self.encoder.fit(self.clean(raw_train)[self.columns])
        self.feature_names = self.encoder.get_feature_names_out().tolist()
        return self

    def _fill(self, data):
        missing = set(self.columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        out = data[self.columns].copy()
        for c in self.columns:
            if c in self.discrete:
                out[c] = out[c].fillna(self.fill_values[c])
                # Stable strings for numeric category codes, including synthetic float codes.
                out[c] = out[c].map(
                    lambda v: (
                        str(int(v))
                        if isinstance(v, (int, float, np.number)) and float(v).is_integer()
                        else str(v)
                    )
                )
            else:
                out[c] = (
                    pd.to_numeric(out[c], errors="raise")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(self.fill_values[c])
                )
        if self.config.target in data:
            out[self.config.target] = data[self.config.target]
        return out

    def clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self._fill(normalize(raw, self.config, self.include_duration))

    def transform_cleaned(self, cleaned: pd.DataFrame) -> pd.DataFrame:
        """Encode real or synthetic CLEANED rows, without refitting or scaling labels/IDs."""
        cleaned = self._fill(cleaned)
        result = pd.DataFrame(
            self.encoder.transform(cleaned[self.columns]),
            columns=self.feature_names,
            index=cleaned.index,
        )
        if self.config.target in cleaned:
            if not cleaned[self.config.target].isin([0, 1]).all():
                raise ValueError("Synthetic target must be binary and nonmissing")
            result[self.config.target] = cleaned[self.config.target].astype(int)
        return result

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.transform_cleaned(self.clean(raw))


def write_treatment_log(raw_train, processor, path):
    normalized = normalize(raw_train, processor.config, processor.include_duration)
    rows = []
    for c in dict.fromkeys([*raw_train.columns, *normalized.columns]):
        issue, action, reason = "No special issue", "Retain", "Preserve source information"
        new = "day_of_month" if c in ("day", "day_of_week") else c
        if new != c:
            issue, action, reason = "Mislabeled day", "Rename to day_of_month", "Values are 1..31"
        elif c == "duration" and not processor.include_duration:
            issue, action, reason = "Post-call leakage", "Drop", "Pre-call prediction"
        elif c == "duration":
            issue, action, reason = "Post-call feature", "Retain by switch", "Post-call diagnostic"
        elif c in ("job", "education", "contact", "poutcome"):
            issue, action, reason = (
                "Unknown category",
                "Preserve/fill unknown",
                "Informative category",
            )
        elif c == "pdays":
            issue, action, reason = (
                "-1 sentinel",
                "Keep -1; add previously_contacted",
                "No prior contact",
            )
        elif c == "previously_contacted":
            issue, action, reason = (
                "Derived feature",
                "1 if pdays != -1",
                "Separate contact history",
            )
        elif processor.config.name == "default" and c in processor.discrete:
            issue = "Category codes, including undocumented values"
            action = "Keep each code as a separate category"
            reason = "EDA flags EDUCATION 0/5/6, MARRIAGE 0, PAY -2/0; avoid invented semantics"
        if new in processor.fill_values:
            count = int(normalized[new].isna().sum())
            if count:
                issue += f"; {count} training gaps"
                action += f"; fill {processor.fill_values[new]!r}"
                method = processor.imputation if new in processor.numeric else "mode"
                reason += f"; training-only {method}"
                if normalized[new].isna().all():
                    reason += "; all-missing fallback"
            action += "; one-hot encode" if new in processor.discrete else "; standard scale"
        rows.append({"column": c, "issue": issue, "treatment": action, "reason": reason})
    pd.DataFrame(rows).to_csv(path, index=False)


def run_preprocessing(
    dataset,
    root=None,
    input_path=None,
    imputation=None,
    include_duration=None,
    config_path=None,
):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    config = load_configs(config_path)[dataset]
    imputation = config.imputation if imputation is None else imputation
    include_duration = config.include_duration if include_duration is None else include_duration
    source = Path(input_path) if input_path else root / config.file
    raw = load_dataset(config, source)
    normalized = normalize(raw, config, include_duration)
    splits = split_dataset(normalized, config)
    train = raw.loc[splits["train"]]
    processor = Preprocessor(config, imputation, include_duration).fit(train)
    output = root / "data" / "interim" / dataset
    output.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        cleaned = processor.clean(raw.loc[ids])
        cleaned.to_csv(output / f"{split}_cleaned.csv", index_label="source_row")
    # Remove only artifacts generated by older versions of this pipeline.
    legacy_files = [
        "preprocessor.pkl",
        "profile.json",
        "treatment_log.csv",
        "manifest.json",
        "default_code_audit.csv",
        "eda_default_code_audit.csv",
        *[f"{split}_{kind}.csv" for split in splits for kind in ("encoded", "missing_mask")],
    ]
    for name in legacy_files:
        (output / name).unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--input", dest="input_path", type=Path)
    parser.add_argument("--imputation", choices=["mean", "median"])
    parser.add_argument("--include-duration", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--config", dest="config_path", type=Path)
    args = vars(parser.parse_args())
    from agentic_gan.tools.preprocessing import run_preprocessing as run

    print(run(**args))


if __name__ == "__main__":
    main()
