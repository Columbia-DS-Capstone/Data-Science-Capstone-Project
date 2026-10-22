import pandas as pd
import pytest

from agentic_gan.tools.augmented import check_holdouts, load_cleaned, run_experiment, summarize
from agentic_gan.utils.datasets import DATASETS
from agentic_gan.utils.synthetic import fill_counts

CONFIG = DATASETS["bank_marketing"]


@pytest.fixture(scope="module")
def tables():
    full = load_cleaned(CONFIG)
    return {
        "train": full["train"].groupby("y", group_keys=False).sample(frac=0.15, random_state=0),
        "validation": full["validation"].sample(1500, random_state=0),
        "test": full["test"].sample(1500, random_state=0),
    }


def test_holdout_check_rejects_synthetic_flag_and_overlap(tables):
    check_holdouts(tables, CONFIG)
    flagged = {**tables, "test": tables["test"].assign(is_synthetic=1)}
    with pytest.raises(ValueError, match="is_synthetic"):
        check_holdouts(flagged, CONFIG)
    leaked = {**tables, "validation": pd.concat([tables["validation"], tables["train"].head(5)])}
    with pytest.raises(ValueError, match="share source rows"):
        check_holdouts(leaked, CONFIG)


def test_smotenc_experiment_end_to_end(tables):
    out = run_experiment(
        tables, CONFIG, "smotenc", "smotenc", seeds=(1, 2), fill_levels=(0.10,),
        model_names=("Logistic regression",),
    )
    assert set(out.arm) == {"real_only", "smotenc"}
    assert (out.status == "success").all()
    assert out.groupby(["arm", "seed"]).selected.sum().eq(1).all()
    expected = fill_counts(tables["train"]["y"], (0.10,))[0.10]
    assert list(out[out.arm == "smotenc"].n_synthetic.unique()) == [expected]
    summary = summarize(out)
    assert {"val_pr_auc_mean", "test_f1_std"} <= set(summary.columns)
