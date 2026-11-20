"""Choose synthetic fill from validation alone, allowing the cleaned arm (0%).

python -m agentic_gan.tools.volume --dataset bank_marketing --run runs/<id>
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_gan.tools.evaluation import load_selected_scores, paired_bootstrap


@dataclass(frozen=True)
class VolumeDecision:
    arm: str
    fill_level: float
    lift: float
    lo: float
    hi: float
    reason: str
    evidence: list[dict]


def pick_fill(
    y_val, val_scores: dict[float, np.ndarray], *, arm: str, n_boot: int = 1000, seed: int = 0
) -> VolumeDecision:
    """Choose the largest positive lift whose paired 95% CI is entirely above zero.

    All scores must describe the same validation rows in the order of ``y_val``.
    Exact lift ties prefer the smaller fill; no eligible candidate selects cleaned.
    """
    if arm not in ("ctgan", "smotenc"):
        raise ValueError("arm must be 'ctgan' or 'smotenc'")
    if 0.0 not in val_scores:
        raise ValueError("val_scores must include key 0.0 for the cleaned reference")
    if any(not np.isfinite(level) or not 0 <= level <= 1 for level in val_scores):
        raise ValueError("fill levels must be finite and between 0 and 1")
    # Validate the reference even when there are no candidate levels.
    paired_bootstrap(y_val, val_scores[0.0], val_scores[0.0], n_boot=n_boot, seed=seed)
    evidence = []
    for level in sorted(val_scores):
        if level == 0:
            continue
        result = paired_bootstrap(
            y_val, val_scores[0.0], val_scores[level], metric="pr_auc", n_boot=n_boot, seed=seed
        )
        evidence.append(dict(fill_level=float(level), lift=result.diff, lo=result.lo, hi=result.hi))
    eligible = [row for row in evidence if row["lo"] > 0 and row["lift"] > 0]
    best = max(eligible, key=lambda row: (row["lift"], -row["fill_level"])) if eligible else None
    detail = "; ".join(
        f"{row['fill_level']:.1%}: lift {row['lift']:+.6f}, "
        f"95% CI [{row['lo']:+.6f}, {row['hi']:+.6f}]"
        for row in evidence
    )
    choice = f"Chose {best['fill_level']:.1%}" if best else "Chose 0% (cleaned)"
    reason = (
        f"{choice} for {arm}: {detail or 'no positive fill levels supplied'}; "
        "only validation lifts with a 95% CI lower bound above 0 are eligible."
    )
    values = best or dict(fill_level=0.0, lift=0.0, lo=0.0, hi=0.0)
    return VolumeDecision(arm=arm, reason=reason, evidence=evidence, **values)


def decisions_from_run(
    run: Path, dataset: str, *, seed: int = 42, n_boot: int = 1000, bootstrap_seed: int = 0
) -> list[VolumeDecision]:
    """Load selected validation predictions, checking row identities before pairing."""
    metrics = pd.read_csv(run / "models" / "metrics.csv")
    metrics = metrics[metrics["dataset"].eq(dataset) & metrics["seed"].eq(seed)]
    if metrics.empty:
        raise ValueError(f"no metrics for dataset {dataset!r}, seed {seed}")
    base = load_selected_scores(
        run / "models", metrics, arm="cleaned", fill_level=0.0, seed=seed, split="validation"
    )
    decisions = []
    for arm in ("ctgan", "smotenc"):
        levels = sorted(metrics.loc[metrics["arm"].eq(arm), "fill_level"].unique())
        if not levels:
            continue
        scores = {0.0: base["probability"].to_numpy()}
        for level in levels:
            other = load_selected_scores(
                run / "models", metrics, arm=arm, fill_level=level, seed=seed, split="validation"
            )
            if not base[["source_row", "y_true"]].equals(other[["source_row", "y_true"]]):
                raise ValueError(f"validation rows/labels do not align for {arm}, fill {level:g}")
            scores[float(level)] = other["probability"].to_numpy()
        decisions.append(
            pick_fill(
                base["y_true"].to_numpy(), scores, arm=arm, n_boot=n_boot, seed=bootstrap_seed
            )
        )
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42, help="model seed")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="also save decisions as JSON")
    args = parser.parse_args()
    decisions = decisions_from_run(
        args.run,
        args.dataset,
        seed=args.seed,
        n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = json.dumps([asdict(item) for item in decisions], indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n")


if __name__ == "__main__":
    main()
