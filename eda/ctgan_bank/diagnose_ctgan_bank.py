"""Sanity checks on the Bank CTGAN run: loss curves, real vs synthetic columns, duplicates.

Reads the files written by run_ctgan_bank.py and compares the synthetic rows against the
real *subscribers* (y=1) in the training split, since only subscribers are synthesised.
Writes a text report and PNG plots to eda/ctgan_bank/artifacts/.

Run from the repo root:
    uv run python eda/ctgan_bank/diagnose_ctgan_bank.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eda" / "baseline-modeling" / "uci_banking" / "scripts"))

from uci_preprocessing import load_and_preprocess_uci  # noqa: E402

SYNTHETIC_DIR = ROOT / "data" / "interim" / "bank_marketing" / "synthetic"
ARTIFACT_DIR = ROOT / "eda" / "ctgan_bank" / "artifacts"

REAL_COLOR, SYNTH_COLOR = "#0072B2", "#E69F00"

report: list[str] = []


def say(text: str = "") -> None:
    print(text)
    report.append(text)


def style(ax) -> None:
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_losses(losses: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses["Epoch"], losses["Generator Loss"], color=REAL_COLOR, lw=1.5,
            label="Generator loss")
    ax.plot(losses["Epoch"], losses["Discriminator Loss"], color=SYNTH_COLOR, lw=1.5,
            label="Discriminator loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CTGAN training losses (Bank, 300 epochs)")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_numeric(real: pd.DataFrame, synth: pd.DataFrame, cols: list[str]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.flat, cols, strict=False):
        r, s = real[col].dropna(), synth[col].dropna()
        bins = np.histogram_bin_edges(r, bins=30)
        ax.hist(r, bins=bins, density=True, histtype="step", lw=1.5, color=REAL_COLOR,
                label="Real subscribers")
        ax.hist(s, bins=bins, density=True, histtype="step", lw=1.5, color=SYNTH_COLOR,
                label="Synthetic")
        ax.set_title(col, fontsize=10)
        style(ax)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", frameon=False)
    fig.suptitle("Numeric columns: real subscribers vs synthetic (density; NaN dropped)")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "dist_numeric.png", dpi=150)
    plt.close(fig)


def plot_categorical(real: pd.DataFrame, synth: pd.DataFrame, cols: list[str]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.flat, cols, strict=False):
        p = real[col].value_counts(normalize=True)
        q = synth[col].value_counts(normalize=True)
        cats = p.index.union(q.index)
        x = np.arange(len(cats))
        ax.bar(x - 0.2, p.reindex(cats, fill_value=0), 0.4, color=REAL_COLOR,
               label="Real subscribers")
        ax.bar(x + 0.2, q.reindex(cats, fill_value=0), 0.4, color=SYNTH_COLOR,
               label="Synthetic")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=60, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10)
        style(ax)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", frameon=False)
    fig.suptitle("Categorical columns: share of rows per category")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "dist_categorical.png", dpi=150)
    plt.close(fig)


def plot_corr_diff(diff: pd.DataFrame) -> None:
    lim = max(0.05, float(np.nanmax(np.abs(diff.values))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(diff.values, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(diff)))
    ax.set_yticks(range(len(diff)))
    ax.set_xticklabels(diff.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(diff.index, fontsize=8)
    for i in range(len(diff)):
        for j in range(len(diff)):
            if not np.isnan(diff.values[i, j]):
                ax.text(j, i, f"{diff.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="synthetic corr − real corr")
    ax.set_title("Correlation difference (numeric columns)")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "corr_difference.png", dpi=150)
    plt.close(fig)


def main() -> None:
    data = load_and_preprocess_uci()
    real_all = data.X_train.copy()
    real_all["y"] = data.y_train.values
    real = real_all[real_all["y"] == 1].reset_index(drop=True)
    synth = pd.read_csv(SYNTHETIC_DIR / "bank_ctgan_fill50.csv")
    meta = json.loads((ARTIFACT_DIR / "run_meta.json").read_text())
    losses = pd.read_csv(ARTIFACT_DIR / "ctgan_losses.csv")

    features = [c for c in real.columns if c != "y"]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(real[c])]
    categorical = [c for c in features if c not in numeric]

    say(f"Real training rows: {len(real_all)} (subscribers: {len(real)})")
    say(f"Synthetic rows (largest file): {len(synth)}")
    say(f"Training time: {meta['train_seconds']}s for {meta['params']['epochs']} epochs")

    say("\n== Training losses ==")
    plot_losses(losses)
    last = losses.tail(50)
    g, d = last["Generator Loss"], last["Discriminator Loss"]
    say(f"Generator loss: first epoch {losses['Generator Loss'].iloc[0]:.3f}, "
        f"last-50 mean {g.mean():.3f} (std {g.std():.3f})")
    say(f"Discriminator loss: first epoch {losses['Discriminator Loss'].iloc[0]:.3f}, "
        f"last-50 mean {d.mean():.3f} (std {d.std():.3f})")
    say("Plot: loss_curves.png")

    say("\n== Numeric columns: real subscribers vs synthetic ==")
    rows = []
    for col in numeric:
        r, s = real[col].dropna(), synth[col].dropna()
        ks = ks_2samp(r, s).statistic if len(r) and len(s) else np.nan
        rows.append({
            "column": col, "real_mean": r.mean(), "synth_mean": s.mean(),
            "real_std": r.std(), "synth_std": s.std(),
            "real_median": r.median(), "synth_median": s.median(), "KS": ks,
        })
    say(pd.DataFrame(rows).round(3).to_string(index=False))
    say("KS = largest gap between the two cumulative distributions (0 = identical, 1 = disjoint).")
    plot_numeric(real, synth, numeric)

    say("\n== Categorical columns: total variation distance ==")
    tvd = {}
    for col in categorical + ["y"]:
        p = real[col].value_counts(normalize=True)
        q = synth[col].value_counts(normalize=True)
        cats = p.index.union(q.index)
        gap = p.reindex(cats, fill_value=0) - q.reindex(cats, fill_value=0)
        tvd[col] = 0.5 * float(gap.abs().sum())
    say(pd.Series(tvd, name="TVD").round(3).to_string())
    say("TVD = half the summed gap in category shares (0 = identical, 1 = no overlap).")
    plot_categorical(real, synth, categorical)

    say("\n== Correlation (numeric columns, pairwise, NaN skipped) ==")
    diff = synth[numeric].corr() - real[numeric].corr()
    upper = diff.where(np.triu(np.ones(diff.shape, dtype=bool), k=1)).stack()
    worst = upper.abs().idxmax()
    say(f"Mean absolute difference: {upper.abs().mean():.3f}")
    say(f"Largest difference: {worst[0]} vs {worst[1]} = {upper[worst]:+.3f}")
    plot_corr_diff(diff)

    say("\n== pdays missing share (never contacted) ==")
    say(f"Real, all rows:        {real_all['pdays'].isna().mean():.4f}")
    say(f"Real, subscribers:     {real['pdays'].isna().mean():.4f}")
    say(f"Synthetic:             {synth['pdays'].isna().mean():.4f}")

    say("\n== Value and consistency checks (synthetic) ==")
    days = synth["pdays"].dropna()
    say(f"pdays non-integer: {int((days != days.round()).sum())}, "
        f"below 1: {int((days < 1).sum())} (of {len(days)} non-missing)")
    contacted = synth["previously_contacted"] == 1
    bad_prev = ((synth["previous"] > 0) != contacted).sum()
    bad_days = (synth["pdays"].isna() == contacted).sum()
    say(f"previous vs previously_contacted disagree: {int(bad_prev)} rows")
    say(f"pdays missing vs previously_contacted disagree: {int(bad_days)} rows")

    say("\n== Repeated rows ==")
    key_cols = features + ["y"]
    fill = {"pdays": -1}
    s_keys = synth[key_cols].fillna(fill)
    r_keys = real_all[key_cols].fillna(fill)
    say(f"Duplicate rows within synthetic: {int(s_keys.duplicated().sum())}")
    say(f"Duplicate rows within real training data: {int(r_keys.duplicated().sum())}")
    copied = s_keys.merge(r_keys.drop_duplicates(), on=key_cols, how="inner")
    say(f"Synthetic rows identical to a real row: {len(copied)}")
    say("(run_ctgan_bank.py already drops exact copies of real rows and repeats, so 0 here is "
        "by construction; near-duplicates are not checked.)")

    (ARTIFACT_DIR / "diagnostics.txt").write_text("\n".join(report) + "\n")
    say("\nSaved diagnostics.txt and PNG plots to eda/ctgan_bank/artifacts/")


if __name__ == "__main__":
    main()
