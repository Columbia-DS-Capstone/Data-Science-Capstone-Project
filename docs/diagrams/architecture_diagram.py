"""Draw docs/architecture_diagram.png, the static copy of the system overview in section 3.

    uv run python docs/diagrams/architecture_diagram.py

Keep it in step with the mermaid diagram in docs/architecture.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

INK = "#1F2933"
LINE = "#4B5563"
ORCH = ("#E4E9EF", "#52606D")
AGENT = ("#D9E8F6", "#2F6DA3")
GATE = ("#FBE8B8", "#B7791F")
ARMS = {
    "raw": ("#E7E9EC", "#6B7280"),
    "baseline": ("#D6EEDD", "#2F855A"),
    "smote": ("#FADADD", "#B83280"),
    "gan": ("#E6DDF6", "#6B46C1"),
}


def box(ax, x, y, w, h, text, colours, size=11.5, bold=False):
    face, edge = colours
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.3,rounding_size=1.6",
            fc=face,
            ec=edge,
            lw=1.8,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        weight="bold" if bold else "normal",
        linespacing=1.4,
    )


def arrow(ax, points, label=None, label_at=None, size=10):
    """A polyline with an arrowhead on its last segment."""
    for a, b in zip(points[:-2], points[1:-1], strict=False):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=LINE, lw=1.8, solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=18,
            lw=1.8,
            color=LINE,
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        ax.text(
            *label_at,
            label,
            fontsize=size,
            color=INK,
            ha="center",
            va="center",
            linespacing=1.3,
            bbox={"fc": "white", "ec": "none", "pad": 1.5},
        )


def draw(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9.9))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 99)
    ax.axis("off")

    ax.text(2, 96, "Agentic GAN framework: system overview", fontsize=19, weight="bold", color=INK)
    ax.text(
        2,
        92.3,
        "v0.2, 16 October 2026. Sub-agents never call each other; the orchestrator "
        "routes every step.",
        fontsize=11.5,
        color=LINE,
    )

    # top row: config to quality gate
    box(ax, 2, 64, 15, 14, "Run config\n+ dataset", ORCH)
    box(ax, 22, 64, 22, 14, "S0 (orchestrator)\nvalidate schema,\nsplit 60/20/20", ORCH)
    box(ax, 49, 64, 25, 14, "Pre-processing Agent\nprofile and clean:\nraw + baseline files", AGENT)
    box(
        ax,
        79,
        64,
        28,
        14,
        "Model Selection Agent\nchoose GAN, train,\none set per fill level",
        AGENT,
    )
    gx, gy = 128, 71
    ax.add_patch(
        Polygon(
            [(gx - 17, gy), (gx, gy + 11), (gx + 17, gy), (gx, gy - 11)],
            closed=True,
            fc=GATE[0],
            ec=GATE[1],
            lw=1.8,
        )
    )
    ax.text(
        gx,
        gy,
        "Quality Evaluation\nAgent: gate\nper fill level",
        ha="center",
        va="center",
        fontsize=11,
        color=INK,
        linespacing=1.35,
    )

    arrow(ax, [(17.5, 71), (21.4, 71)])
    arrow(ax, [(44.6, 71), (48.4, 71)])
    arrow(ax, [(74.6, 71), (78.4, 71)])
    arrow(ax, [(107.6, 71), (111, 71)])

    # retry loop above
    arrow(ax, [(gx, gy + 11), (gx, 84), (93, 84), (93, 78.7)])
    ax.text(
        111,
        87.2,
        "FAIL, attempts left:\nretry with the failed report",
        fontsize=10.5,
        color=INK,
        ha="center",
        va="center",
        linespacing=1.3,
    )

    # bottom row: training to evaluation
    box(
        ax,
        104,
        36,
        48,
        15,
        "Model Training Agent\nthe same models on four arms,\nover all seeds",
        AGENT,
    )
    box(
        ax,
        62,
        36,
        32,
        15,
        "Volume rule (orchestrator)\nsmallest fill level with\na reliable gain, else none",
        ORCH,
    )
    box(ax, 22, 36, 32, 15, "Evaluation Agent\nmetrics, SHAP, segments,\nreport", AGENT)

    arrow(
        ax,
        [(gx - 6, gy - 7.3), (gx - 6, 52)],
        "PASS: passing\nfill levels",
        (gx - 13, 58),
        size=10.5,
    )
    arrow(ax, [(gx + 17, gy), (156, gy), (156, 43.5), (152.7, 43.5)])
    ax.text(
        154.5,
        56,
        "FAIL, attempts exhausted:\nno GAN arm, run continues",
        fontsize=10.5,
        color=INK,
        ha="right",
        va="center",
        linespacing=1.3,
    )
    arrow(ax, [(103.4, 43.5), (94.6, 43.5)])
    arrow(ax, [(61.4, 43.5), (54.6, 43.5)])

    box(ax, 22, 17, 32, 11, "Run folder\nreport, logs, artifacts", ORCH)
    arrow(ax, [(38, 35.4), (38, 28.7)])

    # arms under training
    arms = [
        ("raw", "no repair"),
        ("baseline", "cleaned,\nimputed"),
        ("smote", "+ SMOTE\nrows"),
        ("gan", "+ synthetic\nrows"),
    ]
    centres = []
    for i, (name, note) in enumerate(arms):
        x = 104 + i * 12.3
        box(ax, x, 18, 11.2, 13, f"{name}\n{note}", ARMS[name], size=10.2)
        ax.plot([x + 5.6, x + 5.6], [31.4, 33], color=LINE, lw=1.4)
        centres.append(x + 5.6)
    ax.plot([centres[0], centres[-1]], [33, 33], color=LINE, lw=1.4)
    ax.plot([128, 128], [33, 35.4], color=LINE, lw=1.4)
    ax.text(
        128,
        14.6,
        "Four arms. Fill level: share of the class gap, set per dataset.",
        fontsize=10.5,
        color=LINE,
        ha="center",
        va="center",
    )

    # orchestrator band
    ax.add_patch(
        FancyBboxPatch(
            (2, 2.5),
            156,
            7.5,
            boxstyle="round,pad=0.3,rounding_size=1.6",
            fc=ORCH[0],
            ec=ORCH[1],
            lw=1.6,
        )
    )
    ax.text(
        80,
        6.25,
        "Orchestrator: owns control flow, retries and the audit trail.   "
        "state.json   manifest.json   decisions.jsonl",
        ha="center",
        va="center",
        fontsize=11.5,
        color=INK,
    )

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diamond(ax, cx, cy, hw, hh, text):
    ax.add_patch(
        Polygon(
            [(cx - hw, cy), (cx, cy + hh), (cx + hw, cy), (cx, cy - hh)],
            closed=True,
            fc=GATE[0],
            ec=GATE[1],
            lw=1.8,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=11, color=INK, linespacing=1.3)


def note(ax, x, y, text, ha="center"):
    ax.text(x, y, text, fontsize=10.5, color=INK, ha=ha, va="center", linespacing=1.3)


def draw_volume_loop(path: Path) -> None:
    """The volume loop of section 4.1, from slide 6 of the 9 October deck."""
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.text(
        2,
        58,
        "The volume loop: how much synthetic data to add",
        fontsize=19,
        weight="bold",
        color=INK,
    )

    y = 38
    box(ax, 2, y - 7, 21, 14, "Profile training\ndata, set the\nreal-only baseline", AGENT)
    box(ax, 28, y - 7, 22, 14, "Choose volumes,\ngenerate samples\n(one per fill level)", AGENT)
    diamond(ax, 66, y, 11, 10, "Quality\npasses?")
    box(ax, 82, y - 7, 25, 14, "Test accepted volumes\non real validation\ndata", AGENT)
    diamond(ax, 122, y, 11, 10, "Reliable\ngain?")
    box(ax, 138, y - 7, 20, 14, "Select smallest\nvolume near the\nbest score", AGENT)
    box(ax, 38, 6, 22, 12, "Reject batch,\nrevise generator", ORCH)
    box(ax, 100, 6, 36, 12, "Use real data only,\nadd 0 rows", ORCH)

    arrow(ax, [(23.4, y), (27.4, y)])
    arrow(ax, [(50.6, y), (54.9, y)])
    arrow(ax, [(77.1, y), (81.4, y)], "yes", (79.3, y + 3), size=10.5)
    arrow(ax, [(107.6, y), (110.9, y)])
    arrow(ax, [(133.1, y), (137.4, y)], "yes", (135.3, y + 3), size=10.5)

    # quality fails: revise and try again, or give up when the budget is used
    arrow(ax, [(66, y - 10), (66, 12), (60.6, 12)])
    note(ax, 62.5, 27, "no, within\nbudget", ha="right")
    arrow(ax, [(49, 18.6), (49, y - 7.6)])
    arrow(ax, [(66, 12), (66, 3), (118, 3), (118, 5.4)])
    note(ax, 92, 0.6, "no, budget exhausted", ha="center")

    # no reliable gain
    arrow(ax, [(122, y - 10), (122, 18.6)])
    note(ax, 124, 27, "no", ha="left")

    ax.text(
        2,
        52,
        'Orchestrator logic. The rule for "reliable gain" is in section 4.1 of '
        "docs/architecture.md (proposed, issue #36).",
        fontsize=11.5,
        color=LINE,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    docs = Path(__file__).resolve().parent.parent
    draw(docs / "architecture_diagram.png")
    draw_volume_loop(docs / "volume_loop_diagram.png")
