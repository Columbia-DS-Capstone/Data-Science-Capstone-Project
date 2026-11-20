"""The Bank decile charts: each split's overall response rate is drawn as a reference line."""

import pandas as pd
import pytest

from agentic_gan.tools import bank_deciles


def decile_rows(positives, n=100):
    """The columns the chart reads: ten deciles of `n` customers with these subscriber counts."""
    return pd.DataFrame(
        {
            "decile": range(1, 11),
            "n": [n] * 10,
            "positives": positives,
            "actual_rate": [count / n for count in positives],
            "mean_predicted_probability": [count / n * 0.9 for count in positives],
        }
    )


def reference_line(figure):
    (line,) = [line for line in figure.axes[0].lines if line.get_label().startswith("Overall")]
    return line


def test_the_chart_draws_the_overall_response_rate_as_a_horizontal_line():
    table = decile_rows([50, 20, 10, 5, 5, 4, 3, 2, 1, 0])  # 100 subscribers among 1,000

    figure = bank_deciles._chart(table, "test", 0.6)

    line = reference_line(figure)
    assert list(line.get_ydata()) == pytest.approx([0.10, 0.10])  # the same value across the chart
    assert line.get_label() == "Overall response rate (10.0%)"
    legend = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert legend == ["Actual response", "Predicted response", "Overall response rate (10.0%)"]


def test_each_split_is_drawn_at_its_own_rate():
    train = decile_rows([40, 20, 10, 5, 5, 4, 3, 2, 1, 0])  # 90 of 1,000
    test = decile_rows([60, 30, 15, 8, 6, 4, 3, 2, 1, 0])  # 129 of 1,000

    rates = [
        reference_line(bank_deciles._chart(table, split, 0.7)).get_ydata()[0]
        for split, table in (("train", train), ("test", test))
    ]

    assert rates == pytest.approx([0.09, 0.129])


def test_the_chart_is_written_to_the_requested_file(tmp_path):
    path = tmp_path / "decile.png"

    bank_deciles._write_chart(decile_rows([50, 20, 10, 5, 5, 4, 3, 2, 1, 0]), "test", 0.6, path)

    assert path.read_bytes().startswith(b"\x89PNG") and path.stat().st_size > 10_000
