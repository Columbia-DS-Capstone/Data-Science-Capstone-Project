"""Append-only decisions for one run."""

import json

import pandas as pd

from .manifest import RunDir


def log_decision(agent, rule, inputs, outcome, rationale, *, run: RunDir) -> None:
    entry = dict(agent=agent, rule=rule, inputs=inputs, outcome=outcome, rationale=rationale)
    with run.decisions_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, default=str) + "\n")


def read_decisions(run: RunDir) -> pd.DataFrame:
    columns = ["agent", "rule", "inputs", "outcome", "rationale"]
    if not run.decisions_path.exists():
        return pd.DataFrame(columns=columns)
    with run.decisions_path.open(encoding="utf-8") as file:
        return pd.DataFrame((json.loads(line) for line in file), columns=columns)
