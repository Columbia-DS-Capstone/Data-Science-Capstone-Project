"""Paired project-specific benchmarks for the two-stage orchestrator spike."""

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from spike import (
    Proposal,
    fake_sdk_model,
    gate_for,
    run_agents_sdk,
    run_langgraph,
    summary,
    vertex_clients,
    vertex_stages,
)


@dataclass(frozen=True)
class Case:
    name: str
    context: str
    qualities: tuple[float, ...]


# Scores stand in for deterministic quality metrics; this measures orchestration,
# structured handoffs, and routing, not whether an LLM can evaluate a GAN.
CASES = (
    Case("bank_fidelity_pass",
         "Bank Marketing CTGAN check: mean numeric KS 0.06, mean categorical JS 0.04, "
         "correlation difference 0.05; all configured fidelity limits pass.", (0.83,)),
    Case("bank_fidelity_retry",
         "Bank Marketing CTGAN check: initial categorical JS 0.18 exceeds 0.10; "
         "a retrained generator reduces it to 0.07.",
         (0.42, 0.76)),
    Case("privacy_exhausted",
         "Privacy check: synthetic nearest-neighbor distance remains below the real "
         "validation reference after three CTGAN attempts.",
         (0.20, 0.35, 0.40)),
    Case("smote_utility_retry",
         "Bank Marketing utility check: the first augmentation trails the SMOTENC "
         "PR-AUC benchmark; retry clears the target on real validation rows.",
         (0.55, 0.63)),
    Case("schema_exhausted",
         "Schema check: generated job categories outside the training vocabulary and "
         "negative age values persist across attempts.",
         (0.10, 0.30, 0.45)),
)


def usage_counts(usage) -> tuple[int, int, int]:
    if usage is None:
        raise ValueError("Provider did not report token usage")
    prompt = getattr(usage, "prompt_tokens", None)
    output = getattr(usage, "completion_tokens", None)
    if prompt is None:
        prompt, output = usage.input_tokens, usage.output_tokens
        details = usage.input_tokens_details
    else:
        details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    if prompt is None or output is None or cached > prompt:
        raise ValueError("Provider returned incomplete token usage")
    return prompt, output, cached


def run_case(case: Case, framework: str, client=None, model=None,
             prices: tuple[float, float, float] | None = None) -> dict:
    def quality(attempt):
        return case.qualities[min(attempt - 1, len(case.qualities) - 1)]
    tokens = [0, 0, 0]
    calls = 0

    def record(usage):
        nonlocal calls
        counts = usage_counts(usage)
        calls += getattr(usage, "requests", 1)
        for index, count in enumerate(counts):
            tokens[index] += count

    start = time.perf_counter()
    error = None
    valid = False
    attempts = None
    try:
        if framework == "langgraph":
            if client:
                propose, judge = vertex_stages(client, model, quality, case.context, record)
            else:
                def propose(attempt):
                    return Proposal(attempt=attempt, quality=quality(attempt))

                judge = gate_for
            state = run_langgraph(propose, judge, max_attempts=3)
        else:
            from agents import OpenAIChatCompletionsModel, set_tracing_disabled

            set_tracing_disabled(True)
            sdk_model = (OpenAIChatCompletionsModel(model=model, openai_client=client)
                         if client else fake_sdk_model(quality))
            state = run_agents_sdk(sdk_model, max_attempts=3, quality_for_attempt=quality,
                                   context=case.context, on_usage=record if client else None)
        result = summary(state)
        attempts = result["attempts"]
        expected_attempts = next((i for i, q in enumerate(case.qualities, 1) if q >= 0.6), 3)
        valid = (attempts == expected_attempts and result["passed"] ==
                 (case.qualities[expected_attempts - 1] >= 0.6) and
                 len(result["decisions"]) == 2 * attempts)
        if not valid:
            error = "Unexpected routing or decision log"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start
    cost = None
    if client and prices and calls:
        input_price, output_price, cached_price = prices
        cost = ((tokens[0] - tokens[2]) * input_price + tokens[1] * output_price +
                tokens[2] * cached_price) / 1_000_000
    return {"case": case.name, "framework": framework, "seconds": round(elapsed, 3),
            "attempts": attempts, "valid": valid, "error": error,
            "reported_calls": calls if client else None,
            "input_tokens": tokens[0] if client else None,
            "output_tokens": tokens[1] if client else None,
            "cached_input_tokens": tokens[2] if client else None,
            "estimated_usd": round(cost, 8) if cost is not None else None}


def aggregate(rows: list[dict]) -> dict:
    auth_pairs = {(row["case"], row.get("repeat")) for row in rows
                  if (row.get("error") or "").startswith("AuthenticationError:")}
    rows = [row for row in rows if (row["case"], row.get("repeat")) not in auth_pairs]
    result = {}
    for framework in ("langgraph", "agents_sdk"):
        subset = [row for row in rows if row["framework"] == framework]
        if not subset:
            result[framework] = {"runs": 0, "errors": 0, "error_rate": None,
                                 "valid_runs": 0, "latency_median_s": None,
                                 "latency_p95_s": None, "input_tokens": None,
                                 "output_tokens": None, "estimated_usd": None}
            continue
        seconds = sorted(row["seconds"] for row in subset)
        result[framework] = {
            "runs": len(subset), "errors": sum(row["error"] is not None for row in subset),
            "error_rate": sum(row["error"] is not None for row in subset) / len(subset),
            "valid_runs": sum(row["valid"] for row in subset),
            "latency_median_s": statistics.median(seconds),
            "latency_p95_s": seconds[max(0, (95 * len(seconds) + 99) // 100 - 1)],
            "input_tokens": (sum(row["input_tokens"] for row in subset)
                             if all(row["input_tokens"] is not None for row in subset) else None),
            "output_tokens": (sum(row["output_tokens"] for row in subset)
                              if all(row["output_tokens"] is not None for row in subset) else None),
            "estimated_usd": (round(sum(row["estimated_usd"] for row in subset), 8)
                              if all(row["estimated_usd"] is not None for row in subset) else None),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use the local HTTP fixture")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Override the results file path")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    sync_client = async_client = model = None
    prices = None
    if not args.offline:
        try:
            sync_client, async_client, model = vertex_clients()
        except RuntimeError as exc:
            parser.error(str(exc))
        if model == "google/gemini-3.5-flash":
            location = os.environ["GOOGLE_CLOUD_LOCATION"]
            prices = (1.50, 9.00, 0.15) if location == "global" else (1.65, 9.90, 0.165)
    import agents  # noqa: F401
    import langgraph.graph  # noqa: F401

    output = args.output or Path(__file__).with_name(
        f"benchmark_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("")
    rows = []
    for repeat in range(args.repeats):
        if repeat and not args.offline:
            sync_client, async_client, model = vertex_clients()
        for index, case in enumerate(CASES):
            order = ("langgraph", "agents_sdk") if (repeat + index) % 2 == 0 else (
                "agents_sdk", "langgraph")
            for framework in order:
                client = (sync_client if framework == "langgraph" else async_client)
                row = run_case(case, framework, client if not args.offline else None, model,
                               prices)
                row["repeat"] = repeat + 1
                rows.append(row)
                print(json.dumps(row), flush=True)
                with output.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
    result = {"summary": aggregate(rows), "model": model,
              "prices_usd_per_million": prices,
              "excluded_auth_pairs": len({(row["case"], row["repeat"]) for row in rows
                                          if (row.get("error") or "").startswith(
                                              "AuthenticationError:")})}
    with output.open("a") as stream:
        stream.write(json.dumps(result) + "\n")
    print(json.dumps({**result, "log_file": str(output)}))


if __name__ == "__main__":
    main()
