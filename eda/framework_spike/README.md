# Orchestrator framework spike

[`spike.py`](spike.py) runs the same two stages in LangGraph and the OpenAI Agents SDK: a Pydantic `Proposal`, then a Pydantic `Gate`. A failed gate loops back up to `max_attempts`. Routing checks the gate against a deterministic quality rule; neither model chooses the next stage. Both paths keep the same state and can write decision JSONL files.


The SDK's offline model is an HTTP fixture, so this checks its real `Runner` and structured-output handling without API charges. Default cap 2 fails once then passes; `--max-attempts 1` stops after a failed gate.

## Check Vertex AI with team credits

Put these values in `eda/framework_spike/.env` (already ignored by git). The scripts load this file automatically; existing shell environment variables take precedence. Use Application Default Credentials; no key is stored here:

```sh
export GOOGLE_CLOUD_PROJECT='<team-project-id>'
export GOOGLE_CLOUD_LOCATION='global'
export VERTEX_MODEL='google/gemini-3.5-flash'
```

`--check-vertex` makes one structured Gemini call. Set `VERTEX_MODEL` if the team uses a model other than `google/gemini-3.5-flash`, and choose a location that serves that model. Use `--vertex` to run both full workflows through Vertex. The spike refreshes the ADC token once per process; a long-running orchestrator needs token refresh before expiry.

Vertex is reachable. The first live run had three late `AuthenticationError` failures after its initial ADC token aged out; the benchmark now refreshes ADC between repeats.

## Two-stage orchestration benchmark

The question is whether each framework can
- Carry a typed result through a quality gate
- Make the **same deterministic retry decision**
- Record the cost of doing so.

This is a small stand-in for the `Model Selection → Quality Evaluation` loop in the proposed pipeline.

For each attempt,
1. Gemini receives a project-specific scenario and an **assigned** quality score.
2. Returns a Pydantic `Proposal(attempt, quality)`, then a Pydantic `Gate(passed, reason)`.
3. Code verifies that `passed` agrees with `quality >= 0.6`, records one proposal and one gate decision, and either stops or retries up to three attempts.

The LangGraph version uses nodes and a conditional edge; the Agents SDK version uses two `Runner` calls inside a Python loop. A valid run has the expected attempt count, final verdict, and **two decision-log entries per attempt**. Each attempt normally makes two model calls.

The scores below are **test inputs**. “Retry” means calling the proposal stage again with the next assigned score. There is no GAN trained

### Five benchmark cases

1. **Fidelity passes immediately (`bank_fidelity_pass`).**
- The prompt says a Bank Marketing CTGAN result has mean numeric KS 0.06, mean categorical JS 0.04, and correlation difference 0.05.
- We assign quality **0.83**. The expected output is `Proposal(attempt=1, quality=0.83)`, then a passing gate because 0.83 ≥ 0.6.
- The run stops with one attempt, two decisions, and normally two model calls.
- This exercises the route with no retry.
2. **Fidelity improves on retry (`bank_fidelity_retry`).**
- The prompt says categorical JS starts at 0.18, above a 0.10 limit, and improves to 0.07 after retraining.
- We assign **0.42**, then **0.76**. The first gate must fail; the second must pass.
- The expected record is two attempts, four ordered decisions, and normally four model calls.
- This exercises the conditional loop and its exit after improvement; the retraining is described, not performed.
3. **Privacy never passes (`privacy_exhausted`).**
- The prompt says synthetic rows remain closer to real training rows than the validation reference.
- We assign **0.20 → 0.35 → 0.40**. Every gate must fail; the run must stop after the third attempt, with six decisions and normally six model calls.
- This checks that a persistent failure cannot loop forever. No nearest-neighbor distances are computed here.
4. **Utility improves against SMOTENC (`smote_utility_retry`).**
- The prompt says an augmented model initially trails the SMOTENC PR-AUC target, then clears it.
- We assign **0.55 → 0.63**. The expected route is fail, retry, pass: two attempts and four decisions/calls.
- This checks the same retry rule with a downstream utility scenario. No classifier is trained and no PR-AUC is measured here.
5. **Schema never passes (`schema_exhausted`).**
- The prompt says generated job categories fall outside the training vocabulary and negative ages persist.
- We assign **0.10 → 0.30 → 0.45**. All three gates must fail, then the cap stops the run with six decisions and normally six model calls.
- The script does not generate or inspect records; it tests routing after a stated schema failure.



The live benchmark repeats all five cases three times in each framework (30 planned runs), alternating which framework goes first. It measures the full run's wall time, provider-reported tokens and calls, route validity, and list-price cost. It does **not** train a GAN, compute the data-quality metrics, evaluate classifier predictions, or test the full six-stage pipeline. The two framework implementations use different prompt wrappers, so token differences reflect these implementations as well as framework overhead.

```sh
uv run --no-project --with langgraph --with openai-agents --with google-auth \
  python eda/framework_spike/benchmark.py
```

Each JSONL row records end-to-end wall time, attempts, expected-route validity, errors, reported API calls, input/output/cached tokens, and estimated USD. The final printed summary gives median and nearest-rank p95 latency, errors/runs, valid runs, token totals, and cost totals.

Authentication failures and the matching framework's run are excluded from the summary; raw rows remain in the log. Error rate covers remaining failed or invalid runs only. The SDK counts each completed `Runner` call via its usage object; LangGraph counts each completed Vertex Chat Completion response. Both include model calls caused by quality-gate retries.

Every run writes raw rows and a summary to a timestamped `benchmark_*.jsonl` here. The first live run is in [`benchmark_results.jsonl`](benchmark_results.jsonl); logs are git-ignored because errors can contain project details.

## Live results

These figures use **13 matched successful case/repeat pairs** per framework. Both frameworks' rows were excluded for two pairs affected by authentication failures, so each comparison uses identical cases. All 13 included pairs routed correctly in both frameworks. Calls and tokens are totals; time is median seconds per run. Values are **LangGraph / Agents SDK**.

| Case | Pairs | Calls each | Median s | Tokens | Estimated USD |
|---|---:|---:|---:|---:|---:|
| **All matched cases** | **13** | **56** | **45.46 / 37.62** | **7,873 / 8,646** | **0.021657 / 0.023784** |
| Fidelity pass | 3 | 6 | 4.70 / 4.50 | 925 / 992 | 0.002475 / 0.002561 |
| Fidelity retry | 3 | 12 | 62.69 / 73.99 | 1,809 / 2,036 | 0.004941 / 0.005949 |
| Privacy exhausted | 3 | 18 | 45.46 / 37.62 | 2,368 / 2,604 | 0.006282 / 0.006854 |
| SMOTENC utility retry | 2 | 8 | 114.49 / 66.22 | 1,145 / 1,238 | 0.003165 / 0.003312 |
| Schema exhausted | 2 | 12 | 120.79 / 168.19 | 1,626 / 1,776 | 0.004794 / 0.005109 |

The SDK used about 9.8% more tokens and list-price cost in this implementation. The small, sequential sample does not establish a speed winner. Costs are estimates before credits.

The cost estimate uses `(uncached input × input rate + cached input × cached rate + output × output rate) / 1,000,000`. For the default Gemini 3.5 Flash, the script uses [Google's standard Vertex prices](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing): global $1.50/$0.15/$9.00 per million uncached input/cached input/output tokens; non-global $1.65/$0.165/$9.90. Cost stays null for other models. This is a list-price estimate before credits, not a billing measurement. Offline runs have null tokens and cost. Imports are warmed before timing; workflow setup, Vertex time, and transient retries are included.

## Conclusion

- **Why the prompts differ:** LangGraph calls Vertex Chat Completions directly, with instructions in the user message. The Agents SDK puts task rules in `Agent` instructions and sends a shorter user message. Both receive the same case, assigned scores, model, and gate threshold; the different prompt wrappers mean token and cost differences cannot be attributed to the framework alone.
- **Choice: LangGraph.** The planned orchestrator has a fixed stage order and a quality-gate retry loop. `StateGraph` keeps shared state and the retry condition visible as nodes and edges. The Agents SDK also passed these cases, but this workflow needs a separate Python loop and explicit state passing around its `Runner` calls.
- **What the results support:** Both routed all 13 matched live cases correctly. LangGraph used fewer tokens and estimated dollars in this implementation; the small sequential run does not establish a latency winner.
- **Limits:** LangGraph needs more graph wiring. This spike covers only two stages with assigned scores, not the full pipeline, GAN training, metric calculation, or the separate stage-error retry rule. Prompt differences and two excluded authentication-affected pairs limit the performance comparison.
