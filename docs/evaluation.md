# Evaluation and reporting agent

`agents.evaluation.run_evaluation(request, run, client=...)` implements
`EvaluationInput -> EvaluationOutput`. The default graph calls it after training.
`EvaluationAgent(client=...)` is an injectable callable for tests or custom pipelines.

The dependency tools and tests were imported without unrelated branch changes from
`origin/pH7/sprint3-eval-tools-and-llm-diagnosis`, commit
`86c56f679bc709ef0f20bcad8bd7828f9627b8d8` (the evaluation-tools work for #75).

The agent creates structured results from the training handoff, calls `compare_arms`
and `reliable_gain`, and evaluates selected models' test predictions with
`segment_metrics`. It compares arms against original and cleaned data, and GAN
against SMOTENC at matched fill levels. Selection comes from validation; the report
does not select a new model, threshold, or fill level from test results. An unavailable
GAN arm is omitted from the comparisons. Missing arms are valid.

The arm comparisons report PR-AUC, AUROC, F1, accuracy, precision and recall.
The random baseline is 0.5 for AUROC and the observed positive rate of each split
for PR-AUC. For accuracy it is the expected accuracy of random labels drawn at
that split's positive rate. The report prints the rates and baselines alongside
the scores, and number verification reads them from `comparisons.csv`.

Segment metrics use source-row joins to the source dataset, with observed binary
labels and real-row provenance checks. Available age, job, marital, education and
Default demographic columns are evaluated. Numeric segment bands use evaluated
rows only. Interval labels also export numeric `segment_lower_bound` and
`segment_upper_bound` evidence, so a quoted boundary can pass number verification
without allowing arbitrary numbers. Missing segment values remain excluded.
Small test sets and small positive-count groups are flagged. Missing
segment columns are reported as a limitation, not evidence of fairness.

## Verified narrative and fallback

The shared LLM runtime requests a Pydantic response at temperature zero with a fixed
outline: summary, improvements, segments, negative/inconclusive results and limitations.
The response cannot change rules or thresholds. Configuration, tool rules and result
artifact hashes are checked around the call.

The response model validates the **complete rendered report** with
`verify_report_numbers`. A made-up number, missing section, unknown field or failed
call consumes the shared retry budget: at most two calls in total. After repeated
failure, the runtime logs `LLM_FALLBACK` and returns a verified plain template.
The final rendered report is verified again before publication. If even the template
or final report fails verification, the output is `FAILED` with
`numbers_verified=False`; an unverified report is never returned as successful.
All LLM calls and number-rejection/final-verification decisions are logged.

The report always includes negative/inconclusive results, small-sample caveats,
augmentation availability, the recorded volume decision (or its absence), audit
references, and ethical limitations. Verdicts and tables come from tools, not LLM
calculations. The JSON evidence retains skipped-fit reasons, the quality report,
GAN-unavailability reason, volume rationale and audit paths. Constant validation-score
models excluded by training remain in the metrics and evidence, and are called out
in the limitations section; they do not enter selected-model comparisons.

## Artifacts

All files are written under `report/` and returned for manifest registration:

- `metrics.csv`: the training handoff's reported metrics, including selection/status.
- `comparisons.csv`: means, seed spread, paired differences by reference arm, bootstrap
  intervals (`boot_*`) and split baselines.
- `segments.csv`: per-model, per-seed segment metrics and small-group flags.
- `evidence.json`: deterministic verdicts, test counts, rule constants and audit context.
- `final_report.md`: verified narrative, comparison/verdict tables and caveats.

The available #75 tools cover PR-AUC, F1, precision, recall, seed-based comparisons,
segments and number checking. They do not supply SHAP analysis, calibration analysis
or PDF export; the agent does not claim to have computed those. With at least three paired
seeds the reliable-gain rule compares the paired mean PR-AUC lift with the seed-to-seed spread
on both validation and test. With fewer (one seed, 42, is the default) it uses the paired
bootstrap described below. Neither is a significance test. Selected models may differ across
arms.

## Uncertainty with one seed

Every experiment uses one seed (42), so arm comparisons take their uncertainty from a paired
bootstrap over the rows of each evaluation split (`paired_bootstrap` in `tools/evaluation.py`):

- Each resample draws the positive rows, and the negative rows, with replacement, so the class
  counts never change, and scores both arms on the same drawn rows. There are 1,000 resamples.
- The interval is the 2.5th and 97.5th percentiles of the metric difference (arm minus
  reference). `compare_arms` adds it to `comparisons.csv` as `boot_diff`, `boot_lo` and
  `boot_hi` for PR-AUC and AUROC, with the model seed it used (`boot_seed`, the lowest seed
  both arms share).
- Verdict: a gain if the interval's lower bound is above 0 on both validation and test, a loss if
  its upper bound is below 0 on both, otherwise inconclusive. The verdict table names the rule
  that decided (`basis`).
- The scores come from each selected model's saved prediction files (`load_selected_scores`).
  If they are missing the verdict stays "not enough paired seeds" and the run logs a `bootstrap`
  decision.

**Limits to state in the paper.** The bootstrap covers the sampling noise of the evaluation
rows: a different draw of customers from the same population. It does not cover the randomness
of model training or of the generator, because both are fixed at seed 42, and it does not cover
the choice of split. A seed-to-seed spread measures training randomness, which is a different
thing, and it is no longer the default.

Number verification establishes that numeric tokens occur in the result artifacts,
with the tool's rounding/percentage conventions. It does not prove that every prose
claim associates a valid number with the correct metric, nor validate causal or
fairness interpretations; the fixed tables remain the authoritative evidence.

## Bank Marketing deciles

Export the selected cleaned model's deciles from a saved Bank training-agent run.
Choose a seed explicitly for a run with more than one selected cleaned model.

```bash
uv run python -m agentic_gan.tools.bank_deciles runs/<run_id> --seed 42
```

The exporter reads the validation and test prediction CSVs and scores the
cleaned training rows with the saved estimator and its frozen threshold. It
writes three CSVs and three response-rate charts under `results/`, along with
business takeaways and any held-out response-rate increases between deciles. Each
chart draws that split's overall response rate (11.7% on Bank) as a dotted reference line.
Train results are in-sample. Equal scores are ordered by source-row ID.
For runs with a fitted treatment plan, the estimator uses the saved raw split
to clean the retained training rows with the same fitted encoder.

## Offline tests

```bash
uv run pytest tests/test_evaluation_agent.py tests/tools/test_evaluation.py
```

Tests use `FakeLLM` and no credentials. They cover accepted narratives, fabricated
numbers and regeneration, exhausted retries, network/schema failures, changed rules,
failed final checks, absent GAN/volume decisions, artifact manifests, and the default
pipeline reaching a real report. There is no live LLM test.
