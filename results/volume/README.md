# Fine volume grids (#36)

The Bank grid and the required Default repeat are complete. Both use seed 42,
fixed 60/20/20 splits, frozen original-arm validation-F1 thresholds and models
selected on validation PR-AUC. Each dataset uses its own cleaned reference.

- **Bank:** cleaned validation/test PR-AUC **0.4500 / 0.4811**. CTGAN and SMOTENC both select **0%**.
- **Default, current V2 pipeline files from PR #149:** cleaned **0.5318 / 0.5576**. Both arms select **0%**; all ten synthetic cells have inconclusive #138 verdicts.

No candidate has a strictly positive validation lift CI lower bound. Selection
uses validation only. Verdicts use 95% paired-bootstrap intervals on both splits,
with 1,000 class-stratified resamples and bootstrap seed 0.

The consolidated `grid.csv`, `comparisons.csv`, full model metrics and
`fine_grid.png` now contain the Bank grid and the current Default repeat.
`bank_marketing_decisions.json` and `default_decisions.json` save the decisions.

See [Default report](default_v2pipe/README.md) for the ten cells, selected models,
input hashes, reproduction command, chart and verification. The current files
reproduce all five published SMOTENC cells but differ from the CTGAN table in
#136. A second training runner confirms the current CTGAN 10% result; the report
records the discrepancy without inferring its cause.

Runs (all successful, no synthetic holdout rows):
- Bank: `bank_marketing_20261120-071833`, 48 fits, 30 seconds.
- Current Default: `default_20261120-161532`, 48 fits, 47 seconds.
- Archived Default diagnostic: `default_20261120-071906`, 48 fits, 24 seconds.

The old November 4 Default pool is historical. Its earlier diagnostic tables
are preserved in `archived_default/`, and its decisions remain explicitly named
`default_archived_pool_decisions.json`. They do not represent the new recipe.

CTGAN levels are nested prefixes of one 25% file. SMOTENC also uses one seed-42
pool sized for 25% fill. Its lower-fill draws can differ from the historical
50%-maximum grid. See `docs/training.md` for the commands.

These are offline training comparisons. #136's Default quality-gate failures
remain separate from the volume experiment, and orchestrator wiring belongs to
#134. The #135 Bank weighting sweep is complete in `results/weights/README.md`.
The Bank results are posted on #36 and #135. Slides were delivered in chat to
match the prior deck; the shared deck was not edited.

Model/prediction artifacts remain in untracked `runs/`. The existing scikit-learn
OpenMP library was supplied through `DYLD_FALLBACK_LIBRARY_PATH`; no packages or
system libraries were modified.
