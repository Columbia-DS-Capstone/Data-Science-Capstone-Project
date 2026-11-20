# Fine volume grid (#36)

Implementation base: GitHub main `fedde3a` (fetched November 20, 2026).
Model seed 42, fixed 60/20/20 splits and original-arm validation-F1 thresholds.
The cleaned arm is the reference. Every cell selects its model on validation
PR-AUC. These are offline comparisons with supplied CTGAN V2 files, not quality-gate
acceptance or an end-to-end pipeline run.

- Bank run: `bank_marketing_20261120-071833` (48 successful fits, 30 seconds).
- Archived Default diagnostic run: `default_20261120-071906` (48 successful fits, 24 seconds).
- No synthetic rows in either evaluation split in either run.
- CTGAN: nested prefixes of the existing `*_ctgan_v2_fill25.csv` pool.
- SMOTENC: one seed-42 pool per dataset covering the largest requested fill (25%),
  with nested prefixes. Its pool size differs from the historical 50% grid, so its
  10%/25% cells need not reproduce that grid's synthetic draws.

The committed Default pool predates #136 (November 4, commit `a3c27e4`);
its metadata says no rules were applied and raking used all six PAY codes.
It is **not** the newly merged recipe (rule filter, four rake columns, discrete
integer-code raking). The Default rows are explicitly labelled as an archived-pool
diagnostic and do not satisfy the repeat-after-#136 requirement. That repeat needs
Kyle's current-recipe pool/run artifacts, or a new CTGAN training run. No current
Default-recipe results are inferred from the archived files.

`grid.csv` reports validation and test PR-AUC for every cell and the #138 verdict
against cleaned (95% paired bootstrap, 1,000 stratified resamples, bootstrap seed 0).
`comparisons.csv` saves the intervals for both splits. `*_decisions.json` saves
validation-only decisions and candidate evidence. `fine_grid.png` plots Bank test
PR-AUC and frozen-threshold F1, with the dotted cleaned reference.

Reproduce training and decisions with the commands in `docs/training.md`, using
`--results-dir results/volume` to preserve the historical cross-dataset scoreboard.
The run folders hold selected-model probabilities and are intentionally untracked.

Dependencies checked on GitHub: #138 is closed and its helpers are merged;
#136's Default recipe is merged by PR #143, although #136 remains open because
its fidelity gate and test target were not met. #134's orchestrator wiring is
separate work. #135's weighted experiment explicitly waits for #36 to merge.
No issue comments or shared-deck changes were published by this implementation.

Environment: LightGBM used the existing scikit-learn `libomp.dylib` via
`DYLD_FALLBACK_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/sklearn/.dylibs`;
no packages or system libraries were modified.

Result: Bank selects 0% for both CTGAN and SMOTENC; no tested cell has a reliable gain over cleaned.
