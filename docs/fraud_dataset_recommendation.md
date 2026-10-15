# Credit Card Fraud dataset: recommendation

Issue #24. Analysis by `eda/fraud/fraud_analysis.py`, run on the raw ULB file (`creditcard.csv`, 284,807 rows, 492 fraud). Prepared 15 October 2026 for the mentor meeting on 16 October.

## Recommendation: adjust, do not replace

The mentors' question was whether to proceed, adjust or replace this dataset. I recommend **adjust**:

1. **Drop exact duplicate rows before splitting.** The key is every column of the raw row, `Time` included. This removes 1,081 rows (0.38% of the data), of which 19 are fraud (3.9% of the fraud rows).
2. **Use the stratified 60/20/20 split as the primary split**, made after the duplicates are dropped. Report a split by time as a sensitivity check.
3. **Set the fill-level grid per dataset.** For this dataset use 0.01, 0.02, 0.05 and 0.10, not the default grid that goes up to 1.0.
4. **Treat it as the stress case.** With 284 training positives, a result of no gain from synthetic data is expected and is a reportable result.
5. **Optionally add IEEE-CIS as a second fraud dataset.** It cannot be merged into this one. Details below.

## 1. Duplicates

| | Count |
|---|---|
| Extra exact copies, all rows | 1,081 (773 groups, 1,854 rows involved) |
| Extra exact copies, fraud rows | 19 (13 groups, 32 rows involved) |
| Transactions with identical features but both labels | 0 |
| Extra copies when `Time` is ignored (same V1 to V28 and Amount) | 9,144 |

- **Fraud copies look like data artifacts.** In every one of the 13 fraud groups the copies share a timestamp, so they are not repeated frauds.
- **Do not drop duplicates while ignoring `Time`.** The other 8,063 rows share all of V1 to V28 and Amount with a row at a different time. That is most likely a repeated identical transaction, which is real behaviour. A deduplication step that runs after `Time` has been dropped as a feature would delete them. The key must be the raw row, so the step has to see the raw table.
- **Correction to the issue text.** The 19 duplicates are across all 492 fraud rows. In the stratified training part, 7 of the 295 fraud rows are extra copies inside train.

### The stratified split leaks

With the original stratified split, many validation and test rows have an exact copy in train.

| Part | Rows with a copy in train | Distinct rows | Fraud rows |
|---|---|---|---|
| Validation | 297 | 266 | 4 (4.1% of its 98 fraud) |
| Test | 271 | 231 | 5 (5.1% of its 99 fraud) |

The 231 distinct test rows are the "about 230" on the 9 October slide. A model, and especially a GAN, can score well on these by memorising. The current rule in `architecture.md` section 5.1, "drop duplicates only inside the training split", does not fix this: all 568 rows stay leaked.

### Policies compared

| Policy (stratified split) | Fraud train / val / test | Val and test rows still seen in train | Fraud rows lost |
|---|---|---|---|
| None | 295 / 98 / 99 | 568 | 0 |
| Dedupe inside train only (current rule) | 288 / 98 / 99 | 568 | 7 |
| Drop val and test copies, dedupe inside parts | 288 / 93 / 93 | 0 | 18 |
| **Dedupe everything, then split (recommended)** | **284 / 94 / 95** | **0** | **19** |

Deduping first costs 19 fraud rows, one more than dropping only the validation and test copies (18). In return the split is made once, on data that is already clean, so the stratification holds exactly and there is no second pass to document.

## 2. Split by time

The data covers two days. Splitting the rows by `Time` (first 60% train, next 20% validation, last 20% test):

| | Stratified, after dedupe | By time, after dedupe |
|---|---|---|
| Fraud rows, train / val / test | 284 / 94 / 95 | 342 / 57 / 74 |
| Fraud rate, train / val / test | 0.17% / 0.17% / 0.17% | 0.20% / 0.10% / 0.13% |
| Time covered by val and test | the whole two days | 33 to 48 hours |
| Duplicate leakage | none, once deduped | none, even before deduping |
| Recall uncertainty (95% half-width at recall 0.8), validation | about 8 points | about 10 points |

**What a time split gives.** It is the realistic setting (train on the past, test on later transactions). Copies share a timestamp, so it has no duplicate leakage even before deduplication. The training part has more fraud rows (342 against 284).

**What it costs.**
- Validation has only 57 fraud rows, so one missed case moves recall by 1.75 points.
- The fraud rate is not stable over the day: 0.40% to 0.48% between midnight and 08:00, and about 0.13% in the rest of the day. Train contains both nights. Validation and test cover only 09:24 to 24:00 of the second day, so they never contain the high-fraud hours.
- With only two days of data, "the future" is a matter of hours. The result mixes time shift with time of day.

So the time split is worth reporting as a robustness check, but it should not be the primary split.

## 3. Fill levels for this dataset

The class gap after deduplication is 169,667 rows (284 positives). The default grid would add up to 597 times the real positives, from a generator trained on 284 examples.

| Fill level | Rows added | Times the real positives |
|---|---|---|
| 0.01 | 1,697 | 6 |
| 0.02 | 3,393 | 12 |
| 0.05 | 8,483 | 30 |
| 0.10 | 16,967 | 60 |
| 0.25 | 42,417 | 149 |
| 1.00 | 169,667 | 597 |

I propose 0.01, 0.02, 0.05 and 0.10 for this dataset. Larger levels cost more compute and are unlikely to be informative, since the generator has so few real examples to learn from.

## 4. Candidate datasets for more fraud rows

The issue asked for real, non-simulated datasets that could add fraud rows. The sources I could read do not name any public real dataset that uses the ULB features. The V1 to V28 columns are principal components of confidential variables, so components from any other dataset mean something different and rows cannot be pooled. A candidate can therefore only be a second dataset, not an addition to this one.

| Candidate | Real? | Size and fraud | Features | Licence and access | Verdict |
|---|---|---|---|---|---|
| ULB / Worldline (current) | Real, 2 days in Sept 2013 | 284,807 rows, 492 fraud (0.17%) | `Time`, `Amount`, V1 to V28 (PCA) | Database Contents License v1.0 per the benchmark below and a 2024 paper. A Kaggle metadata copy lists ODbL, so check the Kaggle page | Keep, with the adjustments above |
| IEEE-CIS (Vesta) | Real, donated by Vesta | 590,540 transactions, 20,663 fraud (3.5%) | About 390 to 430 masked columns in a transaction and an identity table, many missing values | Kaggle competition rules apply and must be accepted to download. I could not read the rules, so the terms of use are unconfirmed. The IEEE DataPort copy needs a subscription | Best candidate as a second dataset, not a merge |
| Feedzai Bank Account Fraud (NeurIPS 2022) | Synthetic, generated from an anonymized real dataset | Size and fraud rate not confirmed from the pages I read | Tabular, with built-in bias variants | Code is Apache-2.0. A 2025 paper says the data is CC BY-NC-ND 4.0 on Kaggle (secondary source) | Exclude: synthetic. Possible later as a labelled-synthetic comparison |
| Fraud ecommerce (Kaggle, vbinh002) | Provenance not documented | About 151,000 rows, 9.3% to 10.6% fraud depending on source | Device id, browser, IP, purchase value, sign-up time | No licence stated | Exclude until provenance is known |
| Credit Card Fraud Detection Dataset 2023 (Kaggle) | Unverified. I could not read the page | Third parties report 550,000+ rows. Class balance unverified | V1 to V28, Amount, Class | Not readable | Exclude: cannot certify as real |
| Sparkov simulated (Kaggle) | Simulated | 1.3 million rows, 5.7% fraud | Synthetic | CC0 1.0 | Exclude: simulated |

### If we add IEEE-CIS

- **Why it is attractive.** It has about 20,000 fraud rows, so a generator has thousands of real positives to learn from, unlike ULB. It has masked raw features, so it is not PCA-encoded. It also has real missing values, which makes it a second testbed for the data-repair step.
- **Why it is not free.**
  - Each team member must accept the Kaggle competition rules to download it. I could not confirm that our use is permitted, so someone should read the rules page first.
  - It is large (about 590,000 rows by 390 or more columns). The Fraud Dataset Benchmark keeps 67 of 393 features, which is a reasonable starting point.
  - It needs its own preprocessing, a time-aware split and a new set of fill levels.
- **Recommendation.** Raise it with the mentors on 16 October. Do not start it this sprint.

## 5. What this changes in the pipeline

These need team agreement before anyone builds them:

- **Where deduplication lives.** If duplicates are dropped before the split, it belongs in the orchestrator's split step, not in Pre-processing. The doc's principle that the split comes before any cleaning still holds, because exact-row deduplication on the raw table is not modelling-side cleaning, but it is a change to section 4. It needs a config option and a branch in `split_data`.
- **Split strategy.** The time split becomes an option in the split step (`split_strategy`), as the issue said.
- **Pre-processing rule.** Section 5.1's row "Exact duplicate rows: drop only inside the training split" should say that exact duplicates are removed before the split, keyed on the full raw row.
- **Row-ID files.** `splits/` now holds the recommended split: stratified, and by time, both after deduplication, plus the list of dropped rows. See `splits/README.md`.

## Questions for the mentors

1. Is ULB acceptable as the extreme-imbalance stress case, with a no-gain result reported as a result?
2. Is a second fraud dataset such as IEEE-CIS wanted, given the compute and the Kaggle terms?
3. Do they want the time split reported as a sensitivity check?

## Sources

- [Fraud Dataset Benchmark (Amazon Science)](https://github.com/amazon-science/fraud-dataset-benchmark), the catalogue of public fraud datasets with their sources and licences, and its [paper](https://arxiv.org/pdf/2208.14417)
- [IEEE-CIS Fraud Detection on IEEE DataPort](https://ieee-dataport.org/documents/ieee-cis-fraud-detection), and the [Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection)
- [Bank Account Fraud (Feedzai)](https://github.com/feedzai/bank-account-fraud) and the [NeurIPS 2022 paper](https://arxiv.org/abs/2211.13358)
- [ULB credit card fraud dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Dataset descriptions marked unverified come from secondary sources, because Kaggle pages could not be read from here. Licence terms should be confirmed on the Kaggle pages before any download.
