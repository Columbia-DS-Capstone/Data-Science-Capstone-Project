# Credit Card Fraud analysis (issue #24)

`fraud_analysis.py` looks at duplicates and at a split by time in the ULB credit card fraud data. The conclusions are in `docs/fraud_dataset_recommendation.md`.

It works offline from the raw CSV (Kaggle `mlg-ulb/creditcardfraud`). Nothing is downloaded.

```bash
uv run python eda/fraud/fraud_analysis.py --csv eda/output/data/creditcard.csv
```

Add `--write-splits splits` to also write the recommended row-ID files (see `splits/README.md`). Tables print to the terminal and are saved as CSV in `eda/output/fraud/` (`--out` changes the folder).

| Table | Question it answers |
|---|---|
| Duplicates | How many rows repeat, overall and among fraud, under three meanings of "the same" |
| Same features, both labels | Is any transaction labelled both ways |
| Fraud rate by time of day | Does the fraud rate change over the day |
| Parts, per split | Rows, fraud, fraud rate and time range of each part, and how noisy recall is on it |
| Duplicates and leakage | Copies inside each part, and validation or test rows that also sit in train |
| Duplicate policies | What each way of handling duplicates leaves in each part |
| Fill grid after dedupe | Rows a fill level would add, and as a multiple of the real positives |

Two details that matter when reusing this:

- Duplicates inside a part are found within that part. Comparing across parts makes a policy look cheaper than it is.
- "Duplicate" means an exact copy of the whole row, `Time` included. Rows that share only V1 to V28 and Amount are repeated transactions, not copies.
