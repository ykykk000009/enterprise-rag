# T09 Evaluation

This directory contains the reproducible evaluation harness requested by
`T09_RAG_MVP_Evaluation_for_Codex.md`. The scripts call the running API for
retrieval and answer tests; they do not bypass the production RAG path.

## Official Dataset Rule

Only JSONL cases with `"validated": true` are included in official metrics.
Each such case must have a human-checked question, relevant source path,
evidence span, expected answer, and citation expectation. Automatically derived
cases remain diagnostic candidates and must never be counted as T09 ground truth.

## Build The Current-Corpus Candidate Set

The builder creates 100 source-location cases from the currently selected
knowledge base. It may use several independently verified chunks from the same
document when the corpus has fewer than 100 files. Every case stores the active
document ID, chunk ID, path, page/section and evidence span. Generated cases are
kept as `validated: false` until a human reviews every question.

```powershell
python evaluation\build_gold_benchmark.py `
  --knowledge-base-id <knowledge-base-id>
```

Run the source-verified diagnostic retrieval benchmark, including Top-15:

```powershell
python evaluation\run_retrieval_eval.py `
  --knowledge-base-id <knowledge-base-id> `
  --include-unvalidated `
  --summary-output evaluation\results\retrieval_summary.json
```

These diagnostic scores must not be presented as an official T09 Gold score.
After human review, change only genuinely reviewed cases to `validated: true`
and rerun without `--include-unvalidated`.

## Run

From the repository root:

```powershell
python evaluation\run_all.py `
  --knowledge-base-id <knowledge-base-id>
```

Results are written to `evaluation/results/`, and the readable conclusion is
written to `evaluation/report.md`.

## Optional One-Hour Soak

```powershell
python evaluation\run_stability_eval.py `
  --knowledge-base-id <knowledge-base-id> --execute --duration-seconds 3600
```

The stability output remains diagnostic until the test dataset is manually
validated and resource monitoring is supplied.
