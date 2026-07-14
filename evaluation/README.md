# T09 Evaluation

This directory contains the reproducible evaluation harness requested by
`T09_RAG_MVP_Evaluation_for_Codex.md`. The scripts call the running API for
retrieval and answer tests; they do not bypass the production RAG path.

## Official Dataset Rule

Only JSONL cases with `"validated": true` are included in official metrics.
Each such case must have a human-checked question, relevant source path,
evidence span, expected answer, and citation expectation. Automatically derived
cases remain diagnostic candidates and must never be counted as T09 ground truth.

## Build The Gold Set

The source-location Gold Benchmark reflects the product's primary workflow:
find the file paths containing supplied fields. It uses 100 cases from distinct
real documents with source paths and evidence saved for every case.

```powershell
E:\findfileagent\.venvs\findfileagent\Scripts\python.exe evaluation\build_gold_benchmark.py `
  --knowledge-base-id <knowledge-base-id>
```

## Run

From `E:\findfileagent\codex_agent_mvp`:

```powershell
E:\findfileagent\.venvs\findfileagent\Scripts\python.exe evaluation\run_all.py `
  --knowledge-base-id <knowledge-base-id>
```

Results are written to `evaluation/results/`, and the readable conclusion is
written to `evaluation/report.md`.

## Optional One-Hour Soak

```powershell
E:\findfileagent\.venvs\findfileagent\Scripts\python.exe evaluation\run_stability_eval.py `
  --knowledge-base-id <knowledge-base-id> --execute --duration-seconds 3600
```

The stability output remains diagnostic until the test dataset is manually
validated and resource monitoring is supplied.
