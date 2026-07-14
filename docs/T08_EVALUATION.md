# T08 Evaluation

Run the baseline benchmark against an indexed knowledge base:

```powershell
E:\findfileagent\.venvs\findfileagent\Scripts\python.exe scripts\benchmark.py `
  --knowledge-base-id <knowledge-base-id>
```

The report records per-case outcomes, P50/P95 latency, and peak process RSS.

## 8 GB development profile

Use CPU embedding, one ingestion worker, batch size 8, 150 DPI OCR, and the default
500/60 token chunk configuration. The benchmark reports peak RSS and fails evaluation
when a known answer is not cited or an unknown answer is not refused.

## 10 GB source method

Authorize one bounded source root, then run an initial scan with the service left open.
The worker processes one job at a time and records progress in SQLite. Record the source
file count, total bytes, elapsed time, P50/P95 query latency, and peak RSS. Restart the
service during a queued job to verify that the worker resumes without duplicate chunks.

Do not infer memory use from source-directory size: the service parses one file at a time
and stores metadata, FTS rows, and vectors on disk.

## Known limitations

- OCR runs only for low-text PDF pages and does not process DOCX images or embedded scans.
- Scanned documents with unreadable pages remain failed with an explicit error.
- The baseline uses a deterministic hash embedding provider; model evaluation replaces it
  only after this report is captured.
