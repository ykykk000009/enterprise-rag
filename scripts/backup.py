"""Create a local snapshot of the SQLite metadata and Qdrant Local directory."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up a local document RAG index.")
    parser.add_argument("--database", default="data/agent.db")
    parser.add_argument("--qdrant", default="data/qdrant")
    parser.add_argument("--output", default="backups")
    args = parser.parse_args()

    database = Path(args.database)
    qdrant = Path(args.qdrant)
    output_root = Path(args.output) / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root.mkdir(parents=True, exist_ok=False)

    with sqlite3.connect(database) as source, sqlite3.connect(output_root / "agent.db") as target:
        source.backup(target)

    if qdrant.exists():
        shutil.copytree(qdrant, output_root / "qdrant")

    print(output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
