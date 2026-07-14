"""Record the status of the isolated incremental-ingestion evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(*, output: Path) -> dict:
    result = {
        "status": "NOT_MEASURED",
        "reason": (
            "T09 requires add/update/move/delete checks in an isolated corpus with fixed "
            "ground truth. This run does not alter the user's production knowledge base."
        ),
        "required_scenarios": ["add", "content_update", "move", "delete"],
        "implementation_test_evidence": {
            "status": "REQUIRES_SEPARATE_TEST_RUN",
            "note": (
                "Existing unit tests cover scanner and versioning behavior "
                "but are not a T09 measured result."
            ),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/results/incremental_results.json")
    args = parser.parse_args()
    print(json.dumps(run(output=Path(args.output)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
