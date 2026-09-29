"""Isolated stdlib-only evaluator entry point; never executes candidate code."""

import hashlib
import json
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    if not path.is_relative_to(root):
        raise ValueError("Evaluator task outside authority")
    task = json.loads(path.read_text())
    encoded = json.dumps(task, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(encoded.encode()).hexdigest() != sys.argv[2]:
        raise ValueError("Evaluator fixture drift")
    candidate = json.load(sys.stdin)
    if not isinstance(candidate, dict) or not isinstance(candidate.get("output"), str):
        raise ValueError("Invalid candidate output")
    passed = candidate["output"] == task["expected"]
    print(
        json.dumps(
            {
                "status": "pass" if passed else "fail",
                "score": int(passed),
                "evaluator": task["evaluator"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
