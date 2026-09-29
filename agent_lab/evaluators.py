"""Trusted evaluation outside model/tool authority."""

import json
import os
import subprocess
import sys

from agent_lab.experiments import ROOT, canonical, digest


def independent_evaluate(task, output):
    # The child independently verifies the trusted fixture identity.
    from agent_lab.benchmarks import load_task

    expected = load_task(task["split"], task["id"])
    if expected != task:
        raise ValueError("Candidate cannot supply evaluator expectations")
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "agent_lab/evaluator_worker.py"),
            str(ROOT / "benchmarks" / task["split"] / (task["id"] + ".json")),
            digest(task),
        ],
        input=canonical({"output": output}),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env={"PATH": os.defpath, "LC_ALL": "C.UTF-8", "TZ": "UTC"},
    )
    if process.returncode:
        raise ValueError("Independent evaluator failed")
    return json.loads(process.stdout)
