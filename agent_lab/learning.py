"""Learning entry gate and verified synthetic examples; no ML imports required."""

import copy
import hashlib
import json
import random
from fractions import Fraction

from agent_lab.archive import Archive
from agent_lab.baselines import verify_bundle
from agent_lab.benchmarks import discover, load_task, prompt_for
from agent_lab.distillation import verify_artifact
from agent_lab.evaluators import independent_evaluate
from agent_lab.experiments import ROOT, canonical, digest


def entry_gate():
    baseline = verify_bundle(ROOT / "evidence/baselines/local-v1")
    archive_root = ROOT / "evidence/archive/local-v1"
    report = json.loads((archive_root / "report.json").read_text())
    if (
        Archive(archive_root).read(report["candidate"])["decision"]["status"]
        != "reject"
    ):
        raise ValueError("Regression protection evidence unavailable")
    tasks = [*discover("development"), *discover("held-out")]
    for task in tasks:
        if (
            independent_evaluate(task, task["expected"])["status"] != "pass"
            or independent_evaluate(task, "invalid-gate-output")["status"] != "fail"
        ):
            raise ValueError("Independent evaluator gate failed")
    protected = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            ROOT / "agent_lab/evaluator_worker.py",
            ROOT / "agent_lab/evaluators.py",
            ROOT / "agent_lab/improvement.py",
            ROOT / "agent_lab/archive.py",
        ]
    }
    return {
        "schema_version": 1,
        "passed": True,
        "baseline_samples": baseline["samples"],
        "task_hashes": {t["id"]: digest(t) for t in tasks},
        "protected_sources": protected,
        "held_out_tasks": [t["id"] for t in tasks if t["split"] == "held-out"],
        "held_out_input_hashes": [
            digest(t["input"]) for t in tasks if t["split"] == "held-out"
        ],
    }


def build_data(artifact, *, count=32, seed=17):
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("Training data requires 1..64 examples")
    teacher = verify_artifact(artifact)
    template = load_task("development", "sum")
    forbidden = {digest(t["input"]) for t in discover("held-out")}
    randomizer = random.Random(seed)
    examples, seen = [], set()
    for attempt in range(1000):
        numbers = (
            template["input"]["numbers"]
            if attempt == 0
            else [randomizer.randint(-9, 9) for _ in range(3)]
        )
        inputs = {"numbers": numbers}
        if digest(inputs) in seen | forbidden:
            continue
        seen.add(digest(inputs))
        output = teacher.invoke({"batches": [[n] for n in numbers]})["output"]
        if output != str(int(sum((Fraction(n) for n in numbers), Fraction(0)))):
            raise ValueError("Synthetic teacher disagrees with independent arithmetic")
        task = copy.deepcopy(template)
        task["input"] = inputs
        examples.append(
            {
                "input": inputs,
                "prompt": prompt_for(task),
                "response": canonical({"content": output, "calls": []}),
                "source_runs": [p["run_id"] for p in artifact["provenance"]],
            }
        )
        if len(examples) == count:
            break
    if len(examples) != count:
        raise ValueError("Could not generate bounded unique training examples")
    value = {
        "schema_version": 1,
        "recipe": "verified-batch-tool-to-sum-response-v1",
        "seed": seed,
        "source_artifact_sha256": artifact["artifact_sha256"],
        "examples": examples,
    }
    value["data_sha256"] = digest(value)
    validate_data(value, artifact=artifact, held_out_hashes=forbidden)
    return value


def validate_data(data, *, artifact=None, held_out_hashes=()):
    if artifact is None:
        artifact = json.loads(
            (
                ROOT / "evidence/distillation/local-v1/distilled-artifact.json"
            ).read_text()
        )
    verify_artifact(artifact)
    if (
        data["schema_version"] != 1
        or data["source_artifact_sha256"] != artifact["artifact_sha256"]
        or digest({k: v for k, v in data.items() if k != "data_sha256"})
        != data["data_sha256"]
    ):
        raise ValueError("Training data provenance/content drift")
    forbidden = set(held_out_hashes) | {
        digest(t["input"]) for t in discover("held-out")
    }
    template = load_task("development", "sum")
    sources = [p["run_id"] for p in artifact["provenance"]]
    seen = set()
    if not 1 <= len(data["examples"]) <= 64:
        raise ValueError("Invalid training example budget")
    for example in data["examples"]:
        inputs = example["input"]
        numbers = inputs.get("numbers")
        if (
            set(inputs) != {"numbers"}
            or not isinstance(numbers, list)
            or not 1 <= len(numbers) <= 16
            or any(type(n) is not int for n in numbers)
        ):
            raise ValueError("Invalid training input")
        identity = digest(inputs)
        if identity in forbidden | seen:
            raise ValueError("Held-out or duplicated training material")
        seen.add(identity)
        task = copy.deepcopy(template)
        task["input"] = inputs
        response = json.loads(example["response"])
        expected = str(int(sum((Fraction(n) for n in numbers), Fraction(0))))
        if (
            response != {"content": expected, "calls": []}
            or example["prompt"] != prompt_for(task)
            or example["source_runs"] != sources
        ):
            raise ValueError("Synthetic example failed independent verification")
    return data
