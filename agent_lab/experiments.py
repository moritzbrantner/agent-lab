"""Versioned experiment validation and capability/resource comparisons."""

import hashlib
import json
import statistics
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = (
    "wall_seconds",
    "inference_seconds",
    "tool_seconds",
    "setup_seconds",
    "cpu_seconds",
    "gpu_seconds",
    "peak_ram_bytes",
    "peak_vram_bytes",
    "cpu_utilization",
    "gpu_utilization",
    "energy_joules",
)
WORK = (
    "input_tokens",
    "output_tokens",
    "generated_tokens",
    "model_calls",
    "retries",
    "tool_calls",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_result(value):
    schema = json.loads((ROOT / "schemas/result-v1.json").read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=str)
    if errors:
        raise ValueError("; ".join(error.message for error in errors))
    canonical(value)  # Reject non-finite numbers even when the validator accepts them.
    status = value["correctness"]["status"]
    score = value["correctness"]["score"]
    if status == "pass" and score != 1:
        raise ValueError("A passing result requires full credit")
    if status == "fail" and (score is None or score == 1):
        raise ValueError("A failed result requires less than full credit")
    if status in ("error", "cancelled", "unavailable") and score is not None:
        raise ValueError("Unevaluated results cannot claim correctness")
    for name, measurement in value["measurements"].items():
        if measurement is not None and not value["measurement_sources"].get(name):
            raise ValueError(f"Measurement source required: {name}")
    return value


def compare(baseline, candidate):
    for value in (baseline, candidate):
        validate_result(value)
    for field in ("task", "protocol", "hardware", "software"):
        if baseline[field] != candidate[field]:
            raise ValueError(f"Incompatible {field}; stratify before comparing")
    changes = sorted(
        key
        for key in baseline["configuration"].keys() | candidate["configuration"].keys()
        if baseline["configuration"].get(key) != candidate["configuration"].get(key)
    )
    result = {"configuration_changes": changes}
    for metric, group in (
        ("generated_tokens", "work"),
        ("wall_seconds", "measurements"),
        ("peak_ram_bytes", "measurements"),
    ):
        costs = baseline[group][metric], candidate[group][metric]
        scores = baseline["correctness"]["score"], candidate["correctness"]["score"]
        if None in costs or None in scores:
            outcome = "unavailable"
        elif (
            scores[1] >= scores[0]
            and costs[1] <= costs[0]
            and (scores[1] > scores[0] or costs[1] < costs[0])
        ):
            outcome = "candidate_dominates"
        elif (
            scores[0] >= scores[1]
            and costs[0] <= costs[1]
            and (scores[0] > scores[1] or costs[0] < costs[1])
        ):
            outcome = "baseline_dominates"
        elif scores[0] == scores[1] and costs[0] == costs[1]:
            outcome = "equivalent"
        else:
            outcome = "tradeoff"
        result[metric] = outcome
    return result


def summarize(runs):
    if not runs:
        raise ValueError("At least one sample required")
    repeats = set()
    for run in runs:
        validate_result(run)
        for key in ("task", "hardware", "software"):
            if run[key] != runs[0][key]:
                raise ValueError(f"Mixed {key}")
        configuration = {
            key: value for key, value in run["configuration"].items() if key != "seed"
        }
        first_configuration = {
            key: value
            for key, value in runs[0]["configuration"].items()
            if key != "seed"
        }
        if configuration != first_configuration:
            raise ValueError("Mixed configuration")
        protocol = {k: v for k, v in run["protocol"].items() if k != "repeat"}
        first = {k: v for k, v in runs[0]["protocol"].items() if k != "repeat"}
        if protocol != first or run["protocol"]["repeat"] in repeats:
            raise ValueError("Mixed protocol or duplicate repeat")
        repeats.add(run["protocol"]["repeat"])
    result = {"sampling_seeds": [run["configuration"].get("seed") for run in runs]}
    for key in (*WORK, *MEASUREMENTS, "score"):
        group = (
            "work"
            if key in WORK
            else "correctness"
            if key == "score"
            else "measurements"
        )
        values = [run[group][key] for run in runs if run[group][key] is not None]
        result[key] = {
            "samples": len(values),
            "missing": len(runs) - len(values),
            "mean": statistics.mean(values) if values else None,
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
        }
    return result
