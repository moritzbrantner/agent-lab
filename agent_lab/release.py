"""Fixed-hardware release reporting; missing evidence never becomes a pass."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from jsonschema import Draft202012Validator

from agent_lab.experiments import MEASUREMENTS, ROOT, canonical, digest, validate_result
from agent_lab.runtime import AgentConfig
from agent_lab.trace import load_trace


def validate_profile(profile):
    schema = json.loads((ROOT / "schemas/release-profile-v1.json").read_text())
    Draft202012Validator(schema).validate(profile)
    canonical(profile)
    workloads = profile["workloads"]
    configuration = profile["configuration"]
    if asdict(AgentConfig(**configuration)) != configuration:
        raise ValueError("Release configuration must declare every runtime field")
    variable = {
        "seed",
        "context_size",
        "max_output_tokens",
        "max_steps",
        "max_tool_calls",
        "options",
    }
    for workload in workloads:
        candidate = workload["configuration"]
        if (
            asdict(AgentConfig(**candidate)) != candidate
            or any(
                candidate[key] != value
                for key, value in configuration.items()
                if key not in variable
            )
            or any(
                candidate["options"].get(key) != value
                for key, value in configuration["options"].items()
            )
        ):
            raise ValueError(
                "One release candidate cannot mix model or backend identities"
            )
    ids = [w["task"]["id"] for w in workloads]
    thresholds = profile["thresholds"]
    if len(ids) != len(set(ids)) or len({t["id"] for t in thresholds}) != len(
        thresholds
    ):
        raise ValueError("Release identities must be unique")
    for threshold in thresholds:
        if threshold["source"] == "runs":
            if not set(threshold["workloads"]).issubset(ids):
                raise ValueError("Unknown release workload")
            if threshold["aggregation"] == "success_rate":
                if threshold["field"] != "correctness.status":
                    raise ValueError("Success rate requires independent correctness")
            elif threshold["field"] not in (
                {"measurements." + name for name in MEASUREMENTS}
                | {"configuration.context_size"}
            ):
                raise ValueError("Unknown release measurement")
        elif threshold["aggregation"] != "value":
            raise ValueError("Proof thresholds use explicit values")
    return profile


def family(configuration):
    return {k: v for k, v in configuration.items() if k != "seed"}


def status_for(value, threshold):
    if value is None:
        return "unavailable"
    comparison = (
        value >= threshold["limit"]
        if threshold["operator"] == ">="
        else value <= threshold["limit"]
    )
    return "pass" if comparison else "fail"


def combined(statuses):
    if "fail" in statuses:
        return "fail"
    if not statuses or "unavailable" in statuses:
        return "unavailable"
    return "pass"


def release_report(profile, runs, proofs, software):
    validate_profile(profile)
    if (
        proofs.get("profile_sha256") != digest(profile)
        or proofs.get("hardware") != profile["hardware"]
        or proofs.get("software") != software
        or proofs.get("configuration") != profile["configuration"]
    ):
        raise ValueError("Scaffold proof authority drift")
    expected = {w["task"]["id"]: w for w in profile["workloads"]}
    groups = {identifier: [] for identifier in expected}
    repeats, identifiers = set(), set()
    for sample in runs:
        validate_result(sample)
        identifier = sample["task"]["id"]
        workload = expected.get(identifier)
        if workload is None:
            raise ValueError("Undeclared release workload")
        if (
            sample["task"] != workload["task"]
            or {k: v for k, v in sample["protocol"].items() if k != "repeat"}
            != workload["protocol"]
            or family(sample["configuration"]) != family(workload["configuration"])
            or sample["hardware"] != profile["hardware"]
            or sample["software"] != software
            or sample["correctness"]["evaluator"] != workload["protocol"]["id"]
        ):
            raise ValueError("Release workload, configuration or environment drift")
        repeat = (identifier, sample["protocol"]["repeat"])
        if repeat in repeats or sample["run_id"] in identifiers:
            raise ValueError("Duplicate release sample")
        repeats.add(repeat)
        identifiers.add(sample["run_id"])
        groups[identifier].append(sample)
    entries = []
    for threshold in profile["thresholds"]:
        details = []
        if threshold["source"] == "proofs":
            proof = proofs["values"].get(threshold["field"], {})
            value = proof.get("value")
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value)
            ):
                raise ValueError("Proof value must be finite numeric or unavailable")
            details.append(
                {
                    "value": value,
                    "status": status_for(value, threshold),
                    "scope": proof.get("scope", "No matching proof"),
                    "evidence": proof.get("evidence", []),
                }
            )
        else:
            for identifier in threshold["workloads"]:
                samples = groups[identifier]
                aggregation = threshold["aggregation"]
                values = []
                if aggregation == "success_rate":
                    values = [
                        float(
                            s["correctness"]["status"] == "pass"
                            and s["work"]["tool_calls"] is not None
                            and s["work"]["tool_calls"]
                            >= threshold.get("minimum_tool_calls", 0)
                        )
                        for s in samples
                    ]
                else:
                    section, name = threshold["field"].split(".", 1)
                    values = [s[section].get(name) for s in samples]
                value = None
                if len(samples) >= profile["minimum_repeats"] and all(
                    v is not None for v in values
                ):
                    if aggregation == "success_rate":
                        value = sum(values) / len(values)
                    elif aggregation == "p95":
                        value = sorted(values)[math.ceil(0.95 * len(values)) - 1]
                    elif aggregation == "max":
                        value = max(values)
                    elif aggregation == "min":
                        value = min(values)
                details.append(
                    {
                        "workload": identifier,
                        "samples": len(samples),
                        "available_samples": sum(v is not None for v in values),
                        "value": value,
                        "status": status_for(value, threshold),
                    }
                )
        entries.append(
            {
                **threshold,
                "status": combined([d["status"] for d in details]),
                "details": details,
            }
        )
    required = [entry["status"] for entry in entries if entry["required"]]
    status = combined(required)
    return {
        "schema_version": 1,
        "tier": profile["id"],
        "profile_sha256": digest(profile),
        "configuration": profile["configuration"],
        "hardware": profile["hardware"],
        "software": software,
        "samples": len(runs),
        "status": status,
        "ready": status == "pass",
        "thresholds": entries,
        "scope": profile["scope"],
    }


def load_bundle(profile, root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["profile_sha256"] != digest(profile):
        raise ValueError("Release bundle/profile drift")
    required = {
        "results.jsonl",
        "cold-results.jsonl",
        "proofs.json",
        "profile.json",
        "measured-source.json",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("Release manifest is incomplete")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Contained release artifact required")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Release artifact drift: " + name)
    if (
        json.loads((root / "profile.json").read_text()) != profile
        or digest(json.loads((root / "measured-source.json").read_text()))
        != manifest["software"]["source_hash"]
    ):
        raise ValueError("Measured source/profile drift")
    runs = [
        json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()
    ]
    cold = [
        json.loads(line)
        for line in (root / "cold-results.jsonl").read_text().splitlines()
    ]
    if len(cold) != len(profile["workloads"]) or any(
        s["protocol"]["repeat"] != profile["minimum_repeats"]
        or s["configuration"]["seed"] != 99
        for s in cold
    ):
        raise ValueError("Cold-load cohort is incomplete or misclassified")
    if any(s["protocol"]["repeat"] >= profile["minimum_repeats"] for s in runs):
        raise ValueError("Cold/extra samples cannot enter the fixed warm cohort")
    for sample in [*runs, *cold]:
        for name in ("result.json", "trace.json"):
            if f"runs/{sample['run_id']}/{name}" not in manifest["files"]:
                raise ValueError("Unbound release result/trace")
        run_root = root / "runs" / sample["run_id"]
        trace = load_trace(run_root / "trace.json", artifact_root=run_root)
        if (
            json.loads((run_root / "result.json").read_text()) != sample
            or trace["run_id"] != sample["run_id"]
            or trace["task_id"] != sample["task"]["id"]
            or trace["configuration_id"] != digest(sample["configuration"])
            or trace["artifacts"] != sample["artifacts"]
        ):
            raise ValueError("Release result/trace drift")
        evaluations = [e for e in trace["events"] if e["kind"] == "evaluator"]
        measurements = [e for e in trace["events"] if e["kind"] == "measurements"]
        if (
            not evaluations
            or evaluations[-1]["metadata"] != sample["correctness"]
            or not measurements
            or measurements[-1]["metadata"]
            != {
                "values": sample["measurements"],
                "sources": sample["measurement_sources"],
            }
        ):
            raise ValueError("Release authority/measurement trace drift")
        if sample["task"]["id"] != "repository-unknown-usage" and sample["correctness"][
            "status"
        ] in ("pass", "fail"):
            from agent_lab.benchmarks import load_task
            from agent_lab.evaluators import independent_evaluate

            task = load_task(sample["protocol"]["split"], sample["task"]["id"])
            output = json.loads((run_root / "output.json").read_text())["content"]
            if (
                digest(task) != sample["task"]["fixture_hash"]
                or independent_evaluate(task, output) != sample["correctness"]
            ):
                raise ValueError("Independent release evaluation drift")
        elif sample["task"]["id"] == "repository-unknown-usage":
            verification = json.loads((run_root / "verification.json").read_text())
            if {
                k: verification["after"][k] for k in ("status", "score", "evaluator")
            } != (sample["correctness"]):
                raise ValueError("Protected repository evaluation drift")
    proofs = json.loads((root / "proofs.json").read_text())
    release_report(profile, cold, proofs, manifest["software"])
    return release_report(profile, runs, proofs, manifest["software"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    report = load_bundle(profile, args.bundle)
    print(canonical(report))
    return {"pass": 0, "unavailable": 1, "fail": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
