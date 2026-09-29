"""Derive trusted reusable artifacts from independently rechecked experience."""

import hashlib
import json
from pathlib import Path

from agent_lab.benchmarks import load_task
from agent_lab.capabilities import CapabilityRunner, discover_capabilities
from agent_lab.evaluators import independent_evaluate
from agent_lab.experiments import ROOT, digest, validate_result
from agent_lab.trace import load_trace


def _source(path, source_root):
    root, path = Path(source_root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source outside declared evidence root")
    result = validate_result(json.loads((path / "result.json").read_text()))
    if (
        result["protocol"]["split"] != "development"
        or result["task"]["id"] != "batch-sum"
        or result["correctness"]["status"] != "pass"
    ):
        raise ValueError(
            "Distillation requires successful development batch experience"
        )
    if (
        result["configuration"]["backend"] == "fixture"
        or not result["work"]["generated_tokens"]
        or not result["work"]["model_calls"]
    ):
        raise ValueError("Source must be measured model experience")
    trace = load_trace(path / "trace.json", artifact_root=path)
    if (
        trace["run_id"] != result["run_id"]
        or trace["configuration_id"] != digest(result["configuration"])
        or trace["artifacts"] != result["artifacts"]
    ):
        raise ValueError("Source trace identity drift")
    task = load_task("development", "batch-sum")
    output = json.loads((path / "output.json").read_text())["content"]
    if (
        result["task"]["fixture_hash"] != digest(task)
        or independent_evaluate(task, output)["status"] != "pass"
    ):
        raise ValueError("Source no longer independently verifies")
    proof = {
        "run_id": result["run_id"],
        "path": path.relative_to(root).as_posix(),
        "result_sha256": hashlib.sha256(
            (path / "result.json").read_bytes()
        ).hexdigest(),
        "trace_sha256": hashlib.sha256((path / "trace.json").read_bytes()).hexdigest(),
        "fixture_sha256": digest(task),
        "output_sha256": digest(output),
        "configuration": result["configuration"],
    }
    return proof, {
        "input": task["input"],
        "output": output,
        "source_run_id": result["run_id"],
    }


def distill(sources, *, source_root=ROOT):
    if not sources or len(sources) > 32:
        raise ValueError("Select 1..32 verified source runs")
    records = [_source(path, source_root) for path in sorted(sources)]
    proofs, examples = zip(*records, strict=True)
    if len({p["run_id"] for p in proofs}) != len(proofs):
        raise ValueError("Duplicate source experience")
    capability = next(c for c in discover_capabilities() if c.id == "batch-sum-v1")
    for example in examples:
        if capability.invoke(example["input"])["output"] != example["output"]:
            raise ValueError("Distilled procedure differs from verified experience")
    artifact = {
        "schema_version": 1,
        "kind": "deterministic-tool-and-verified-examples",
        "recipe": "batch-sum-experience-v1",
        "capability_id": capability.id,
        "capability_sha256": capability.identity,
        "provenance": list(proofs),
        "examples": list(examples),
        "input_schema": capability.manifest["input_schema"],
        "procedure": "Sum integer batches, total them, return decimal text.",
    }
    return {**artifact, "artifact_sha256": digest(artifact)}


def verify_artifact(artifact, *, source_root=ROOT):
    body = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    if digest(body) != artifact["artifact_sha256"]:
        raise ValueError("Distilled artifact drift")
    paths = [(Path(source_root) / p["path"]).resolve() for p in artifact["provenance"]]
    rebuilt = distill(paths, source_root=source_root)
    if rebuilt != artifact:
        raise ValueError("Distilled provenance or implementation drift")
    return next(c for c in discover_capabilities() if c.id == artifact["capability_id"])


class DistilledRunner(CapabilityRunner):
    def __init__(self, artifact, inputs, *, source_root=ROOT, cache=None):
        self.artifact = artifact
        super().__init__(
            verify_artifact(artifact, source_root=source_root),
            inputs,
            cache=cache,
            source_runs=[p["run_id"] for p in artifact["provenance"]],
        )

    async def __call__(self, adapter, tools, config, state, emit):
        if config.options.get("distilled_artifact") != self.artifact["artifact_sha256"]:
            raise ValueError("Distilled execution identity drift")
        emit(
            "authority",
            {
                "scope": "verified-distilled-tool",
                "artifact_sha256": self.artifact["artifact_sha256"],
            },
        )
        return await super().__call__(adapter, tools, config, state, emit)
