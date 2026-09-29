"""Real historical fixture -> scoped agent patch -> independent executable evidence."""

import argparse
import asyncio
import difflib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from agent_lab.backends import OllamaAdapter
from agent_lab.baselines import inventory, unload_experiment_models, validate_manifest
from agent_lab.experiments import (
    ROOT,
    canonical,
    compare,
    digest,
    summarize,
    validate_result,
)
from agent_lab.repository_workload import (
    ADDITION,
    FIXTURE,
    NEEDLE,
    RepositoryTools,
    check_patch,
    fixture_manifest,
    resolve_conventions,
)
from agent_lab.runtime import (
    AgentConfig,
    AgentState,
    Reply,
    SequenceAdapter,
    ToolCall,
    run,
)
from agent_lab.telemetry import (
    EnergyCounter,
    Telemetry,
    hardware_profile,
    software_identity,
)
from agent_lab.trace import Trace, atomic_json


def reference_adapter():
    return SequenceAdapter(
        [
            Reply(
                calls=(ToolCall("read_source", {"path": "runtime.py"}),),
                input_tokens=0,
                output_tokens=0,
            ),
            Reply(
                calls=(
                    ToolCall("replace_text", {"old": NEEDLE, "new": NEEDLE + ADDITION}),
                ),
                input_tokens=0,
                output_tokens=0,
            ),
            Reply(
                calls=(ToolCall("check_patch", {}),), input_tokens=0, output_tokens=0
            ),
            Reply("patch ready", input_tokens=0, output_tokens=0),
        ]
    )


async def repository_run(
    adapter, config, output, *, repeat=0, hardware=None, software=None, conventions=None
):
    manifest = fixture_manifest()
    hardware, software = hardware or hardware_profile(), software or software_identity()
    conventions = conventions or resolve_conventions(FIXTURE)
    config = replace(
        config,
        options={
            **config.options,
            "repository_fixture": digest(manifest),
            "convention_sourceRevision": conventions["sourceRevision"],
            "permission_profile": "historical-scoped-repair-v1",
        },
    )
    run_id = digest(
        {
            "task": manifest,
            "configuration": asdict(config),
            "repeat": repeat,
            "hardware": hardware,
            "software": software,
        }
    )[:24]
    root, workspace = output / run_id, output / run_id / "workspace"
    if root.exists():
        raise ValueError("Repository run already exists")
    shutil.copytree(FIXTURE, workspace)
    active = resolve_conventions(workspace)
    if (
        active["sourceRevision"] != conventions["sourceRevision"]
        or active["files"] != conventions["files"]
    ):
        raise ValueError("Live policy drift during workload")
    atomic_json(root / "conventions.json", active)
    trace = Trace(run_id, manifest["id"], digest(asdict(config)), retain_content=True)
    meter = Telemetry(energy=EnergyCounter.discover())

    def emit(kind, payload):
        trace.emit(kind, payload)
        meter.emit(kind, payload)

    tools = RepositoryTools(workspace, emit)
    atomic_json(root / "permissions.json", tools.policy.document())
    prompt = (
        "You are repairing a real historical Python runtime. Follow AGENTS.md:\n"
        + active["local_instructions"]["AGENTS.md"]
        + "\nShared policy was resolved/read by the controller at "
        + active["sourceRevision"]
        + ". Return JSON replies with content and calls. Available tools: "
        "read_source(path: runtime.py/AGENTS.md/pyproject.toml), "
        "replace_text(old: existing text, new: replacement) edits runtime.py "
        "exactly once, check_patch({}) runs protected executable checks. "
        "Unknown token work must be None when an adapter request raises Exception "
        "or asyncio.CancelledError. Preserve known work from successful responses "
        "and tool failures. Read runtime.py, make the narrow fix, check and finish."
    )
    state = AgentState(prompt)
    agent_error = None
    meter.start()
    try:
        before = await check_patch(workspace, allow_baseline=True)
        if before["status"] != "fail":
            raise ValueError("Historical negative control no longer fails")
        try:
            await run(adapter, tools.registry(), config, state, emit)
        except Exception as error:
            agent_error = type(error).__name__
        evaluated = await check_patch(workspace)
    finally:
        measurements, sources = meter.stop()
    correctness = {k: evaluated[k] for k in ("status", "score", "evaluator")}
    trace.emit("evaluator", correctness)
    original, candidate = (
        (FIXTURE / "runtime.py").read_text(),
        (workspace / "runtime.py").read_text(),
    )
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile="a/runtime.py",
            tofile="b/runtime.py",
        )
    )
    (root / "patch.diff").write_text(patch)
    atomic_json(
        root / "verification.json",
        {
            "before": before,
            "after": evaluated,
            "agent_status": state.status,
            "agent_error": agent_error,
            "agent_output": state.output,
        },
    )
    for name in (
        "patch.diff",
        "verification.json",
        "permissions.json",
        "conventions.json",
        "workspace/runtime.py",
        "workspace/AGENTS.md",
        "workspace/pyproject.toml",
    ):
        trace.artifact(name, root / name)
    trace.emit("measurements", {"values": measurements, "sources": sources})
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "task": {
            "id": manifest["id"],
            "version": manifest["version"],
            "fixture_hash": digest(manifest),
        },
        "protocol": {
            "id": manifest["checker"],
            "split": "development",
            "repeat": repeat,
        },
        "configuration": asdict(config),
        "hardware": hardware,
        "software": software,
        "correctness": correctness,
        "work": {
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
            "generated_tokens": state.output_tokens,
            "model_calls": state.model_calls,
            "retries": state.retries,
            "tool_calls": state.tool_calls,
        },
        "measurements": measurements,
        "measurement_sources": sources,
        "artifacts": dict(trace.artifacts),
    }
    trace.save(root / "trace.json")
    atomic_json(root / "result.json", validate_result(result))
    return result


async def experiment(output):
    if output.exists():
        raise ValueError("Choose a new evidence directory")
    conventions = resolve_conventions(FIXTURE)
    hardware, software = hardware_profile(), software_identity()
    manifest = json.loads((ROOT / "configurations/local-baselines-v1.json").read_text())
    strong = replace(
        AgentConfig(**manifest["profiles"]["medium-q4-gpu"]),
        context_size=8192,
        max_output_tokens=512,
    )
    validate_manifest(asdict(strong), *inventory())
    if hardware != manifest["hardware"]:
        raise ValueError("Hardware drift")
    allowed = {p["model"] for p in manifest["profiles"].values()}
    groups, cold = {}, []
    for label, profile in {
        "reference-fixture": AgentConfig(
            model="historical-reference", context_size=8192, max_output_tokens=512
        ),
        "local-model": strong,
    }.items():
        await asyncio.to_thread(unload_experiment_models, allowed)
        groups[label] = []
        for repeat in range(6):
            result = await repository_run(
                reference_adapter()
                if label == "reference-fixture"
                else OllamaAdapter(),
                replace(profile, seed=repeat),
                output / "runs",
                repeat=repeat,
                hardware=hardware,
                software=software,
                conventions=conventions,
            )
            (cold if repeat == 0 else groups[label]).append(result)
            print(
                canonical(
                    {
                        "configuration": label,
                        "repeat": repeat,
                        "status": result["correctness"]["status"],
                    }
                ),
                flush=True,
            )
    results = [r for group in groups.values() for r in group]
    report = {
        label: {
            "passed": sum(r["correctness"]["status"] == "pass" for r in group),
            "summary": summarize(group),
        }
        for label, group in groups.items()
    }
    report["comparisons"] = [
        compare(a, b) for a, b in zip(*groups.values(), strict=True)
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in results))
    atomic_json(output / "cold-results.json", cold)
    atomic_json(output / "report.json", report)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "results_sha256": digest(results),
            "cold_sha256": digest(cold),
            "sourceRevision": conventions["sourceRevision"],
            "repository_task": fixture_manifest(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(experiment(parser.parse_args().output))


if __name__ == "__main__":
    main()
