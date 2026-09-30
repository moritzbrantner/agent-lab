"""Acquire a fresh, source-bound release cohort and controlled scaffold proofs."""

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from agent_lab.backends import OllamaAdapter
from agent_lab.baselines import inventory, unload_experiment_models, validate_manifest
from agent_lab.benchmarks import benchmark, discover, load_task
from agent_lab.checkpoints import CheckpointStore, run_checkpointed
from agent_lab.experiments import ROOT, canonical, digest
from agent_lab.permission_fixture import fixture as permission_fixture
from agent_lab.permissions import PermissionDenied, Policy, SandboxUnavailable
from agent_lab.release import load_bundle, validate_profile
from agent_lab.repository_experiment import repository_run
from agent_lab.repository_workload import FIXTURE, fixture_manifest, resolve_conventions
from agent_lab.runtime import (
    AgentConfig,
    AgentState,
    FunctionAdapter,
    Reply,
    RuntimeFailure,
    ToolCall,
)
from agent_lab.telemetry import hardware_profile, software_identity
from agent_lab.trace import Trace, atomic_json
from agent_lab.workspace_store import Workspace

SOURCE_REVISION = "e6acb5310afaf15c0cba24f87108f5f4ad1bedc3"


def default_profile():
    baseline = json.loads((ROOT / "configurations/local-baselines-v1.json").read_text())
    base = AgentConfig(**baseline["profiles"]["medium-q4-gpu"])
    workloads = []
    for task in discover() + discover("held-out"):
        config = replace(
            base,
            max_steps=min(base.max_steps, task["budgets"]["steps"]),
            max_tool_calls=min(base.max_tool_calls, task["budgets"]["tool_calls"]),
        )
        workloads.append(
            {
                "task": {
                    "id": task["id"],
                    "version": task["version"],
                    "fixture_hash": digest(task),
                },
                "protocol": {"id": task["evaluator"], "split": task["split"]},
                "configuration": asdict(config),
            }
        )
    repository = fixture_manifest()
    workloads.append(
        {
            "task": {
                "id": repository["id"],
                "version": repository["version"],
                "fixture_hash": digest(repository),
            },
            "protocol": {"id": repository["checker"], "split": "development"},
            "configuration": asdict(
                replace(
                    base,
                    context_size=8192,
                    max_output_tokens=512,
                    options={
                        **base.options,
                        "repository_fixture": digest(repository),
                        "convention_sourceRevision": SOURCE_REVISION,
                        "permission_profile": "historical-scoped-repair-v1",
                    },
                )
            ),
        }
    )
    thresholds = []

    def threshold(
        identifier,
        category,
        field,
        aggregation,
        limit,
        description,
        tasks=None,
        *,
        operator=">=",
        required=True,
        minimum_tool_calls=0,
    ):
        value = {
            "id": identifier,
            "category": category,
            "source": "runs" if tasks else "proofs",
            "field": field,
            "aggregation": aggregation,
            "operator": operator,
            "limit": limit,
            "required": required,
            "description": description,
        }
        if tasks:
            value["workloads"] = tasks
            if minimum_tool_calls:
                value["minimum_tool_calls"] = minimum_tool_calls
        thresholds.append(value)

    interactive = [w["task"]["id"] for w in workloads[:-1]]
    all_tasks = interactive + [repository["id"]]
    threshold(
        "chat-reasoning",
        "interactive chat/reasoning",
        "correctness.status",
        "success_rate",
        0.95,
        "Exact arithmetic replies, per task; no subjective judge.",
        ["sum", "held-sum"],
    )
    threshold(
        "coding",
        "coding",
        "correctness.status",
        "success_rate",
        0.95,
        "Source repair plus protected executable historical repository repair.",
        ["repair-add", repository["id"]],
    )
    threshold(
        "tool-reliability",
        "tools",
        "correctness.status",
        "success_rate",
        0.95,
        "Correct final output and at least one completed/attempted registered tool.",
        ["batch-sum", "index-changes", "sort-records", "retrieve-fact"],
        minimum_tool_calls=1,
    )
    threshold(
        "held-out",
        "held-out correctness",
        "correctness.status",
        "success_rate",
        0.95,
        "Each controller-owned held-out task must independently meet the rate.",
        ["held-sum", "held-batch-sum", "held-retrieve-fact"],
    )
    threshold(
        "context-retention",
        "context",
        "correctness.status",
        "success_rate",
        0.95,
        "Exact key retention on explicit short retrieval fixtures.",
        ["retrieve-fact", "held-retrieve-fact"],
    )
    threshold(
        "requested-context",
        "context",
        "configuration.context_size",
        "min",
        4096,
        "Configured window only; does not establish effective model capacity.",
        all_tasks,
    )
    threshold(
        "effective-context",
        "context",
        "effective_context_tokens",
        "value",
        2048,
        "Correct independently checked retrieval from a >=2048-token single prompt.",
    )
    threshold(
        "interactive-latency",
        "latency",
        "measurements.wall_seconds",
        "p95",
        10,
        "Warm end-to-end wall time per explicit interactive task, including failures.",
        interactive,
        operator="<=",
    )
    threshold(
        "coding-latency",
        "latency",
        "measurements.wall_seconds",
        "p95",
        30,
        "Warm repository repair including setup and protected verification.",
        [repository["id"]],
        operator="<=",
    )
    for name, limit in (("ram", 24 * 1024**3), ("vram", 8 * 1024**3)):
        threshold(
            "peak-" + name,
            "memory",
            "measurements.peak_" + name + "_bytes",
            "max",
            limit,
            "Maximum sampled harness plus scoped backend allocation.",
            all_tasks,
            operator="<=",
        )
    threshold(
        "energy",
        "energy",
        "measurements.energy_joules",
        "p95",
        500,
        "Optional measured RAPL package joules; unavailable is never zero.",
        all_tasks,
        operator="<=",
        required=False,
    )
    for name, description in (
        (
            "recovery",
            "Interrupt a three-step task; resume without repeating completed tools.",
        ),
        (
            "bounded-resume",
            "Repeated cancellations cannot create free request attempts.",
        ),
        ("ambiguous-action", "Refuse unknown completion of a non-idempotent action."),
        (
            "durable-memory",
            "Owner-enabled memory survives controller recreation and can be cleared.",
        ),
        (
            "offline",
            "Loopback-only adapter, denied network tools, offline native process.",
        ),
    ):
        threshold(name, name, name, "value", 1, description)
    return validate_profile(
        {
            "schema_version": 1,
            "id": "useful-local-3060ti-v1",
            "scope": (
                "Explicit fixture capabilities on one fixed host; no hosted-assistant "
                "parity claim. Scaffold proofs are controlled sequences, not "
                "model-quality samples. Held-out fixtures are controller-owned "
                "public regression tasks, not a secret test set."
            ),
            "hardware": baseline["hardware"],
            "configuration": asdict(base),
            "minimum_repeats": 5,
            "workloads": workloads,
            "thresholds": thresholds,
        }
    )


async def scaffold_proofs(profile, root, software):
    root.mkdir(parents=True)
    config = AgentConfig(**profile["configuration"])
    task = load_task("development", "batch-sum")
    state = AgentState("Controlled recovery of the three-batch fixture")
    trace = Trace("release-recovery", "batch-sum", digest(asdict(config)))
    store = CheckpointStore(
        root / "recovery.json", digest(task), config, idempotent_tools={"sum"}
    )
    executed = []

    def add(args):
        executed.append(args["numbers"])
        return sum(args["numbers"])

    def respond(messages, _):
        done = sum(m["role"] == "tool" for m in messages)
        return (
            Reply("45", input_tokens=0, output_tokens=0)
            if done == 3
            else Reply(
                calls=(ToolCall("sum", {"numbers": task["input"]["batches"][done]}),),
                input_tokens=0,
                output_tokens=0,
            )
        )

    def interrupt(stage):
        if stage == "tool_completed":
            raise asyncio.CancelledError()

    try:
        await run_checkpointed(
            FunctionAdapter(respond),
            {"sum": add},
            config,
            state,
            trace,
            store,
            after_save=interrupt,
        )
    except asyncio.CancelledError:
        pass
    atomic_json(root / "interrupted.json", asdict(state))
    state, trace = store.load()
    await run_checkpointed(
        FunctionAdapter(respond),
        {"sum": add},
        config,
        state,
        trace,
        store,
    )
    trace.save(root / "recovery-trace.json")
    recovery = (
        state.status == "completed"
        and state.output == "45"
        and executed == task["input"]["batches"]
        and state.tool_calls == 3
        and state.model_calls == 4
    )
    cancelled = AgentState("Repeated interrupted requests")
    bounded_trace = Trace("release-budget", "cancel-budget", digest(asdict(config)))
    bounded_store = CheckpointStore(root / "budget.json", "cancel-budget", config)

    def cancel(*_):
        raise asyncio.CancelledError()

    refused = False
    for _ in range(config.max_steps + config.max_retries + 1):
        try:
            await run_checkpointed(
                FunctionAdapter(cancel),
                {},
                config,
                cancelled,
                bounded_trace,
                bounded_store,
            )
        except asyncio.CancelledError:
            cancelled, bounded_trace = bounded_store.load()
        except RuntimeFailure:
            refused = True
            break
    bounded_trace.save(root / "budget-trace.json")
    ambiguous_store = CheckpointStore(root / "ambiguous.json", "write", config)
    call = {"name": "write", "arguments": {}}
    ambiguous_store.save(
        AgentState("write", pending_calls=[call], active_call=call),
        "tool_started",
        Trace("release-ambiguous", "write", digest(asdict(config))),
    )
    ambiguous = False
    try:
        ambiguous_store.load()
    except RuntimeFailure:
        ambiguous = True
    workspace = Workspace(root / "history")
    project = root / "project"
    project.mkdir()
    session = workspace.start("memory proof", project, profile="fixture-demo")
    workspace.memory(session["id"], enabled=True)
    memory_config = {
        "project": str(project.resolve()),
        "permissions": workspace.session(session["id"])["permissions"],
        "attachments": [],
        "memory_enabled": True,
        "memory_root": str(workspace._directory(session["id"]) / "memory"),
    }
    tools = workspace._tools(memory_config, root, lambda *_: None)[0]
    tools["remember"]({"key": "proof", "value": "durable fixture value"})
    recreated = Workspace(workspace.root)
    read_back = recreated._tools(memory_config, root, lambda *_: None)[0]["recall"](
        {"key": "proof"}
    )
    recreated.memory(session["id"], clear=True)
    memory = (
        read_back == "durable fixture value"
        and not (Path(memory_config["memory_root"]) / "proof").exists()
    )
    denied = False
    try:
        Policy().require("network", "https://example.invalid", "get")
    except PermissionDenied:
        denied = True
    nonlocal_endpoint = False
    try:
        OllamaAdapter("https://example.invalid")
    except ValueError:
        nonlocal_endpoint = True
    offline = None
    offline_evidence = []
    try:
        native = await permission_fixture(root / "offline")
        offline = int(
            denied
            and nonlocal_endpoint
            and native["denied"]
            and native["sandbox"]["stdout"].splitlines() == ["False", "True"]
        )
        offline_evidence = ["proofs/offline/report.json", "proofs/offline/trace.json"]
    except SandboxUnavailable as error:
        atomic_json(
            root / "offline-unavailable.json", {"category": type(error).__name__}
        )
    values = {}
    for name, value, evidence in (
        (
            "recovery",
            int(recovery),
            ["proofs/interrupted.json", "proofs/recovery.json"],
        ),
        (
            "bounded-resume",
            int(
                refused
                and cancelled.request_attempts == config.max_steps + config.max_retries
            ),
            ["proofs/budget.json"],
        ),
        ("ambiguous-action", int(ambiguous), ["proofs/ambiguous.json"]),
        (
            "durable-memory",
            int(memory),
            ["proofs/history", "proofs/memory-observation.json"],
        ),
        ("offline", offline, offline_evidence),
        ("effective_context_tokens", None, []),
    ):
        values[name] = {
            "value": value,
            "scope": (
                "No >=2048-token retrieval workload measured; "
                "configured window is separate"
                if name == "effective_context_tokens"
                else (
                    "Controlled scaffold proof with fixture replies; native process "
                    "offline isolation only, not host airgap"
                )
            ),
            "evidence": evidence,
        }
    atomic_json(
        root / "memory-observation.json", {"read_back": read_back, "cleared": memory}
    )
    return {
        "profile_sha256": digest(profile),
        "hardware": profile["hardware"],
        "software": software,
        "configuration": profile["configuration"],
        "values": values,
    }


async def experiment(profile, output):
    validate_profile(profile)
    if output.exists():
        raise ValueError("Choose a new immutable release evidence directory")
    hardware, software = hardware_profile(), software_identity()
    if hardware != profile["hardware"]:
        raise ValueError("Fixed release hardware drift")
    conventions = resolve_conventions(FIXTURE)
    if conventions["sourceRevision"] != SOURCE_REVISION:
        raise ValueError("Review the release profile against current shared policy")
    for workload in profile["workloads"]:
        validate_manifest(workload["configuration"], *inventory())
    output.mkdir(parents=True)
    atomic_json(output / "profile.json", profile)
    atomic_json(output / "conventions.json", conventions)
    atomic_json(
        output / "measured-source.json",
        {
            path.relative_to(ROOT).as_posix(): path.read_text()
            for path in sorted((ROOT / "agent_lab").glob("*.py"))
        },
    )
    proofs = await scaffold_proofs(profile, output / "proofs", software)
    atomic_json(output / "proofs.json", proofs)
    runs, cold = [], []
    allowed = {profile["configuration"]["model"]}
    for workload in profile["workloads"]:
        config = AgentConfig(**workload["configuration"])
        repository = workload["task"]["id"] == "repository-unknown-usage"
        await asyncio.to_thread(unload_experiment_models, allowed)
        for repeat in [profile["minimum_repeats"], *range(profile["minimum_repeats"])]:
            is_cold = repeat == profile["minimum_repeats"]
            actual_config = replace(config, seed=99 if is_cold else repeat)
            if repository:
                sample = await repository_run(
                    OllamaAdapter(),
                    actual_config,
                    output / "runs",
                    repeat=repeat,
                    hardware=hardware,
                    software=software,
                    conventions=conventions,
                )
            else:
                task = load_task(workload["protocol"]["split"], workload["task"]["id"])
                sample = await benchmark(
                    task,
                    OllamaAdapter(),
                    actual_config,
                    repeat=repeat,
                    hardware=hardware,
                    software=software,
                    output_root=output / "runs",
                    retain_content=True,
                )
            (cold if is_cold else runs).append(sample)
            for name, values in (("results.jsonl", runs), ("cold-results.jsonl", cold)):
                (output / name).write_text("".join(canonical(r) + "\n" for r in values))
            print(
                canonical(
                    {
                        "task": sample["task"]["id"],
                        "cold": is_cold,
                        "repeat": repeat,
                        "status": sample["correctness"]["status"],
                    }
                ),
                flush=True,
            )
    if software != software_identity() or hardware != hardware_profile():
        raise ValueError("Release environment changed during measurement")
    files = {
        path.relative_to(output).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "profile_sha256": digest(profile),
            "software": software,
            "conventions_sourceRevision": conventions["sourceRevision"],
            "files": files,
            "cold_policy": (
                "One separate cold-load sample per workload; "
                "excluded from warm thresholds"
            ),
        },
    )
    report = load_bundle(profile, output)
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("profile", "run"))
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "profile":
        if args.output.exists():
            raise ValueError("Choose a new profile path")
        atomic_json(args.output, default_profile())
    else:
        if args.profile is None:
            parser.error("run requires --profile")
        report = asyncio.run(
            experiment(json.loads(args.profile.read_text()), args.output)
        )
        print(canonical({"status": report["status"], "ready": report["ready"]}))


if __name__ == "__main__":
    main()
