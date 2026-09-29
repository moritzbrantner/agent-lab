"""Bounded generation/verification with all candidate and checking costs retained."""

import asyncio
import resource
import time
from dataclasses import asdict, replace

from agent_lab.benchmarks import benchmark
from agent_lab.evaluators import independent_evaluate
from agent_lab.experiments import digest, validate_result
from agent_lab.telemetry import (
    EnergyCounter,
    Telemetry,
    hardware_profile,
    software_identity,
)
from agent_lab.trace import Trace, atomic_json


def _total(values):
    return sum(values) if all(value is not None for value in values) else None


class TimedEvaluator:
    def __init__(self):
        self.calls = 0
        self.wall_seconds = 0

    def __call__(self, task, output):
        started = time.perf_counter()
        self.calls += 1
        try:
            return independent_evaluate(task, output)
        finally:
            self.wall_seconds += time.perf_counter() - started


async def generate_verify(
    task,
    configurations,
    adapter_factory,
    *,
    max_parallel=1,
    repeat=0,
    hardware=None,
    software=None,
    output_root=None,
):
    if not configurations or len(configurations) > 8 or not 1 <= max_parallel <= 4:
        raise ValueError("Generation requires 1..8 candidates and parallelism 1..4")
    hardware, software = hardware or hardware_profile(), software or software_identity()
    policy = {
        "strategy": "generate-verify-v1",
        "schedule": [asdict(c) for c in configurations],
        "max_candidates": len(configurations),
        "seed": repeat,
        "seed_derivation": "base + repeat * candidate_count + candidate_index",
        "max_parallel": max_parallel,
    }
    run_id = digest(
        {
            "task": digest(task),
            "configuration": policy,
            "repeat": repeat,
            "hardware": hardware,
            "software": software,
        }
    )[:24]
    trace = Trace(run_id, task["id"], digest(policy))
    meter = Telemetry(energy=EnergyCounter.discover())
    evaluator = TimedEvaluator()
    child_cpu_started = resource.getrusage(resource.RUSAGE_CHILDREN)
    meter.start()
    candidates = []
    selected = None
    try:
        for offset in range(0, len(configurations), max_parallel):
            batch = []
            for index in range(offset, min(offset + max_parallel, len(configurations))):
                feedback = (
                    "Earlier candidates failed verification; revise the solution."
                    if offset
                    else ""
                )
                config = replace(
                    configurations[index],
                    feedback=feedback,
                    seed=configurations[index].seed
                    + repeat * len(configurations)
                    + index,
                )
                trace.emit(
                    "child", {"child_id": f"candidate:{index}", "kind": "run_start"}
                )
                batch.append(
                    benchmark(
                        task,
                        adapter_factory(config),
                        config,
                        repeat=repeat,
                        hardware=hardware,
                        software=software,
                        output_root=output_root / "candidates" if output_root else None,
                        evaluate=evaluator,
                        parent_run_id=run_id,
                    )
                )
            async with asyncio.TaskGroup() as group:
                jobs = [group.create_task(job) for job in batch]
            completed = [job.result() for job in jobs]
            for index, candidate in enumerate(completed, offset):
                candidates.append(candidate)
                trace.emit(
                    "child",
                    {
                        "child_id": candidate["run_id"],
                        "kind": "run_complete",
                        "candidate_index": index,
                    },
                )
                if selected is None and candidate["correctness"]["status"] == "pass":
                    selected = index
            if selected is not None:
                break
    finally:
        measurements, sources = meter.stop()
    child_cpu_ended = resource.getrusage(resource.RUSAGE_CHILDREN)
    for key in ("inference_seconds", "setup_seconds", "tool_seconds"):
        measurements[key] = _total([c["measurements"][key] for c in candidates])
        if measurements[key] is None:
            sources.pop(key, None)
        else:
            sources[key] = "reported:sum-all-candidate-" + key
    correctness = (
        {"status": "pass", "score": 1, "evaluator": task["evaluator"]}
        if selected is not None
        else {
            "status": "fail" if evaluator.calls else "error",
            "score": 0 if evaluator.calls else None,
            "evaluator": task["evaluator"],
        }
    )
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "task": candidates[0]["task"],
        "protocol": candidates[0]["protocol"],
        "configuration": policy,
        "hardware": hardware,
        "software": software,
        "correctness": correctness,
        "work": {
            key: _total([c["work"][key] for c in candidates])
            for key in candidates[0]["work"]
        },
        "measurements": measurements,
        "measurement_sources": sources,
        "artifacts": {},
    }
    verification = {
        "kind": "deterministic-exact",
        "calls": evaluator.calls,
        "wall_seconds": evaluator.wall_seconds,
        "job_child_cpu_seconds": (
            child_cpu_ended.ru_utime
            + child_cpu_ended.ru_stime
            - child_cpu_started.ru_utime
            - child_cpu_started.ru_stime
        ),
        "cpu_scope": "job children: evaluators and telemetry commands",
    }
    report = {
        "schema_version": 1,
        "result": result,
        "candidates": candidates,
        "selected_index": selected,
        "verification": verification,
    }
    trace.emit("evaluator", correctness)
    trace.emit("measurements", {"values": measurements, "sources": sources})
    if output_root:
        atomic_json(
            output_root / "selection.json",
            {
                "selected_index": selected,
                "candidate_runs": [c["run_id"] for c in candidates],
            },
        )
        trace.artifact("selection.json", output_root / "selection.json")
        result["artifacts"] = dict(trace.artifacts)
        trace.save(output_root / "trace.json")
        atomic_json(output_root / "report.json", report)
    validate_result(result)
    return report
