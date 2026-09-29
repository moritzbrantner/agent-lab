"""Immutable versioned tasks, constrained capabilities, canonical run results."""

import asyncio
import copy
import json
import platform
import re
from dataclasses import asdict, replace
from pathlib import Path

from jsonschema import Draft202012Validator

from agent_lab.evaluators import independent_evaluate
from agent_lab.experiments import MEASUREMENTS, ROOT, canonical, digest, validate_result
from agent_lab.oracles import incremental_sequence
from agent_lab.runtime import AgentState, Reply, SequenceAdapter, ToolCall, run
from agent_lab.trace import Trace, atomic_json


def load_task(split, task_id):
    if split not in ("development", "held-out") or not re.fullmatch(
        r"[a-z0-9-]+", task_id
    ):
        raise ValueError("Invalid task/split")
    task = json.loads((ROOT / "benchmarks" / split / f"{task_id}.json").read_text())
    schema = json.loads((ROOT / "schemas/task-v1.json").read_text())
    errors = list(Draft202012Validator(schema).iter_errors(task))
    if errors:
        raise ValueError("Invalid task: " + errors[0].message)
    if task["id"] != task_id or task["split"] != split:
        raise ValueError("Task identity drift")
    return task


def discover(split="development"):
    if split not in ("development", "held-out"):
        raise ValueError("Invalid split")
    return [
        load_task(split, path.stem)
        for path in sorted((ROOT / "benchmarks" / split).glob("*.json"))
    ]


def tool_registry():
    return {
        "index": lambda args: incremental_sequence(args["changes"]),
        "sum": lambda args: sum(args["numbers"]),
        "sort": lambda args: sorted(
            args["records"], key=lambda item: (item["rank"], item["id"])
        ),
        "lookup": lambda args: args["facts"][args["key"]],
        "repair": lambda args: args["source"].replace("return a - b", "return a + b"),
    }


def prompt_for(task):
    return (
        "Return a JSON reply with content (string) and calls (array of objects with "
        "name and arguments). Use only these tools: "
        + canonical(task["tools"])
        + ". When finished return calls: [] and the final answer in content. "
        + task["instruction"]
        + "\nInput: "
        + canonical(task["input"])
    )


def fixture_adapter(task):
    """A recorded fixture, never reported as real model inference."""
    tools = tool_registry()
    replies = []
    for planned in task["fixture_plan"]:
        replies.append(
            Reply(
                calls=tuple(ToolCall(c["name"], c["arguments"]) for c in planned),
                input_tokens=0,
                output_tokens=0,
            )
        )
    outputs = [
        tools[c["name"]](c["arguments"])
        for planned in task["fixture_plan"]
        for c in planned
    ]
    if task["category"] == "structured-reasoning":
        answer = str(sum(task["input"]["numbers"]))
    elif task["category"] == "long-running":
        answer = str(sum(outputs))
    elif task["category"] == "tool-transformation":
        answer = canonical(outputs[-1])
    else:
        answer = str(outputs[-1])
    replies.append(Reply(answer, input_tokens=0, output_tokens=0))
    return SequenceAdapter(replies)


async def benchmark(
    task,
    adapter,
    config,
    *,
    repeat=0,
    hardware=None,
    software=None,
    output_root=None,
    retain_content=False,
    evaluate=independent_evaluate,
):
    fixture = load_task(task["split"], task["id"])
    if task != fixture:
        raise ValueError("Candidate cannot redefine benchmark fixture")
    task = copy.deepcopy(fixture)
    config = replace(
        config,
        max_steps=min(config.max_steps, task["budgets"]["steps"]),
        max_tool_calls=min(config.max_tool_calls, task["budgets"]["tool_calls"]),
    )
    config_identity = asdict(config)
    run_id = digest(
        {"task": digest(task), "configuration": config_identity, "repeat": repeat}
    )[:24]
    trace = Trace(
        run_id, task["id"], digest(config_identity), retain_content=retain_content
    )
    state = AgentState(prompt_for(task))
    tools = {name: tool_registry()[name] for name in task["tools"]}
    try:
        await run(adapter, tools, config, state, trace.emit)
        correctness = evaluate(task, state.output)
    except asyncio.CancelledError:
        raise  # Caller owns lifecycle cancellation and retained state.
    except Exception as error:
        # Experiment boundary retains all failed samples without relabelling them.
        correctness = {"status": "error", "score": None, "evaluator": task["evaluator"]}
        trace.emit("failure", {"category": type(error).__name__})
    if load_task(task["split"], task["id"]) != fixture:
        raise ValueError("Benchmark fixture changed during run")
    trace.emit("evaluator", correctness)
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "task": {
            "id": task["id"],
            "version": task["version"],
            "fixture_hash": digest(fixture),
        },
        "protocol": {"id": task["evaluator"], "split": task["split"], "repeat": repeat},
        "configuration": config_identity,
        "hardware": hardware or {"profile": "deterministic-fixture"},
        "software": software
        or {"python": platform.python_version(), "runtime": "agent-lab-v1"},
        "correctness": correctness,
        "work": {
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
            "generated_tokens": state.output_tokens,
            "model_calls": state.model_calls,
            "retries": state.retries,
            "tool_calls": state.tool_calls,
        },
        "measurements": dict.fromkeys(MEASUREMENTS),
        "measurement_sources": {},
        "artifacts": {},
    }
    if output_root is not None:
        root = Path(output_root) / run_id
        atomic_json(root / "output.json", {"content": state.output})
        trace.artifact("output.json", root / "output.json")
        result["artifacts"] = dict(trace.artifacts)
        trace.save(root / "trace.json")
        atomic_json(root / "result.json", validate_result(result))
    return validate_result(result)
