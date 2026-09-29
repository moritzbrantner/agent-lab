"""Private atomic checkpoints binding state, authority and trace continuation."""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from jsonschema import Draft202012Validator

from agent_lab.experiments import ROOT, digest
from agent_lab.runtime import AgentState, Reply, RuntimeFailure, run
from agent_lab.trace import Trace, atomic_json


class CheckpointStore:
    def __init__(
        self,
        path,
        task_identity,
        config,
        *,
        idempotent_tools=(),
        artifact_root=None,
        tool_identity=None,
    ):
        self.path = Path(path)
        self.identity = {
            "task": task_identity,
            "configuration": asdict(config),
            "idempotent_tools": sorted(idempotent_tools),
            "tools": tool_identity or {},
        }
        self.idempotent_tools = set(idempotent_tools)
        self.artifact_root = Path(artifact_root or self.path.parent).resolve()

    def save(self, state, stage, trace):
        trace.emit("checkpoint", {"stage": stage})
        value = {
            "schema_version": 1,
            "identity": self.identity,
            "stage": stage,
            "state": asdict(state),
            "trace": trace.document(),
        }
        atomic_json(self.path, {**value, "sha256": digest(value)})

    def load(self):
        value = json.loads(self.path.read_text())
        expected = value.pop("sha256")
        if (
            digest(value) != expected
            or value["identity"] != self.identity
            or value["schema_version"] != 1
        ):
            raise ValueError("Checkpoint content or authority drift")
        state = AgentState(**value["state"])
        for name in (
            "steps",
            "model_calls",
            "tool_calls",
            "retries",
            "input_tokens",
            "output_tokens",
        ):
            count = getattr(state, name)
            if count is None and name in ("input_tokens", "output_tokens"):
                continue
            if type(count) is not int or count < 0:
                raise ValueError("Invalid checkpoint counter")
        config = self.identity["configuration"]
        if (
            state.steps > config["max_steps"]
            or state.retries > config["max_retries"]
            or state.tool_calls > config["max_tool_calls"]
        ):
            raise ValueError("Checkpoint exceeds original budget")
        if (
            not isinstance(state.task, str)
            or not isinstance(state.messages, list)
            or not isinstance(state.pending_calls, list)
            or not isinstance(state.working_memory, list)
        ):
            raise ValueError("Invalid checkpoint state")
        if state.status not in (
            "pending",
            "running",
            "completed",
            "cancelled",
            "failed",
        ):
            raise ValueError("Invalid checkpoint status")
        Reply.parse({"calls": state.pending_calls})
        if state.active_call is not None and (
            not state.pending_calls or state.active_call != state.pending_calls[0]
        ):
            raise ValueError("Active action differs from outstanding work")
        document = value["trace"]
        metadata = {k: v for k, v in document.items() if k != "payloads"}
        schema = json.loads((ROOT / "schemas/trace-v1.json").read_text())
        Draft202012Validator(schema).validate(metadata)
        for index, event in enumerate(document["events"]):
            if (
                event["sequence"] != index
                or event["id"] != f"{document['run_id']}:{index}"
            ):
                raise ValueError("Checkpoint trace continuation drift")
        for name, expected in document["artifacts"].items():
            path = (self.artifact_root / name).resolve()
            if (
                not path.is_relative_to(self.artifact_root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("Checkpoint artifact drift")
        if state.active_call is not None:
            if state.active_call["name"] not in self.idempotent_tools:
                raise RuntimeFailure("Non-idempotent action completion is ambiguous")
            state.active_call = None
        if state.request_in_flight:
            # Inference has no tool side effects, but unknown spent work is retained.
            state.input_tokens = state.output_tokens = None
            state.request_in_flight = False
        trace = Trace(
            document["run_id"],
            document["task_id"],
            document["configuration_id"],
            retain_content=document["replayable"],
            parent_run_id=document["parent_run_id"],
        )
        trace.events = copy.deepcopy(document["events"])
        trace.payloads = copy.deepcopy(document["payloads"])
        trace.artifacts = dict(document["artifacts"])
        return state, trace


async def run_checkpointed(
    adapter, tools, config, state, trace, store, *, after_save=None
):
    if asdict(config) != store.identity["configuration"]:
        raise ValueError("Checkpoint runtime configuration drift")

    def checkpoint(stage, current):
        store.save(current, stage, trace)
        if after_save is not None:
            after_save(stage)

    return await run(adapter, tools, config, state, trace.emit, checkpoint=checkpoint)
