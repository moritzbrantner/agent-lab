"""Versioned ordered metadata, opt-in private content, and strict replay."""

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from jsonschema import Draft202012Validator

from agent_lab.experiments import ROOT, canonical, digest
from agent_lab.runtime import Reply, ResponseFailure, TransientModelError


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Trace:
    def __init__(
        self,
        run_id,
        task_id,
        configuration_id,
        *,
        retain_content=False,
        parent_run_id=None,
    ):
        self.identity = {
            "schema_version": 1,
            "run_id": run_id,
            "task_id": task_id,
            "configuration_id": configuration_id,
            "parent_run_id": parent_run_id,
            "replayable": retain_content,
        }
        self.retain_content = retain_content
        self.events = []
        self.payloads = {}
        self.artifacts = {}

    def emit(self, kind, payload):
        sequence = len(self.events)
        reference = str(sequence) if self.retain_content else None
        public = {}
        if kind == "state":
            public = {"status": payload["status"]}
        elif kind in ("artifact", "evaluator", "measurements", "authority"):
            public = copy.deepcopy(payload)
        elif kind == "failure":
            public = {"category": payload["category"]}
        self.events.append(
            {
                "id": f"{self.identity['run_id']}:{sequence}",
                "sequence": sequence,
                "kind": kind,
                "metadata": public,
                "content_ref": reference,
            }
        )
        if self.retain_content:
            self.payloads[reference] = copy.deepcopy(payload)

    def artifact(self, name, path):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifacts need contained relative names")
        hash_value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.artifacts[relative.as_posix()] = hash_value
        self.emit("artifact", {"path": relative.as_posix(), "sha256": hash_value})

    def document(self):
        return {
            **self.identity,
            "events": copy.deepcopy(self.events),
            "artifacts": dict(self.artifacts),
            "payloads": copy.deepcopy(self.payloads),
        }

    def save(self, path):
        path = Path(path)
        document = self.document()
        payloads = document.pop("payloads")
        if self.retain_content:
            atomic_json(
                path.with_suffix(path.suffix + ".content.json"),
                {
                    "payloads": payloads,
                    "sha256": digest(payloads),
                    "metadata_sha256": digest(document),
                },
            )
        atomic_json(path, document)


def load_trace(path, *, artifact_root=None):
    path = Path(path)
    document = json.loads(path.read_text())
    schema = json.loads((ROOT / "schemas/trace-v1.json").read_text())
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        raise ValueError("Invalid trace: " + errors[0].message)
    for index, event in enumerate(document["events"]):
        if event["sequence"] != index or event["id"] != f"{document['run_id']}:{index}":
            raise ValueError("Trace ordering/identity drift")
    if artifact_root is not None:
        root = Path(artifact_root).resolve()
        for name, expected in document["artifacts"].items():
            artifact = (root / name).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise ValueError("Artifact outside root or unavailable")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Artifact drift: {name}")
    document["payloads"] = {}
    if document["replayable"]:
        private = json.loads(
            path.with_suffix(path.suffix + ".content.json").read_text()
        )
        if digest(private["payloads"]) != private["sha256"]:
            raise ValueError("Private content drift")
        metadata = {key: value for key, value in document.items() if key != "payloads"}
        if digest(metadata) != private["metadata_sha256"]:
            raise ValueError("Trace/content continuation drift")
        document["payloads"] = private["payloads"]
    return document


class Replay:
    def __init__(self, document):
        if not document["replayable"]:
            raise ValueError("Replay requires explicit private content retention")
        self.payloads = document["payloads"]
        self.configuration = next(
            (
                self.payloads[event["content_ref"]]["configuration"]
                for event in document["events"]
                if event["kind"] == "state"
                and "configuration" in self.payloads[event["content_ref"]]
            ),
            None,
        )
        self.events = [
            event
            for event in document["events"]
            if event["kind"]
            in (
                "model_request",
                "model_response",
                "retry",
                "tool_request",
                "tool_result",
            )
        ]
        self.position = 0

    def _next(self, kind):
        if self.position >= len(self.events):
            raise ValueError("Replay exhausted")
        event = self.events[self.position]
        if event["kind"] != kind:
            raise ValueError(f"Replay expected {event['kind']}, received {kind}")
        self.position += 1
        try:
            return copy.deepcopy(self.payloads[event["content_ref"]])
        except KeyError as error:
            raise ValueError("Replay content unavailable") from error

    async def complete(self, messages, config):
        if self.configuration is not None and self.configuration != asdict(config):
            raise ValueError("Configuration drift")
        if self._next("model_request")["messages"] != messages:
            raise ValueError("Model request drift")
        if (
            self.position < len(self.events)
            and self.events[self.position]["kind"] == "retry"
        ):
            raise TransientModelError(self._next("retry")["reason"])
        payload = self._next("model_response")
        if payload.get("failure"):
            raise ResponseFailure(
                "Recorded malformed response",
                input_tokens=payload.get("input_tokens"),
                output_tokens=payload.get("output_tokens"),
                measurements=payload.get("measurements"),
            )
        return Reply.parse(payload)

    def tools(self):
        names = {
            self.payloads[e["content_ref"]]["name"]
            for e in self.events
            if e["kind"] == "tool_request"
        }

        def tool(name, arguments):
            request = self._next("tool_request")
            if request != {"name": name, "arguments": arguments}:
                raise ValueError("Tool request drift")
            response = self._next("tool_result")
            if response["name"] != name:
                raise ValueError("Tool result drift")
            return response["value"]

        return {
            name: lambda arguments, name=name: tool(name, arguments) for name in names
        }

    def assert_consumed(self):
        if self.position != len(self.events):
            raise ValueError("Unconsumed replay events")
