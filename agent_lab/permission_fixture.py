"""Retain allowed, denied and owner-elevated local tool/sandbox evidence."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from agent_lab.experiments import ROOT, digest
from agent_lab.permissions import Grant, LocalTools, PermissionDenied, Policy
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


async def fixture(output):
    if output.exists():
        raise ValueError("Choose a new evidence directory")
    root = (output / "workspace").resolve()
    root.mkdir(parents=True, mode=0o700)
    (root / "note.txt").write_text("fixture read")
    trace = Trace(
        digest({"fixture": "permissions-v1", "root": str(root)})[:24],
        "permissions-v1",
        "explicit-policy",
    )
    meter = Telemetry(energy=EnergyCounter.discover())
    meter.start()

    def emit(kind, payload):
        trace.emit(kind, payload)
        meter.emit(kind, payload)

    try:
        policy = Policy(
            [
                Grant("repositories", str(root), "read"),
                Grant("secrets", "fixture-key", "handle"),
                Grant("process", "/usr/bin/python3.14", "execute"),
            ],
            emit,
        )
        tools = LocalTools(
            policy, secrets={"fixture-key": "fixture-secret-do-not-retain"}
        )
        args = {"scope": "repositories", "root": str(root), "path": "note.txt"}
        allowed = AgentState("read a file and request an opaque credential handle")
        await run(
            SequenceAdapter(
                [
                    Reply(
                        calls=(ToolCall("read_file", args),),
                        input_tokens=0,
                        output_tokens=0,
                    ),
                    Reply(
                        calls=(ToolCall("credential_handle", {"name": "fixture-key"}),),
                        input_tokens=0,
                        output_tokens=0,
                    ),
                    Reply("done", input_tokens=0, output_tokens=0),
                ]
            ),
            tools.registry(),
            AgentConfig(),
            allowed,
            emit,
        )
        denied = False
        try:
            await run(
                SequenceAdapter(
                    [
                        Reply(
                            calls=(
                                ToolCall("write_file", {**args, "content": "denied"}),
                            ),
                            input_tokens=0,
                            output_tokens=0,
                        )
                    ]
                ),
                tools.registry(),
                AgentConfig(),
                AgentState("attempt unapproved write"),
                emit,
            )
        except PermissionDenied:
            denied = True
        if (root / "note.txt").read_text() != "fixture read":
            raise ValueError("Denied action changed data")
        elevated = policy.elevate(
            Grant("repositories", str(root), "write"),
            reason="fixture owner approves bounded text write",
            approval_id="fixture-owner:1",
        )
        elevated_tools = LocalTools(elevated)
        state = AgentState("perform owner-approved write and isolated process check")
        script = (
            "import os,socket; print(os.path.exists('/home')); "
            "print(socket.socket().connect_ex(('127.0.0.1',9)) != 0)"
        )
        await run(
            SequenceAdapter(
                [
                    Reply(
                        calls=(
                            ToolCall("write_file", {**args, "content": "approved"}),
                        ),
                        input_tokens=0,
                        output_tokens=0,
                    ),
                    Reply(
                        calls=(
                            ToolCall(
                                "run_process",
                                {"argv": ["/usr/bin/python3.14", "-I", "-c", script]},
                            ),
                        ),
                        input_tokens=0,
                        output_tokens=0,
                    ),
                    Reply("done", input_tokens=0, output_tokens=0),
                ]
            ),
            elevated_tools.registry(),
            AgentConfig(),
            state,
            emit,
        )
        process_result = state.working_memory[-1]["result"]
    finally:
        values, sources = meter.stop()
    if process_result["returncode"] or process_result["stdout"].splitlines() != [
        "False",
        "True",
    ]:
        raise ValueError("Native isolation fixture failed")
    if "fixture-secret-do-not-retain" in str(trace.document()):
        raise ValueError("Credential leaked into trace")
    report = {
        "schema_version": 1,
        "allowed": allowed.status == "completed",
        "denied": denied,
        "elevated": (root / "note.txt").read_text() == "approved",
        "sandbox": process_result,
        "policy": policy.document(),
        "elevated_policy": elevated.document(),
        "hardware": hardware_profile(),
        "software": software_identity(),
        "measurements": values,
        "measurement_sources": sources,
        "sandbox_pin": json.loads(
            (ROOT / "configurations/sandbox-v1.json").read_text()
        ),
        "tool_semantics": {
            "idempotent": sorted(tools.idempotent_tools),
            "non_idempotent": ["write_file", "delete_file", "run_process", "http_get"],
        },
        "grants_sha256": digest([asdict(g) for g in elevated.grants]),
    }
    atomic_json(output / "report.json", report)
    trace.artifact("report.json", output / "report.json")
    trace.artifact("workspace/note.txt", root / "note.txt")
    trace.save(output / "trace.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(fixture(parser.parse_args().output))


if __name__ == "__main__":
    main()
