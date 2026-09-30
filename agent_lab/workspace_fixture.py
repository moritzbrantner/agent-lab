"""Retain an interrupt/resume workspace walkthrough using public fixture content."""

import argparse
import asyncio
import json
from pathlib import Path

from agent_lab.experiments import digest
from agent_lab.trace import atomic_json, load_trace
from agent_lab.workspace_store import Workspace


async def walkthrough(output):
    if output.exists():
        raise ValueError("Evidence output must be new")
    project = output / "project"
    project.mkdir(parents=True)
    (project / "README.md").write_text("Public workspace fixture project.\n")
    document = project / "context.txt"
    document.write_text("Verified attachment survives interrupt and resume.\n")
    workspace = Workspace(output / "history")
    session = workspace.start("public walkthrough", project, profile="fixture-demo")
    attachment = workspace.attach(session["id"], document)

    def interrupt(stage):
        if stage == "tool_completed":
            raise asyncio.CancelledError()

    try:
        await workspace.execute(
            session["id"], "Read the attached context", after_save=interrupt
        )
    except asyncio.CancelledError:
        pass
    before = workspace.inspect(session["id"])
    atomic_json(output / "interrupted.json", before)
    after = await workspace.resume(session["id"])
    workspace.memory(session["id"], enabled=True)
    configuration = workspace.export_config(session["id"])
    atomic_json(output / "exported-configuration.json", configuration)
    # start() returned the pre-run snapshot; the session record owns latest_run.
    root = (
        workspace._directory(session["id"])
        / "runs"
        / workspace.session(session["id"])["latest_run"]
    )
    load_trace(root / "trace.json", artifact_root=root)
    checks = {
        "interrupted": before["state"]["status"] == "cancelled",
        "completed": after["status"] == "completed",
        "attachment_output": after["output"] == document.read_text(),
        "completed_tool_not_repeated": before["state"]["tool_calls"]
        == after["tool_calls"]
        == 1,
        "configuration_retained": configuration == before["configuration"],
        "history_retained": len(workspace.history(session["id"])) == 1,
    }
    report = {
        "schema_version": 1,
        "scope": "controlled workspace fixture; no model-quality claim",
        "session_id": session["id"],
        "attachment": attachment,
        "checks": checks,
        "before_work": {
            name: before["state"][name]
            for name in ("request_attempts", "model_calls", "tool_calls")
        },
        "after_work": {
            name: after[name]
            for name in ("request_attempts", "model_calls", "tool_calls")
        },
        "configuration_sha256": digest(configuration),
        "conventions_sourceRevision": "e6acb5310afaf15c0cba24f87108f5f4ad1bedc3",
    }
    atomic_json(output / "report.json", report)
    if not all(checks.values()):
        raise ValueError("Workspace fixture failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(walkthrough(args.output)), indent=2))


if __name__ == "__main__":
    main()
