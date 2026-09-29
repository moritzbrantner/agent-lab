import asyncio
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from agent_lab.checkpoints import CheckpointStore, run_checkpointed
from agent_lab.runtime import (
    AgentConfig,
    AgentState,
    FunctionAdapter,
    Reply,
    RuntimeFailure,
    ToolCall,
    run,
)
from agent_lab.trace import Trace


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_interruptions_do_not_repeat_completed_work(self):
        config = AgentConfig()
        for point in ("response", "tool_completed", "completed"):
            with self.subTest(point=point), tempfile.TemporaryDirectory() as directory:
                calls = {"model": 0, "tool": 0}

                def respond(messages, _, calls=calls):
                    calls["model"] += 1
                    return (
                        Reply("5", input_tokens=1, output_tokens=1)
                        if messages[-1]["role"] == "tool"
                        else Reply(
                            calls=(ToolCall("add", {}),),
                            input_tokens=1,
                            output_tokens=1,
                        )
                    )

                def add(_, calls=calls):
                    calls["tool"] += 1
                    return 5

                store = CheckpointStore(
                    Path(directory) / "checkpoint.json",
                    "task-hash",
                    config,
                    idempotent_tools={"add"},
                )
                trace = Trace("recovery", "task", "config")
                interrupted = False

                def pause(stage, point=point):
                    nonlocal interrupted
                    if stage == point and not interrupted:
                        interrupted = True
                        raise asyncio.CancelledError()

                with self.assertRaises(asyncio.CancelledError):
                    await run_checkpointed(
                        FunctionAdapter(respond),
                        {"add": add},
                        config,
                        AgentState("sum"),
                        trace,
                        store,
                        after_save=pause,
                    )
                state, continued = store.load()
                await run_checkpointed(
                    FunctionAdapter(respond),
                    {"add": add},
                    config,
                    state,
                    continued,
                    store,
                )
                self.assertEqual(state.output, "5")
                self.assertEqual(calls, {"model": 2, "tool": 1})
                self.assertEqual(state.model_calls, 2)
                ids = [e["id"] for e in continued.events]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(store.path.stat().st_mode & 0o777, 0o600)
                reference = AgentState("sum")
                await run(FunctionAdapter(respond), {"add": add}, config, reference)
                self.assertEqual(asdict(state), asdict(reference))

    async def test_ambiguous_action_and_identity_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AgentConfig()
            path = Path(directory) / "checkpoint.json"
            store = CheckpointStore(path, "task-hash", config)
            state = AgentState(
                "write",
                pending_calls=[{"name": "write", "arguments": {}}],
                active_call={"name": "write", "arguments": {}},
            )
            store.save(state, "tool_started", Trace("run", "task", "config"))
            with self.assertRaises(RuntimeFailure):
                store.load()
            with self.assertRaises(ValueError):
                CheckpointStore(path, "other-task", config).load()
            value = json.loads(path.read_text())
            value["state"]["steps"] = -1
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                store.load()

    async def test_artifact_drift_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "result.txt"
            artifact.write_text("verified")
            trace = Trace("run", "task", "config")
            trace.artifact("result.txt", artifact)
            store = CheckpointStore(root / "checkpoint.json", "task", AgentConfig())
            store.save(AgentState("task"), "pending", trace)
            artifact.write_text("changed")
            with self.assertRaises(ValueError):
                store.load()
