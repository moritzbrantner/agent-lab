import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agent_lab.runtime import (
    AgentConfig,
    AgentState,
    Reply,
    SequenceAdapter,
    ToolCall,
    run,
)
from agent_lab.trace import Replay, Trace, load_trace


class TraceTests(unittest.TestCase):
    def test_record_and_replay_without_real_tools_or_model(self):
        trace = Trace("run", "sum", "config", retain_content=True)
        original = AgentState("sum")
        asyncio.run(
            run(
                SequenceAdapter(
                    [Reply(calls=(ToolCall("sum", {"numbers": [2, 3]}),)), Reply("5")]
                ),
                {"sum": lambda a: sum(a["numbers"])},
                AgentConfig(),
                original,
                trace.emit,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            trace.save(path)
            recording = load_trace(path)
            replay = Replay(recording)
            state = AgentState("sum")
            asyncio.run(run(replay, replay.tools(), AgentConfig(), state))
            replay.assert_consumed()
            self.assertEqual(state, original)
            self.assertEqual(
                [e["sequence"] for e in recording["events"]],
                list(range(len(recording["events"]))),
            )

    def test_private_payloads_are_absent_by_default(self):
        trace = Trace("run", "task", "config")
        trace.emit("model_request", {"messages": [{"content": "secret-value"}]})
        self.assertNotIn("secret-value", json.dumps(trace.document()))
        with self.assertRaises(ValueError):
            Replay(trace.document())

    def test_artifact_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "output.txt"
            artifact.write_text("first")
            trace = Trace("run", "task", "config", retain_content=True)
            trace.artifact("output.txt", artifact)
            path = root / "trace.json"
            trace.save(path)
            load_trace(path, artifact_root=root)
            artifact.write_text("changed")
            with self.assertRaises(ValueError):
                load_trace(path, artifact_root=root)

    def test_replay_rejects_changed_requests(self):
        trace = Trace("run", "task", "config", retain_content=True)
        trace.emit("model_request", {"messages": [{"role": "user", "content": "a"}]})
        trace.emit("model_response", {"content": "ok", "calls": []})
        with self.assertRaises(ValueError):
            asyncio.run(Replay(trace.document()).complete([], AgentConfig()))
