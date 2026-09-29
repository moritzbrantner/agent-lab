import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_lab.permissions import (
    Grant,
    LocalTools,
    PermissionDenied,
    Policy,
    SandboxUnavailable,
)
from agent_lab.trace import Trace


class PermissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_allowed_denied_and_owner_elevation_are_secret_free(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.txt").write_text("hello")
            trace = Trace("permissions", "fixture", "policy", retain_content=True)
            policy = Policy(
                [
                    Grant("repositories", str(root), "read"),
                    Grant("secrets", "key", "handle"),
                ],
                trace.emit,
            )
            tools = LocalTools(policy, secrets={"key": "credential-never-in-trace"})
            self.assertEqual(
                tools.read(
                    {"scope": "repositories", "root": str(root), "path": "note.txt"}
                ),
                "hello",
            )
            with self.assertRaises(PermissionDenied):
                tools.write(
                    {
                        "scope": "repositories",
                        "root": str(root),
                        "path": "note.txt",
                        "content": "new",
                    }
                )
            with self.assertRaises(PermissionDenied):
                tools.read(
                    {
                        "scope": "repositories",
                        "root": str(root),
                        "path": "../private.txt",
                    }
                )
            handle = tools.secret_handle({"name": "key"})
            self.assertTrue(handle.startswith("credential:"))
            elevated = policy.elevate(
                Grant("repositories", str(root), "write"),
                reason="fixture owner approval",
                approval_id="owner:1",
            )
            LocalTools(elevated).write(
                {
                    "scope": "repositories",
                    "root": str(root),
                    "path": "note.txt",
                    "content": "new",
                }
            )
            self.assertEqual((root / "note.txt").read_text(), "new")
            self.assertNotIn("credential-never-in-trace", str(trace.document()))
            self.assertTrue(
                any(e["metadata"].get("action") == "elevation" for e in trace.events)
            )
            (root / "link").symlink_to(root / "note.txt")
            with self.assertRaises(PermissionDenied):
                tools.read({"scope": "repositories", "root": str(root), "path": "link"})

    async def test_process_is_offline_without_host_secrets_and_cancel_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = Trace("sandbox", "fixture", "policy")
            executable = "/usr/bin/python3.14"
            policy = Policy(
                [
                    Grant("process", executable, "execute"),
                    Grant("repositories", str(root), "read"),
                ],
                trace.emit,
            )
            tools = LocalTools(policy)
            try:
                with patch.dict(os.environ, {"AGENT_LAB_TEST_SECRET": "private"}):
                    result = await tools.process(
                        {
                            "argv": [
                                executable,
                                "-I",
                                "-c",
                                "import os; print(os.getenv('AGENT_LAB_TEST_SECRET')); "
                                "print(os.path.exists('/home'))",
                            ]
                        }
                    )
            except SandboxUnavailable as error:
                self.skipTest(str(error))
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["stdout"].splitlines(), ["None", "False"])
            with self.assertRaises(PermissionDenied):
                await tools.process({"argv": ["/bin/sh", "-c", "true"]})
            task = asyncio.create_task(
                tools.process(
                    {"argv": [executable, "-I", "-c", "import time; time.sleep(30)"]}
                )
            )
            await asyncio.sleep(0.1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_network_document_and_memory_grants_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.txt").write_text("document")
            policy = Policy([Grant("documents", str(root), "read")])
            tools = LocalTools(policy)
            self.assertEqual(
                tools.read(
                    {"scope": "documents", "root": str(root), "path": "note.txt"}
                ),
                "document",
            )
            with self.assertRaises(PermissionDenied):
                tools.read({"scope": "memory", "root": str(root), "path": "note.txt"})
            with self.assertRaises(PermissionDenied):
                await tools.fetch({"url": "http://127.0.0.1:1/private"})
