import asyncio
import tempfile
import unittest
from pathlib import Path

from agent_lab.workspace_store import Workspace


class WorkspaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_durable_memory_requires_owner_enablement_and_clear_removes_value(
        self,
    ):
        from agent_lab.permissions import PermissionDenied

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            workspace = Workspace(root / "history")
            session = workspace.start("memory", project, profile="fixture-demo")

            def tools():
                current = workspace.session(session["id"])
                configuration = {
                    "project": str(project),
                    "permissions": current["permissions"],
                    "attachments": [],
                    "memory_enabled": current["memory_enabled"],
                    "memory_root": str(workspace._directory(session["id"]) / "memory"),
                }
                return workspace._tools(configuration, root, lambda *args: None)[0]

            with self.assertRaises(PermissionDenied):
                tools()["remember"]({"key": "note", "value": "context"})
            workspace.memory(session["id"], enabled=True)
            tools()["remember"]({"key": "note", "value": "context"})
            # A new storage/controller object reads the same durable memory.
            workspace = Workspace(root / "history")
            self.assertEqual(tools()["recall"]({"key": "note"}), "context")
            workspace.memory(session["id"], clear=True)
            with self.assertRaises(PermissionDenied):
                tools()["recall"]({"key": "note"})
            workspace.memory(session["id"], enabled=False)
            with self.assertRaises(PermissionDenied):
                tools()["remember"]({"key": "note", "value": "context"})

    async def test_session_project_attachment_interrupt_resume_history_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            (project / "README.md").write_text("project context")
            document = root / "note.txt"
            document.write_text("attached context")
            workspace = Workspace(root / "workspace")
            session = workspace.start("demo", project, profile="fixture-demo")
            attachment = workspace.attach(session["id"], document)
            interrupted = False

            def pause(stage):
                nonlocal interrupted
                if stage == "tool_completed" and not interrupted:
                    interrupted = True
                    raise asyncio.CancelledError()

            with self.assertRaises(asyncio.CancelledError):
                await workspace.execute(
                    session["id"], "read attachment", after_save=pause
                )
            before = workspace.inspect(session["id"])
            self.assertEqual(before["state"]["status"], "cancelled")
            self.assertEqual(before["state"]["tool_calls"], 1)
            self.assertIn(attachment["id"], str(before["configuration"]))
            completed = await workspace.resume(session["id"])
            self.assertEqual(completed["output"], "attached context")
            self.assertEqual(completed["tool_calls"], 1)
            self.assertEqual(completed["model_calls"], 2)
            configuration = workspace.export_config(session["id"])
            self.assertEqual(
                configuration, workspace.inspect(session["id"])["configuration"]
            )
            self.assertEqual(len(workspace.history(session["id"])), 1)
            workspace.memory(session["id"], enabled=True)
            workspace.memory(session["id"], clear=True)
            workspace.preferences(language="de", theme="dark")
            self.assertEqual(workspace.preferences()["language"], "de")
            workspace.forget(session["id"])
            self.assertEqual(workspace.sessions(), [])

    async def test_native_cancel_targets_owned_process_and_saves_checkpoint(self):
        import os
        import sys

        if not hasattr(os, "pidfd_open"):
            self.skipTest("Owned cancellation requires Linux pidfd")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            (project / "README.md").write_text("context")
            workspace = Workspace(root / "history")
            session = workspace.start("cancel", project, profile="fixture-demo")
            script = (
                "import asyncio,sys; from agent_lab.workspace_store import Workspace\n"
                "class Waiting:\n"
                " async def complete(self,*args): await asyncio.sleep(30)\n"
                "class Slow(Workspace):\n def _adapter(self,config): return Waiting()\n"
                "try: asyncio.run(Slow(sys.argv[1]).execute(sys.argv[2],'wait'))\n"
                "except KeyboardInterrupt: raise SystemExit(130)\n"
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                str(workspace.root),
                session["id"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                active = workspace._directory(session["id"]) / "active.json"
                for _ in range(100):
                    if active.exists():
                        break
                    await asyncio.sleep(0.05)
                else:
                    self.fail("Child did not create authoritative active record")
                self.assertTrue(workspace.cancel(session["id"])["cancel_requested"])
                await asyncio.wait_for(process.communicate(), timeout=5)
                self.assertEqual(process.returncode, 130)
                self.assertEqual(
                    workspace.inspect(session["id"])["state"]["status"], "cancelled"
                )
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
