import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from agent_lab.workspace import COMMANDS, main
from agent_lab.workspace_locale import TEXT


class WorkspaceCliTests(unittest.TestCase):
    def test_localized_command_discovery_and_machine_contract(self):
        self.assertEqual(set(TEXT["en"]), set(TEXT["de"]))
        self.assertEqual(set(TEXT["en"]), set(TEXT["es"]))
        self.assertTrue(set(COMMANDS).issubset(TEXT["en"]))
        with tempfile.TemporaryDirectory() as directory:
            for language in ("en", "de", "es"):
                stream = io.StringIO()
                with (
                    contextlib.redirect_stdout(stream),
                    self.assertRaises(SystemExit) as result,
                ):
                    main(["--root", directory, "--language", language, "--help"])
                self.assertEqual(result.exception.code, 0)
                self.assertIn(TEXT[language]["title"], stream.getvalue())
            project = Path(directory) / "project"
            project.mkdir()
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                status = main(
                    [
                        "--root",
                        str(Path(directory) / "history"),
                        "--json",
                        "start",
                        "demo",
                        str(project),
                        "--profile",
                        "fixture-demo",
                    ]
                )
            self.assertEqual(status, 0)
            import json

            self.assertEqual(json.loads(stream.getvalue())["profile"], "fixture-demo")
