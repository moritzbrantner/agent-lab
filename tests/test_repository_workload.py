import shutil
import tempfile
import unittest
from pathlib import Path

from agent_lab.experiments import ROOT
from agent_lab.permissions import SandboxUnavailable
from agent_lab.repository_workload import (
    check_patch,
    reference_patch,
    safe_patch,
)


class RepositoryWorkloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_historical_patch_has_independent_negative_and_positive_controls(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ROOT / "benchmarks/repositories/unknown-usage-v1"
            shutil.copytree(fixture, root, dirs_exist_ok=True)
            original = (root / "runtime.py").read_text()
            try:
                before = await check_patch(root, allow_baseline=True)
            except SandboxUnavailable as error:
                self.skipTest(str(error))
            self.assertEqual(before["status"], "fail")
            (root / "runtime.py").write_text(reference_patch(original))
            self.assertTrue(safe_patch(original, (root / "runtime.py").read_text()))
            self.assertEqual((await check_patch(root))["status"], "pass")
            (root / "runtime.py").write_text(
                reference_patch(original) + '\nprint("owned")\n'
            )
            self.assertEqual((await check_patch(root))["status"], "fail")

    def test_fixture_source_and_patch_authority_are_immutable(self):
        original = (
            ROOT / "benchmarks/repositories/unknown-usage-v1/runtime.py"
        ).read_text()
        patch = reference_patch(original)
        self.assertFalse(
            safe_patch(
                original,
                patch.replace(
                    "raise\n            state.steps",
                    'print("fake")\n            state.steps',
                ),
            )
        )
        self.assertFalse(safe_patch(original, patch + "\nimport os\n"))
