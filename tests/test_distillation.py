import copy
import tempfile
import unittest
from pathlib import Path

from agent_lab.distillation import distill, verify_artifact
from agent_lab.experiments import ROOT


class DistillationTests(unittest.TestCase):
    def test_verified_experience_becomes_provenanced_tool_and_examples(self):
        source = ROOT / "evidence/capabilities/local-v1/runs"
        paths = [p for p in source.iterdir() if (p / "result.json").is_file()]
        import json

        paths = [
            p
            for p in paths
            if json.loads((p / "result.json").read_text())["configuration"]["backend"]
            == "ollama"
        ]
        artifact = distill(paths[:2])
        self.assertEqual(len(artifact["examples"]), 2)
        self.assertEqual(verify_artifact(artifact).id, "batch-sum-v1")
        broken = copy.deepcopy(artifact)
        broken["provenance"][0]["result_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_artifact(broken)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                distill([Path(directory)])
