import json
import unittest

from agent_lab.experiments import ROOT, digest
from agent_lab.learning import build_data, entry_gate, validate_data


class LearningTests(unittest.TestCase):
    def test_verified_synthetic_data_excludes_held_out_material(self):
        artifact = json.loads(
            (
                ROOT / "evidence/distillation/local-v1/distilled-artifact.json"
            ).read_text()
        )
        data = build_data(artifact, count=16)
        self.assertEqual(len(data["examples"]), 16)
        self.assertTrue(data["source_artifact_sha256"])
        validate_data(data)
        held = {digest(example["input"]) for example in data["examples"][:1]}
        with self.assertRaises(ValueError):
            validate_data(data, held_out_hashes=held)
        data["examples"][0]["response"] = '{"content":"999","calls":[]}'
        data["data_sha256"] = digest(
            {k: v for k, v in data.items() if k != "data_sha256"}
        )
        with self.assertRaises(ValueError):
            validate_data(data)

    def test_learning_gate_requires_measured_independent_evidence(self):
        gate = entry_gate()
        self.assertTrue(gate["passed"])
        self.assertGreaterEqual(len(gate["held_out_tasks"]), 3)
