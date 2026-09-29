import json
import unittest
from pathlib import Path

from agent_lab.baselines import verify_bundle
from agent_lab.experiments import ROOT, validate_result
from agent_lab.trace import load_trace


class EvidenceTests(unittest.TestCase):
    def test_immutable_baselines_and_artifacts(self):
        root = ROOT / "evidence/baselines/local-v1"
        self.assertEqual(verify_bundle(root)["samples"], 120)
        for name in ("results.jsonl", "cold-results.jsonl"):
            for line in (root / name).read_text().splitlines():
                result = validate_result(json.loads(line))
                run_root = root / "runs" / result["run_id"]
                self.assertEqual(
                    json.loads((run_root / "result.json").read_text()), result
                )
                trace = load_trace(run_root / "trace.json", artifact_root=run_root)
                self.assertEqual(trace["artifacts"], result["artifacts"])
                self.assertFalse(trace["replayable"])
                self.assertFalse(list(Path(run_root).glob("*.content.json")))
