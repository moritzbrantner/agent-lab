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

    def test_recorded_context_experiment_preserves_negative_results(self):
        from agent_lab.context_experiment import context_report
        from agent_lab.experiments import digest

        root = ROOT / "evidence/context/local-v1"
        results = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            digest(results),
            json.loads((root / "manifest.json").read_text())["results_sha256"],
        )
        report = context_report(results)
        self.assertEqual(report, json.loads((root / "report.json").read_text()))
        self.assertFalse(report[0]["correctness_parity"])
        self.assertEqual(report[1]["candidate"]["score"]["mean"], 1)
        for result in [*results, json.loads((root / "cold-result.json").read_text())]:
            run_root = root / "runs" / result["run_id"]
            self.assertEqual(
                load_trace(run_root / "trace.json", artifact_root=run_root)[
                    "artifacts"
                ],
                result["artifacts"],
            )

    def test_reusable_capability_evidence(self):
        from agent_lab.experiments import digest

        root = ROOT / "evidence/capabilities/local-v1"
        results = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(results), 10)
        self.assertEqual(
            digest(results),
            json.loads((root / "manifest.json").read_text())["results_sha256"],
        )
        report = json.loads((root / "report.json").read_text())
        self.assertTrue(report["correctness_parity"])
        self.assertEqual(report["capability"]["generated_tokens"]["mean"], 0)
        self.assertGreater(report["reference"]["generated_tokens"]["mean"], 0)
        for result in results:
            run_root = root / "runs" / result["run_id"]
            load_trace(run_root / "trace.json", artifact_root=run_root)

    def test_routing_evidence_accounts_for_rejected_attempts(self):
        from agent_lab.experiments import digest

        root = ROOT / "evidence/routing/local-v1"
        results = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text().splitlines()
        ]
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(digest(results), manifest["results_sha256"])
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["cheap-first"]["passed"], 5)
        self.assertGreater(
            report["cheap-first"]["summary"]["generated_tokens"]["mean"],
            report["single-strong"]["summary"]["generated_tokens"]["mean"],
        )
        for result in results:
            run_root = root / "runs" / result["run_id"]
            trace = load_trace(run_root / "trace.json", artifact_root=run_root)
            if result["configuration"]["backend"] == "routing":
                requests = [
                    e
                    for e in trace["events"]
                    if e["kind"] == "child"
                    and e["metadata"].get("kind") == "model_request"
                ]
                self.assertEqual(len(requests), result["work"]["model_calls"])
