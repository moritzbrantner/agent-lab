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

    def test_generation_evidence_retains_every_candidate(self):
        from agent_lab.experiments import digest

        root = ROOT / "evidence/generation/local-v1"
        results = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            digest(results),
            json.loads((root / "manifest.json").read_text())["results_sha256"],
        )
        self.assertEqual(len(results), 10)
        for label in ("single", "regenerate"):
            for repeat in range(6):
                run_root = root / label / str(repeat)
                report = json.loads((run_root / "report.json").read_text())
                result = validate_result(report["result"])
                load_trace(run_root / "trace.json", artifact_root=run_root)
                self.assertEqual(
                    result["work"]["generated_tokens"],
                    sum(c["work"]["generated_tokens"] for c in report["candidates"]),
                )
                for candidate in report["candidates"]:
                    child = run_root / "candidates" / candidate["run_id"]
                    trace = load_trace(child / "trace.json", artifact_root=child)
                    self.assertEqual(trace["parent_run_id"], result["run_id"])

    def test_inference_features_have_comparable_evidence(self):
        from agent_lab.experiments import digest

        root = ROOT / "evidence/inference/local-v1"
        results = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(results), 25)
        self.assertEqual(
            digest(results),
            json.loads((root / "manifest.json").read_text())["results_sha256"],
        )
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(
            set(report["comparisons"]),
            {"small-q4-gpu", "small-q8-gpu", "medium-context-2048"},
        )
        self.assertEqual(report["negotiation"]["speculation"]["status"], "unavailable")
        for result in results:
            validate_result(result)
            run_root = root / "runs" / result["run_id"]
            load_trace(run_root / "trace.json", artifact_root=run_root)

    def test_self_proposed_mutation_is_auditable_and_guarded(self):
        from agent_lab.experiments import digest
        from agent_lab.improvement import decide

        root = ROOT / "evidence/improvement/local-v1"
        results = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(results), 70)
        self.assertEqual(
            digest(results),
            json.loads((root / "manifest.json").read_text())["results_sha256"],
        )
        proposal = json.loads((root / "proposal.json").read_text())
        self.assertEqual(
            proposal["mutation"]["changes"], {"context_policy": "relevance"}
        )
        self.assertGreater(proposal["work"]["output_tokens"], 0)
        self.assertEqual(
            decide(results[:35], results[35:]),
            json.loads((root / "decision.json").read_text()),
        )
        load_trace(root / "proposal-trace.json", artifact_root=root)
        for result in results:
            run_root = root / "runs" / result["run_id"]
            load_trace(run_root / "trace.json", artifact_root=run_root)

    def test_evolutionary_archive_preserves_measured_rejection(self):
        from agent_lab.archive import Archive

        root = ROOT / "evidence/archive/local-v1"
        report = json.loads((root / "report.json").read_text())
        archive = Archive(root)
        self.assertEqual(archive.read(report["candidate"])["parents"], [report["root"]])
        self.assertEqual(
            archive.read(report["candidate"])["decision"]["status"], "reject"
        )
        self.assertEqual(archive.frontier(), report["frontier"])
        self.assertEqual(report["frontier"][0]["entries"], [])

    def test_distilled_artifact_provenance_and_cost_parity(self):
        from agent_lab.distillation import verify_artifact
        from agent_lab.experiments import digest

        root = ROOT / "evidence/distillation/local-v1"
        artifact = json.loads((root / "distilled-artifact.json").read_text())
        verify_artifact(artifact)
        report = json.loads((root / "report.json").read_text())
        self.assertTrue(report["correctness_parity"])
        self.assertEqual(report["capability"]["generated_tokens"]["mean"], 0)
        self.assertGreater(report["reference"]["generated_tokens"]["mean"], 0)
        results = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            digest(results),
            json.loads((root / "manifest.json").read_text())["results_sha256"],
        )
        for result in results:
            run_root = root / "runs" / result["run_id"]
            load_trace(run_root / "trace.json", artifact_root=run_root)

    def test_learned_candidate_retains_real_weights_and_separate_cost(self):
        import struct

        from agent_lab.experiments import digest
        from agent_lab.learning import validate_data

        root = ROOT / "evidence/learning/local-v1"
        data = json.loads((root / "training-data.json").read_text())
        validate_data(data)
        training = json.loads((root / "training-cost.json").read_text())
        self.assertEqual(training["optimizer_steps"], 64)
        self.assertEqual(len(training["losses"]), 64)
        self.assertGreater(training["trainable_parameters"], 0)
        self.assertGreater(training["measurements"]["wall_seconds"], 0)
        weights = (root / "adapter/adapter_model.safetensors").read_bytes()
        length = struct.unpack("<Q", weights[:8])[0]
        header = json.loads(weights[8 : 8 + length])
        payload = weights[8 + length :]
        self.assertTrue(
            any(
                any(payload[value["data_offsets"][0] : value["data_offsets"][1]])
                for key, value in header.items()
                if "lora_B" in key
            )
        )
        results = [
            json.loads(line)
            for line in (root / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(results), 50)
        self.assertEqual(
            digest(results),
            json.loads((root / "manifest.json").read_text())["results_sha256"],
        )
        self.assertEqual(
            {r["protocol"]["split"] for r in results}, {"development", "held-out"}
        )
        report = json.loads((root / "report.json").read_text())
        self.assertEqual(report["decision"]["status"], "reject")
        load_trace(root / "trace.json", artifact_root=root)
        for result in results:
            run_root = root / "runs" / result["run_id"]
            load_trace(run_root / "trace.json", artifact_root=run_root)

    def test_permission_fixture_is_attributed_and_secret_free(self):
        root = ROOT / "evidence/permissions/local-v1"
        report = json.loads((root / "report.json").read_text())
        self.assertTrue(all(report[key] for key in ("allowed", "denied", "elevated")))
        self.assertEqual(report["sandbox"]["stdout"].splitlines(), ["False", "True"])
        trace = load_trace(root / "trace.json", artifact_root=root)
        authorities = [
            e["metadata"] for e in trace["events"] if e["kind"] == "authority"
        ]
        self.assertTrue(any(e.get("allowed") is False for e in authorities))
        self.assertTrue(any(e.get("action") == "elevation" for e in authorities))
        self.assertNotIn(
            "fixture-secret-do-not-retain", (root / "trace.json").read_text()
        )

    def test_real_repository_patch_evidence_and_replay(self):
        from agent_lab.experiments import digest
        from agent_lab.runtime import AgentConfig, AgentState, run
        from agent_lab.trace import Replay

        root = ROOT / "evidence/repository/local-v1"
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
        self.assertEqual(report["reference-fixture"]["passed"], 5)
        for result in results:
            run_root = root / "runs" / result["run_id"]
            trace = load_trace(run_root / "trace.json", artifact_root=run_root)
            verification = json.loads((run_root / "verification.json").read_text())
            self.assertEqual(verification["before"]["status"], "fail")
            if result["configuration"]["backend"] == "fixture":
                replay = Replay(trace)
                state = AgentState(
                    "replay task uses recorded configuration and request"
                )
                first = next(e for e in trace["events"] if e["kind"] == "model_request")
                state.task = trace["payloads"][first["content_ref"]]["messages"][0][
                    "content"
                ]
                import asyncio

                asyncio.run(
                    run(
                        replay,
                        replay.tools(),
                        AgentConfig(**result["configuration"]),
                        state,
                    )
                )
                replay.assert_consumed()
                self.assertEqual(state.output, verification["agent_output"])
                self.assertEqual(state.tool_calls, result["work"]["tool_calls"])
