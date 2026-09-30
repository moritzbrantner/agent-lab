import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_experiments import result

from agent_lab.experiments import ROOT, digest
from agent_lab.permissions import SandboxUnavailable
from agent_lab.release import release_report
from agent_lab.release_experiment import default_profile, scaffold_proofs


def cohort():
    profile = json.loads(
        (ROOT / "configurations/release-thresholds-v1.json").read_text()
    )
    software = {"runtime": "controlled release test"}
    runs = []
    for workload in profile["workloads"]:
        for repeat in range(5):
            sample = result()
            sample.update(
                run_id=workload["task"]["id"] + "-" + str(repeat),
                task=copy.deepcopy(workload["task"]),
                protocol={**workload["protocol"], "repeat": repeat},
                configuration={**workload["configuration"], "seed": repeat},
                hardware=copy.deepcopy(profile["hardware"]),
                software=software,
            )
            sample["correctness"]["evaluator"] = workload["protocol"]["id"]
            sample["work"]["tool_calls"] = 1
            for name, value in (
                ("wall_seconds", 1),
                ("peak_ram_bytes", 1024),
                ("peak_vram_bytes", 1024),
            ):
                sample["measurements"][name] = value
                sample["measurement_sources"][name] = "controlled test observation"
            runs.append(sample)
    proofs = {
        "profile_sha256": digest(profile),
        "hardware": profile["hardware"],
        "software": software,
        "configuration": profile["configuration"],
        "values": {
            t["field"]: {
                "value": 2048 if t["field"] == "effective_context_tokens" else 1,
                "scope": "controlled test",
            }
            for t in profile["thresholds"]
            if t["source"] == "proofs"
        },
    }
    return profile, runs, proofs, software


class ReleaseTests(unittest.TestCase):
    def test_complete_capability_can_pass_with_optional_energy_unavailable(self):
        profile, runs, proofs, software = cohort()
        self.assertEqual(profile, default_profile())
        report = release_report(profile, runs, proofs, software)
        self.assertTrue(report["ready"])
        energy = next(t for t in report["thresholds"] if t["id"] == "energy")
        self.assertEqual(energy["status"], "unavailable")
        self.assertTrue(all(d["value"] is None for d in energy["details"]))

    def test_each_held_out_task_counts_errors_as_unsuccessful(self):
        profile, runs, proofs, software = cohort()
        failing = next(s for s in runs if s["task"]["id"] == "held-sum")
        failing["correctness"] = {
            "status": "error",
            "score": None,
            "evaluator": "exact-v1",
        }
        report = release_report(profile, runs, proofs, software)
        self.assertFalse(report["ready"])
        held = next(t for t in report["thresholds"] if t["id"] == "held-out")
        self.assertEqual(held["status"], "fail")
        self.assertEqual(held["details"][0]["value"], 0.8)

    def test_partial_telemetry_or_insufficient_repeats_is_unavailable(self):
        profile, runs, proofs, software = cohort()
        runs[0]["measurements"]["wall_seconds"] = None
        report = release_report(profile, runs, proofs, software)
        latency = next(
            t for t in report["thresholds"] if t["id"] == "interactive-latency"
        )
        self.assertEqual(latency["status"], "unavailable")
        self.assertFalse(report["ready"])
        runs.pop()
        report = release_report(profile, runs, proofs, software)
        coding = next(t for t in report["thresholds"] if t["id"] == "coding")
        self.assertEqual(coding["status"], "unavailable")

    def test_environment_config_and_duplicate_samples_cannot_claim_a_release(self):
        for key in ("hardware", "software", "configuration"):
            profile, runs, proofs, software = cohort()
            runs[0][key] = {"different": True}
            with self.assertRaises(ValueError):
                release_report(profile, runs, proofs, software)
        profile, runs, proofs, software = cohort()
        with self.assertRaises(ValueError):
            release_report(profile, [*runs, copy.deepcopy(runs[0])], proofs, software)
        proofs["profile_sha256"] = "other"
        with self.assertRaises(ValueError):
            release_report(profile, runs, proofs, software)

    def test_profile_cannot_grade_another_model_as_the_declared_candidate(self):
        profile, runs, proofs, software = cohort()
        profile["configuration"]["model_digest"] = "b" * 64
        proofs["profile_sha256"] = digest(profile)
        proofs["configuration"] = profile["configuration"]
        with self.assertRaises(ValueError):
            release_report(profile, runs, proofs, software)


class ScaffoldProofTests(unittest.IsolatedAsyncioTestCase):
    async def test_controlled_recovery_and_missing_native_backend_are_explicit(self):
        profile, _, _, software = cohort()

        async def unavailable(_):
            raise SandboxUnavailable("Reviewed native backend unavailable")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agent_lab.release_experiment.permission_fixture", unavailable),
        ):
            proofs = await scaffold_proofs(
                profile, Path(directory) / "proofs", software
            )
            for name in (
                "recovery",
                "bounded-resume",
                "ambiguous-action",
                "durable-memory",
            ):
                self.assertEqual(proofs["values"][name]["value"], 1)
            self.assertIsNone(proofs["values"]["offline"]["value"])
            self.assertIsNone(proofs["values"]["effective_context_tokens"]["value"])
            checkpoint = json.loads(
                (Path(directory) / "proofs/budget.json").read_text()
            )
            self.assertEqual(checkpoint["state"]["request_attempts"], 10)
            self.assertIsNone(checkpoint["state"]["input_tokens"])
