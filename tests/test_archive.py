import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from test_experiments import result

from agent_lab.archive import Archive
from agent_lab.runtime import AgentConfig


def cohort(tokens, wall, held_score=1, development_score=1, config=None):
    runs = []
    for split, score in (("development", development_score), ("held-out", held_score)):
        sample = result(tokens)
        sample["protocol"]["split"] = split
        sample["configuration"] = asdict(config or AgentConfig())
        sample["correctness"].update(
            status="pass" if score == 1 else "fail", score=score
        )
        sample["measurements"].update(
            wall_seconds=wall, peak_ram_bytes=100, peak_vram_bytes=0
        )
        sample["measurement_sources"].update(
            wall_seconds="fixture", peak_ram_bytes="fixture", peak_vram_bytes="fixture"
        )
        runs.append(sample)
    return runs


class ArchiveTests(unittest.TestCase):
    def test_multiple_lineages_tradeoffs_and_rejected_regression_survive_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Archive(Path(directory))
            root = archive.add(
                AgentConfig(), cohort(20, 20, development_score=0), mutation="baseline"
            )
            a = archive.add(
                replace(AgentConfig(), feedback="a"),
                cohort(5, 10, config=replace(AgentConfig(), feedback="a")),
                parents=[root],
                mutation="fewer tokens",
            )
            b = archive.add(
                replace(AgentConfig(), feedback="b"),
                cohort(10, 5, config=replace(AgentConfig(), feedback="b")),
                parents=[root],
                mutation="lower latency",
            )
            bad = archive.add(
                replace(AgentConfig(), feedback="bad"),
                cohort(
                    1, 1, held_score=0, config=replace(AgentConfig(), feedback="bad")
                ),
                parents=[root],
                mutation="overfit",
            )
            restored = Archive(Path(directory))
            self.assertEqual(set(restored.frontier()[0]["entries"]), {a, b})
            self.assertEqual(restored.read(bad)["decision"]["status"], "reject")
            self.assertEqual(restored.read(a)["parents"], [root])
            self.assertTrue(restored.regression_due(interval=3))
            restored.record_regression(
                a,
                cohort(
                    5, 10, held_score=0, config=replace(AgentConfig(), feedback="a")
                ),
                suite_id="broader-v1",
            )
            self.assertNotIn(a, restored.frontier()[0]["entries"])
            self.assertEqual(len(restored.entries()), 4)


class PeriodicRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_survivors_receive_protected_broader_fixture_evaluation(self):
        from agent_lab.benchmarks import benchmark, discover, fixture_adapter
        from agent_lab.telemetry import Telemetry

        with tempfile.TemporaryDirectory() as directory:
            archive = Archive(Path(directory) / "archive")
            config = AgentConfig()
            environment = {
                "hardware": {"profile": "fixture"},
                "software": {"runtime": "fixture"},
            }
            initial = [
                await benchmark(
                    task,
                    fixture_adapter(task),
                    config,
                    telemetry=Telemetry(periodic=False, memory_probe=lambda: (100, 0)),
                    **environment,
                )
                for task in [discover("development")[0], discover("held-out")[0]]
            ]
            archive.add(config, initial, mutation="fixture root")
            reports = await archive.evaluate_survivors(
                lambda _, task: fixture_adapter(task),
                discover("held-out")[1:],
                output_root=Path(directory) / "runs",
                interval=1,
                **environment,
            )
            self.assertEqual(len(reports), 1)
            self.assertTrue(reports[0]["passed"])
            self.assertFalse(archive.regression_due(interval=1))
