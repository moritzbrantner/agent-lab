import copy
import tempfile
import unittest
from pathlib import Path

from agent_lab.benchmarks import benchmark, discover, fixture_adapter, load_task
from agent_lab.experiments import validate_result
from agent_lab.runtime import AgentConfig, Reply, SequenceAdapter, ToolCall


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_undeclared_tools_cannot_edit_evaluator_authority(self):
        task = load_task("development", "sum")
        candidate = SequenceAdapter(
            [
                Reply(
                    calls=(
                        ToolCall(
                            "write_file",
                            {
                                "path": "benchmarks/development/sum.json",
                                "content": "owned",
                            },
                        ),
                    )
                )
            ]
        )
        result = await benchmark(task, candidate, AgentConfig())
        self.assertEqual(result["correctness"]["status"], "error")
        self.assertEqual(load_task("development", "sum"), task)
        self.assertEqual(result["work"]["tool_calls"], 0)

    async def test_all_categories_across_fixture_configurations(self):
        tasks = discover("development")
        self.assertEqual(len({task["category"] for task in tasks}), 5)
        with tempfile.TemporaryDirectory() as directory:
            for task in tasks:
                for model in ("fixture-a", "fixture-b"):
                    result = await benchmark(
                        task,
                        fixture_adapter(task),
                        AgentConfig(model=model),
                        output_root=Path(directory),
                    )
                    validate_result(result)
                    self.assertEqual(
                        result["correctness"]["status"], "pass", task["id"]
                    )
                    self.assertEqual(task, load_task(task["split"], task["id"]))

    async def test_fixture_mutation_is_rejected(self):
        task = copy.deepcopy(discover("development")[0])
        task["expected"] = "changed"
        with self.assertRaises(ValueError):
            await benchmark(task, fixture_adapter(task), AgentConfig())

    def test_separate_held_out_discovery(self):
        dev = {task["id"] for task in discover("development")}
        held = {task["id"] for task in discover("held-out")}
        self.assertTrue(held)
        self.assertFalse(dev & held)
        with self.assertRaises(ValueError):
            load_task("development", "../held-out/held-sum")
