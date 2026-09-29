import copy
import unittest

from agent_lab.benchmarks import load_task
from agent_lab.evaluators import independent_evaluate
from agent_lab.oracles import IncrementalIndex, reference_sequence


class EvaluatorTests(unittest.TestCase):
    def test_candidate_cannot_redefine_correctness(self):
        task = load_task("development", "sum")
        self.assertEqual(independent_evaluate(task, "49")["status"], "pass")
        self.assertEqual(independent_evaluate(task, "50")["status"], "fail")
        changed = copy.deepcopy(task)
        changed["expected"] = "50"
        with self.assertRaises(ValueError):
            independent_evaluate(changed, "50")

    def test_serialization_and_order_are_observable(self):
        task = load_task("development", "sort-records")
        self.assertEqual(independent_evaluate(task, task["expected"])["status"], "pass")
        self.assertEqual(
            independent_evaluate(task, task["expected"] + "\n")["status"], "fail"
        )

    def test_incremental_matches_full_reference_after_every_change(self):
        changes = []
        for i in range(40):
            changes.append({"op": "put", "record": {"id": str(i % 7), "rank": i % 3}})
            if i % 4 == 0:
                changes.append({"op": "delete", "id": str((i + 1) % 7)})
        expected = reference_sequence(changes)
        index = IncrementalIndex()
        observed = [index.apply(change) for change in changes]
        self.assertEqual(observed, expected)
        self.assertEqual(
            IncrementalIndex().apply({"op": "delete", "id": "missing"}), []
        )
