import unittest
from dataclasses import replace

from test_experiments import result as sample

from agent_lab.improvement import decide, materialize
from agent_lab.runtime import AgentConfig


class ImprovementTests(unittest.TestCase):
    def test_mutation_surface_excludes_authority(self):
        baseline = AgentConfig()
        self.assertEqual(
            materialize(
                baseline,
                {
                    "changes": {"context_policy": "relevance"},
                    "reason": "reduce irrelevant facts",
                },
            ),
            replace(baseline, context_policy="relevance"),
        )
        for changes in (
            {"evaluator": "always-pass"},
            {"max_retries": 100},
            {"feedback": "x" * 2000},
            {"context_policy": "unknown"},
        ):
            with self.assertRaises(ValueError):
                materialize(baseline, {"changes": changes, "reason": "test"})

    def test_held_out_regression_rejects_development_improvement(self):
        def result(split, score, tokens):
            value = sample(tokens)
            value["task"]["id"] = split
            value["protocol"]["split"] = split
            value["correctness"].update(
                status="pass" if score == 1 else "fail", score=score
            )
            return value

        baseline = [result("development", 0, 10), result("held-out", 1, 10)]
        candidate = [result("development", 1, 1), result("held-out", 0, 1)]
        self.assertEqual(decide(baseline, candidate)["status"], "reject")
        candidate[1]["correctness"].update(status="pass", score=1)
        self.assertEqual(decide(baseline, candidate)["status"], "accept")

    def test_proposal_view_refuses_held_out_results(self):
        from agent_lab.improvement import development_view

        value = sample()
        value["protocol"]["split"] = "held-out"
        with self.assertRaises(ValueError):
            development_view([value])


class ProposalTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_has_only_a_bounded_proposal_tool(self):
        from agent_lab.experiments import digest
        from agent_lab.improvement import propose
        from agent_lab.runtime import Reply, SequenceAdapter, ToolCall

        config = AgentConfig()
        change = {"changes": {"context_policy": "relevance"}, "reason": "less context"}
        adapter = SequenceAdapter(
            [
                Reply(
                    calls=(ToolCall("propose", change),),
                    input_tokens=1,
                    output_tokens=2,
                ),
                Reply("done", input_tokens=1, output_tokens=2),
            ]
        )
        mutation, trace, work = await propose(adapter, config, [sample()])
        self.assertEqual(mutation, change)
        self.assertEqual(work["output_tokens"], 4)
        from dataclasses import asdict

        self.assertEqual(
            trace.identity["configuration_id"],
            digest(asdict(replace(config, max_steps=2, max_tool_calls=1))),
        )
