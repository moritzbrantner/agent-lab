import json
import unittest

from agent_lab.benchmarks import benchmark, load_task
from agent_lab.context import project_context
from agent_lab.experiments import canonical
from agent_lab.runtime import AgentConfig, AgentState, FunctionAdapter, Reply, ToolCall


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_incremental_memory_preserves_multistep_output_and_reduces_tokens(
        self,
    ):
        task = load_task("development", "batch-sum")
        results = []
        for policy in ("full", "retained"):

            def respond(messages, config):
                if config.context_policy == "retained":
                    completed = json.loads(messages[1]["content"].split("\n", 1)[1])[
                        "completed"
                    ]
                    values = [entry[2] for entry in completed]
                else:
                    values = [m["content"] for m in messages if m["role"] == "tool"]
                index = len(values)
                reply = (
                    Reply(
                        calls=(
                            ToolCall(
                                "sum", {"numbers": task["input"]["batches"][index]}
                            ),
                        )
                    )
                    if index < 3
                    else Reply(str(sum(values)))
                )
                return Reply(
                    reply.content, reply.calls, len(canonical(messages).encode()), 10
                )

            result = await benchmark(
                task,
                FunctionAdapter(respond),
                AgentConfig(
                    context_policy=policy, model_digest="byte-tokenizer-fixture-v1"
                ),
            )
            self.assertEqual(result["correctness"]["status"], "pass")
            results.append(result)
        self.assertLess(
            results[1]["work"]["input_tokens"], results[0]["work"]["input_tokens"]
        )

    def test_unknown_policy_and_durable_memory_are_not_implicit(self):
        with self.assertRaises(ValueError):
            AgentConfig(context_policy="unknown")
        state = AgentState("task", messages=[{"role": "user", "content": "task"}])
        self.assertEqual(project_context(state, "full"), state.messages)
        self.assertEqual(state.working_memory, [])
