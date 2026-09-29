import asyncio
import unittest

from agent_lab.routing import Requirements, RoutingAdapter
from agent_lab.runtime import (
    AgentConfig,
    AgentState,
    Reply,
    ResponseFailure,
    SequenceAdapter,
    ToolCall,
    run,
)


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_escalation_keeps_known_invocation_count(self):
        class Waiting:
            async def complete(self, messages, config):
                await asyncio.sleep(100)

        adapter = RoutingAdapter(
            SequenceAdapter([Reply("", input_tokens=1, output_tokens=2)]),
            AgentConfig(),
            Waiting(),
            AgentConfig(),
        )
        state = AgentState("test")
        job = asyncio.create_task(run(adapter, {}, adapter.configuration(), state))
        await asyncio.sleep(0)
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        self.assertEqual(state.model_calls, 2)
        self.assertIsNone(state.output_tokens)

    async def test_escalation_keeps_rejected_cost_and_stays_on_strong(self):
        adapter = RoutingAdapter(
            SequenceAdapter([Reply("", input_tokens=2, output_tokens=3)]),
            AgentConfig(model="cheap"),
            SequenceAdapter(
                [
                    Reply(
                        calls=(ToolCall("sum", {"numbers": [1, 2]}),),
                        input_tokens=5,
                        output_tokens=7,
                    ),
                    Reply("3", input_tokens=4, output_tokens=1),
                ]
            ),
            AgentConfig(model="strong"),
            Requirements(("sum",), 1, "integer"),
        )
        events = []
        state = AgentState("sum")
        await run(
            adapter,
            {"sum": lambda a: sum(a["numbers"])},
            adapter.configuration(),
            state,
            lambda kind, payload: events.append((kind, payload)),
        )
        self.assertEqual(state.output, "3")
        self.assertEqual(state.model_calls, 3)
        self.assertEqual(state.output_tokens, 11)
        self.assertEqual(adapter.escalations, 1)
        self.assertTrue(
            any(
                kind == "routing" and p["reason"] == "empty-reply" for kind, p in events
            )
        )

    async def test_escalation_and_call_budgets_are_hard_bounds(self):
        adapter = RoutingAdapter(
            SequenceAdapter([Reply("", input_tokens=1, output_tokens=2)]),
            AgentConfig(),
            SequenceAdapter([Reply("answer")]),
            AgentConfig(),
            Requirements(),
            max_escalations=0,
        )
        state = AgentState("test")
        with self.assertRaises(ResponseFailure):
            await run(adapter, {}, adapter.configuration(), state)
        self.assertEqual(state.model_calls, 1)
        self.assertEqual(state.output_tokens, 2)
        self.assertEqual(adapter.escalations, 0)
