import asyncio
import unittest

from agent_lab.runtime import (
    AgentConfig,
    AgentState,
    FunctionAdapter,
    Reply,
    RuntimeFailure,
    SequenceAdapter,
    ToolCall,
    TransientModelError,
    run,
)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_interchangeable_adapters_and_explicit_state(self):
        replies = [Reply(calls=(ToolCall("sum", {"numbers": [2, 3]}),)), Reply("5")]
        for adapter in (
            SequenceAdapter(replies),
            FunctionAdapter(lambda messages, config: replies[len(messages) // 2]),
        ):
            state = AgentState("sum two numbers")
            events = []
            await run(
                adapter,
                {"sum": lambda args: sum(args["numbers"])},
                AgentConfig(),
                state,
                lambda kind, payload, events=events: events.append(kind),
            )
            self.assertEqual(state.output, "5")
            self.assertEqual(state.status, "completed")
            self.assertEqual(state.tool_calls, 1)
            self.assertIn("tool_result", events)

    async def test_budgets_and_nontransient_failure(self):
        state = AgentState("loop")
        adapter = FunctionAdapter(lambda *_: Reply(calls=(ToolCall("echo", {}),)))
        with self.assertRaises(RuntimeFailure):
            await run(adapter, {"echo": lambda _: 1}, AgentConfig(max_steps=2), state)
        self.assertEqual(state.model_calls, 2)
        self.assertEqual(state.status, "failed")
        state = AgentState("unknown tool")
        with self.assertRaises(RuntimeFailure):
            await run(
                SequenceAdapter([Reply(calls=(ToolCall("missing", {}),))]),
                {},
                AgentConfig(),
                state,
            )
        self.assertEqual(state.retries, 0)

    async def test_transient_retry_is_bounded(self):
        count = 0

        def respond(*_):
            nonlocal count
            count += 1
            raise TransientModelError("busy")

        state = AgentState("retry")
        with self.assertRaises(RuntimeFailure):
            await run(
                FunctionAdapter(respond),
                {},
                AgentConfig(max_retries=1, retry_delay=0),
                state,
            )
        self.assertEqual(count, 2)
        self.assertEqual(state.retries, 1)

    async def test_cancellation_propagates(self):
        class WaitingAdapter:
            async def complete(self, messages, config):
                await asyncio.sleep(100)

        state = AgentState("cancel")
        task = asyncio.create_task(run(WaitingAdapter(), {}, AgentConfig(), state))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(state.status, "cancelled")
