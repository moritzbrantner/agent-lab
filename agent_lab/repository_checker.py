"""Trusted observation worker; candidate input is structurally bounded by parent."""

import asyncio
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path


def snapshot(state):
    value = asdict(state)
    return {
        k: value[k]
        for k in (
            "input_tokens",
            "output_tokens",
            "model_calls",
            "tool_calls",
            "status",
            "output",
        )
    }


async def observations(path):
    spec = importlib.util.spec_from_file_location("historical_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    results = {}

    def failed(*_):
        raise runtime.RuntimeFailure("unknown completion")

    state = runtime.AgentState("first unknown")
    try:
        await runtime.run(
            runtime.FunctionAdapter(failed), {}, runtime.AgentConfig(), state
        )
    except runtime.RuntimeFailure:
        pass
    results["first_unknown"] = snapshot(state)

    async def waiting(*_):
        await asyncio.sleep(30)

    class Waiting:
        complete = staticmethod(waiting)

    state = runtime.AgentState("cancel unknown")
    job = asyncio.create_task(runtime.run(Waiting(), {}, runtime.AgentConfig(), state))
    await asyncio.sleep(0)
    job.cancel()
    try:
        await job
    except asyncio.CancelledError:
        pass
    results["cancel_unknown"] = snapshot(state)
    state = runtime.AgentState("known response")
    await runtime.run(
        runtime.SequenceAdapter(
            [runtime.Reply("done", input_tokens=2, output_tokens=3)]
        ),
        {},
        runtime.AgentConfig(),
        state,
    )
    results["known_response"] = snapshot(state)
    state = runtime.AgentState("known tool failure")
    try:
        await runtime.run(
            runtime.SequenceAdapter(
                [
                    runtime.Reply(
                        calls=(runtime.ToolCall("fail", {}),),
                        input_tokens=2,
                        output_tokens=3,
                    )
                ]
            ),
            {"fail": failed},
            runtime.AgentConfig(),
            state,
        )
    except runtime.RuntimeFailure:
        pass
    results["known_tool_failure"] = snapshot(state)
    state = runtime.AgentState("later unknown")

    def later(messages, _):
        if messages[-1]["role"] == "tool":
            return failed()
        return runtime.Reply(
            calls=(runtime.ToolCall("echo", {}),), input_tokens=2, output_tokens=3
        )

    try:
        await runtime.run(
            runtime.FunctionAdapter(later),
            {"echo": lambda _: 1},
            runtime.AgentConfig(),
            state,
        )
    except runtime.RuntimeFailure:
        pass
    results["later_unknown"] = snapshot(state)
    return results


if __name__ == "__main__":
    print(json.dumps(asyncio.run(observations(Path(sys.argv[1]))), sort_keys=True))
