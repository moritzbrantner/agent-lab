"""Bounded orchestration with explicit state and backend-independent messages."""

import asyncio
import copy
import inspect
from dataclasses import asdict, dataclass, field
from typing import Protocol

from agent_lab.context import project_context


class RuntimeFailure(Exception):
    """A permanent runtime or tool failure."""


class TransientModelError(RuntimeFailure):
    """An adapter declares that a request failed safely before completion."""


@dataclass(frozen=True)
class AgentConfig:
    model: str = "fixture"
    model_digest: str = "fixture-v1"
    backend: str = "fixture"
    backend_version: str = "1"
    quantization: str = "none"
    seed: int = 0
    temperature: float = 0
    context_size: int = 4096
    max_output_tokens: int = 256
    max_steps: int = 8
    max_retries: int = 2
    max_tool_calls: int = 16
    request_timeout: float = 60
    retry_delay: float = 0.1
    context_policy: str = "full"
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.context_policy not in ("full", "retained", "relevance"):
            raise ValueError("Unknown context policy")
        if min(self.max_steps, self.max_output_tokens, self.context_size) < 1:
            raise ValueError("Step, token and context budgets must be positive")
        if min(self.max_retries, self.max_tool_calls, self.retry_delay) < 0:
            raise ValueError("Retry/tool budgets and delay cannot be negative")
        if self.request_timeout <= 0:
            raise ValueError("Request timeout must be positive")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class Reply:
    content: str = ""
    calls: tuple[ToolCall, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    measurements: dict = field(default_factory=dict)
    model_calls: int = 1

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or not isinstance(value.get("content", ""), str):
            raise RuntimeFailure("Invalid model reply")
        calls = value.get("calls", [])
        if not isinstance(calls, list):
            raise RuntimeFailure("Invalid tool calls")
        parsed = []
        for call in calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("name"), str)
                or not isinstance(call.get("arguments"), dict)
            ):
                raise RuntimeFailure("Invalid tool call")
            parsed.append(ToolCall(call["name"], call["arguments"]))
        for key in ("input_tokens", "output_tokens"):
            count = value.get(key)
            if count is not None and (type(count) is not int or count < 0):
                raise RuntimeFailure("Invalid token count")
        if (
            type(value.get("model_calls", 1)) is not int
            or value.get("model_calls", 1) < 0
        ):
            raise RuntimeFailure("Invalid invocation count")
        return cls(
            value.get("content", ""),
            tuple(parsed),
            value.get("input_tokens"),
            value.get("output_tokens"),
            value.get("measurements", {}),
            value.get("model_calls", 1),
        )


class ResponseFailure(RuntimeFailure):
    """A completed but unusable response, retaining its consumed resources."""

    def __init__(
        self,
        message,
        *,
        input_tokens=None,
        output_tokens=None,
        measurements=None,
        model_calls=1,
    ):
        super().__init__(message)
        self.usage = Reply(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            measurements=measurements or {},
            model_calls=model_calls,
        )


class ModelAdapter(Protocol):
    async def complete(self, messages: list[dict], config: AgentConfig) -> Reply: ...


class SequenceAdapter:
    """Recorded deterministic responses, owned by one run."""

    def __init__(self, replies):
        self.replies = iter(replies)

    async def complete(self, messages, config):
        try:
            return next(self.replies)
        except StopIteration as error:
            raise RuntimeFailure("Fixture responses exhausted") from error


class FunctionAdapter:
    """A deterministic fixture driven by observable messages."""

    def __init__(self, respond):
        self.respond = respond

    async def complete(self, messages, config):
        return self.respond(messages, config)


@dataclass
class AgentState:
    task: str
    messages: list[dict] = field(default_factory=list)
    status: str = "pending"
    output: str = ""
    steps: int = 0
    model_calls: int = 0
    retries: int = 0
    tool_calls: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    pending_calls: list[dict] = field(default_factory=list)
    working_memory: list[dict] = field(default_factory=list)


def _ignore_event(kind, payload):
    """Default hook intentionally retains no content."""


async def run(adapter, tools, config, state, emit=_ignore_event):
    """Run until completion; failures/cancellation propagate with retained state.

    Tools are trusted callables, not a sandbox. Sync tools must be finite local
    computations; process/network tools must implement async cancellation bounds.
    """
    observer = getattr(adapter, "observe", None)
    if observer is not None:
        observer(emit)
    if not state.messages:
        state.messages.append({"role": "user", "content": state.task})
    state.status = "running"
    emit("state", {"status": state.status, "configuration": asdict(config)})
    try:
        while state.steps < config.max_steps:
            messages = project_context(state, config.context_policy)
            emit(
                "context",
                {
                    "policy": config.context_policy,
                    "message_count": len(messages),
                    "retained_tools": len(state.working_memory),
                },
            )
            for attempt in range(config.max_retries + 1):
                request_invocations = getattr(adapter, "invocation_count", None)
                state.model_calls += 1
                emit("model_request", {"messages": copy.deepcopy(messages)})
                try:
                    async with asyncio.timeout(config.request_timeout):
                        reply = await adapter.complete(copy.deepcopy(messages), config)
                    break
                except TransientModelError as error:
                    if state.retries >= config.max_retries:
                        raise RuntimeFailure("Retry budget exhausted") from error
                    state.retries += 1
                    emit("retry", {"reason": str(error), "attempt": attempt + 1})
                    await asyncio.sleep(config.retry_delay * 2**attempt)
                except ResponseFailure as error:
                    state.steps += 1
                    state.model_calls += error.usage.model_calls - 1
                    emit("model_response", {**asdict(error.usage), "failure": True})
                    for name in ("input_tokens", "output_tokens"):
                        old, new = getattr(state, name), getattr(error.usage, name)
                        setattr(
                            state,
                            name,
                            old + new if old is not None and new is not None else None,
                        )
                    raise
                except Exception, asyncio.CancelledError:
                    if request_invocations is not None:
                        state.model_calls += (
                            adapter.invocation_count - request_invocations - 1
                        )
                    state.input_tokens = state.output_tokens = None
                    raise
            state.steps += 1
            state.model_calls += reply.model_calls - 1
            emit("model_response", asdict(reply))
            for name in ("input_tokens", "output_tokens"):
                old, new = getattr(state, name), getattr(reply, name)
                setattr(
                    state,
                    name,
                    old + new if old is not None and new is not None else None,
                )
            state.messages.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "calls": [asdict(call) for call in reply.calls],
                }
            )
            if not reply.calls:
                state.output = reply.content
                state.status = "completed"
                emit("state", {"status": state.status})
                return state
            if state.tool_calls + len(reply.calls) > config.max_tool_calls:
                raise RuntimeFailure("Tool budget exhausted")
            for call in reply.calls:
                if call.name not in tools:
                    raise RuntimeFailure(f"Unknown tool: {call.name}")
                emit("tool_request", asdict(call))
                value = tools[call.name](copy.deepcopy(call.arguments))
                if inspect.isawaitable(value):
                    async with asyncio.timeout(config.request_timeout):
                        value = await value
                state.tool_calls += 1
                state.working_memory.append(
                    {
                        "name": call.name,
                        "arguments": copy.deepcopy(call.arguments),
                        "result": copy.deepcopy(value),
                    }
                )
                state.messages.append(
                    {"role": "tool", "name": call.name, "content": value}
                )
                emit("tool_result", {"name": call.name, "value": value})
        raise RuntimeFailure("Step budget exhausted")
    except asyncio.CancelledError:
        state.status = "cancelled"
        emit("state", {"status": state.status})
        raise
    except Exception as error:
        # The run boundary retains failures; tools are never silently retried.
        state.status = "failed"
        emit("failure", {"category": type(error).__name__})
        raise
