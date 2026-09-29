"""Bounded cheap-first/role routing with observable decisions and complete work."""

import re
from dataclasses import asdict, dataclass, replace

from agent_lab.experiments import digest
from agent_lab.runtime import AgentConfig, Reply, ResponseFailure, RuntimeFailure


@dataclass(frozen=True)
class Requirements:
    tools: tuple[str, ...] = ()
    minimum_tool_results: int = 0
    output_kind: str = "text"

    def completed(self, messages):
        return sum(message["role"] == "tool" for message in messages)

    def reject(self, reply, messages):
        if any(call.name not in self.tools for call in reply.calls):
            return "unknown-tool"
        if not reply.calls:
            if not reply.content.strip():
                return "empty-reply"
            if self.completed(messages) < self.minimum_tool_results:
                return "incomplete-required-tools"
            if self.output_kind == "integer" and not re.fullmatch(
                r"-?[0-9]+", reply.content
            ):
                return "invalid-answer-shape"
        return None


def combine(replies, final=None):
    final = final or replies[-1]
    totals = {}
    for name in ("input_tokens", "output_tokens"):
        values = [getattr(reply, name) for reply in replies]
        totals[name] = sum(values) if all(v is not None for v in values) else None
    keys = (
        set.intersection(*(set(reply.measurements) for reply in replies))
        if replies
        else set()
    )
    measurements = {
        key: sum(reply.measurements[key] for reply in replies) for key in keys
    }
    return Reply(
        final.content,
        final.calls,
        totals["input_tokens"],
        totals["output_tokens"],
        measurements,
        sum(reply.model_calls for reply in replies),
    )


class RoutingAdapter:
    def __init__(
        self,
        cheap,
        cheap_config,
        strong,
        strong_config,
        requirements=None,
        *,
        max_escalations=1,
        max_calls=16,
        mode="cheap-first",
    ):
        if (
            max_escalations < 0
            or max_calls < 1
            or mode not in ("cheap-first", "role-specialized")
        ):
            raise ValueError("Invalid routing policy")
        requirements = requirements or Requirements()
        self.adapters = {"cheap": cheap, "strong": strong}
        self.configs = {"cheap": cheap_config, "strong": strong_config}
        self.requirements = requirements
        self.max_escalations, self.max_calls, self.mode = (
            max_escalations,
            max_calls,
            mode,
        )
        self.escalations = self.calls = 0
        self.forced_strong = False
        self.emit = lambda kind, payload: None
        self.policy = {
            "models": {k: asdict(v) for k, v in self.configs.items()},
            "requirements": asdict(requirements),
            "mode": mode,
            "max_escalations": max_escalations,
            "max_calls": max_calls,
        }
        self.identity = digest(self.policy)

    @property
    def invocation_count(self):
        return self.calls

    def observe(self, emit):
        self.emit = emit

    def configuration(self, **overrides):
        return AgentConfig(
            model="routing-v1",
            model_digest=self.identity,
            backend="routing",
            quantization="mixed",
            options=self.policy,
            **overrides,
        )

    async def _attempt(self, tier, messages, outer):
        if self.calls >= self.max_calls:
            raise ResponseFailure(
                "Routing call budget exhausted",
                input_tokens=0,
                output_tokens=0,
                model_calls=0,
            )
        self.calls += 1
        config = replace(
            self.configs[tier],
            seed=outer.seed,
            request_timeout=min(
                outer.request_timeout, self.configs[tier].request_timeout
            ),
        )
        child = f"routing:{self.calls}"
        self.emit(
            "child",
            {
                "child_id": child,
                "stage": tier,
                "model_digest": config.model_digest,
                "kind": "model_request",
                "messages": messages,
                "configuration": asdict(config),
            },
        )
        try:
            reply = await self.adapters[tier].complete(messages, config)
        except ResponseFailure as error:
            self.emit(
                "child",
                {
                    "child_id": child,
                    "stage": tier,
                    "kind": "model_response",
                    "failure": True,
                    "usage": asdict(error.usage),
                },
            )
            raise
        except RuntimeFailure as error:
            raise ResponseFailure("Backend completion unknown") from error
        self.emit(
            "child",
            {
                "child_id": child,
                "stage": tier,
                "kind": "model_response",
                "reply": asdict(reply),
            },
        )
        return reply

    async def complete(self, messages, config):
        if config.model_digest != self.identity or config.backend != "routing":
            raise ResponseFailure(
                "Routing identity mismatch",
                input_tokens=0,
                output_tokens=0,
                model_calls=0,
            )
        completed = self.requirements.completed(messages)
        tier = (
            "strong"
            if (
                self.forced_strong
                or (
                    self.mode == "role-specialized"
                    and completed < self.requirements.minimum_tool_results
                )
            )
            else "cheap"
        )
        self.emit(
            "routing",
            {
                "stage": tier,
                "reason": "policy-selection",
                "completed_tools": completed,
                "model_digest": self.configs[tier].model_digest,
                "budget_remaining": self.max_calls - self.calls,
            },
        )
        replies = []
        try:
            reply = await self._attempt(tier, messages, config)
            replies.append(reply)
            reason = self.requirements.reject(reply, messages)
        except ResponseFailure as error:
            replies.append(error.usage)
            reason = "malformed-or-failed-response"
        if reason is None:
            return combine(replies)
        if tier == "strong" or self.escalations >= self.max_escalations:
            usage = combine(replies)
            raise ResponseFailure(
                "Routing validation/escalation budget failed",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                measurements=usage.measurements,
                model_calls=usage.model_calls,
            )
        self.escalations += 1
        self.forced_strong = True
        self.emit(
            "routing",
            {
                "stage": "strong",
                "reason": reason,
                "completed_tools": completed,
                "model_digest": self.configs["strong"].model_digest,
                "budget_remaining": self.max_calls - self.calls,
            },
        )
        try:
            strong = await self._attempt("strong", messages, config)
            replies.append(strong)
            reason = self.requirements.reject(strong, messages)
        except ResponseFailure as error:
            replies.append(error.usage)
            reason = "failed-strong-response"
        usage = combine(replies)
        if reason:
            raise ResponseFailure(
                "Strong response rejected",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                measurements=usage.measurements,
                model_calls=usage.model_calls,
            )
        return usage
