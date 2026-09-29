# Routing

`RoutingAdapter` selects among pinned configured adapters using declared tool/
answer requirements. Cheap-first tries the cheap model, escalates on malformed or
empty replies, unknown tools, incomplete mandatory tool work or invalid answer
shape, then stays on the strong model. Role-specialized mode initially assigns
mandatory tool work to the strong model. Shape validation is not task correctness;
the independent evaluator still judges the final result.

Escalation and total primitive-call budgets are explicit. Configuration captures
all inner models, sampling, requirements and policy. Routing decisions include
their observed tool count and remaining budget. Child request/response content is
private by default; public trace metadata identifies attempts and model digests.
Every rejected response contributes tokens, model calls and backend durations.
Unknown completion usage remains null; cancellation preserves known call counts.

Adapters may expose an observer hook and a monotonic invocation counter when they
compose multiple requests. Core orchestration contains no model-selection policy.
`Reply.model_calls` is trusted adapter telemetry, never model-authored JSON.

Run `python -m agent_lab.routing_experiment --output .runs/routing-new` through uv
to compare single cheap/strong, cheap-first and role-specialized policies on the
same task. Five steady-state repeats follow a separate cold run per configuration.
The report reveals whether routing improves correctness, tokens, latency or memory.
