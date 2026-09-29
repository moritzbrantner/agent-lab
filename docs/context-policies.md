# Context policies

`AgentConfig.context_policy` chooses `full`, `retained`, or `relevance`. Full
history remains the reference. Retained state appends structured tool name,
arguments and result once per completed operation, then projects the initial task
and compact completed-work record into each model request. It avoids re-encoding
conversation wrappers; the full authoritative history remains available for
research/debugging, so reduced host RAM is not assumed.

Relevance selection for the retrieval fixture retains only the requested fact
from immutable input. The fixture/evaluator identity remains unchanged and the
agent never receives expected answers. No durable cross-task memory is implicit.
All policies capture their identity in configuration and emit context trace events.

Run `uv run --locked python -m agent_lab.context_experiment --output .runs/context-new`
for five paired samples per case using the pinned medium model. Reports separate
correctness parity, input-token reduction, generated tokens, latency and memory.
A retrieval failure becoming a success is not labelled correctness parity.
The deterministic byte-tokenizer fixture tests sequence equivalence independently;
only actual backend counts are used for live efficiency claims.
