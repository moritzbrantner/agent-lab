# Runtime

`run(adapter, tools, config, state, emit)` owns one explicit `AgentState`.
Messages, steps, model attempts, retries and tool counts remain available after
failure. No hidden global state or backend-specific routing exists in the loop.

`SequenceAdapter` and `FunctionAdapter` run the same deterministic tool fixture.
`OllamaAdapter` consumes the local [chat API](https://docs.ollama.com/api/chat)
with schema-constrained replies (`content`, `calls`). Prompts declare available
tool names/arguments. Capture the digest from `/api/tags` in `AgentConfig`.
Only loopback endpoints are supported. Sampling/context/output bounds are explicit.

Only explicitly transient pre-completion model errors retry with bounded backoff.
Tool failures and uncertain transport failures propagate. Cancelling a run marks
state cancelled and propagates cancellation; a finite blocking HTTP request may
finish in its worker thread but its result is discarded. Trusted synchronous
tools must be finite computations. Async external tools receive a timeout and
cancellation. These callables do not constitute filesystem/process isolation.
