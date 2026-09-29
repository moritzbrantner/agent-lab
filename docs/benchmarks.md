# Benchmark suite v1

`uv run --locked python -m agent_lab.benchmark_cli` runs all development fixtures
and writes canonical results and ordered traces under ignored `.runs/`.
Pass `--configuration config.json` with `AgentConfig` fields for real Ollama runs;
`backend` is `ollama`, and exact model/backend digests and versions must be captured.
`--task ID` selects tasks, `--split held-out` deliberately invokes the isolated
regression set, and `--retain-content` explicitly enables private replay.

Tasks are discoverable JSON files validated by `schemas/task-v1.json`, with stable
IDs/versions, immutable input, tool allowlists, step/tool budgets, evaluator hooks,
expected observable output, a fixture reference, and repeat policy. Candidate
prompts contain only instruction, input and declared tool names. Expected answers,
reference plans and evaluator implementation are never provided to the model.
Tools have no filesystem/network/process authority. The runner checks fixture
identity before and after a run, and keeps failed results instead of dropping them.

The five initial categories are arithmetic, stable data transformation, a source
repair fixture, fact retrieval and multi-step batch work. The repository fixture
is currently a source-string workload; issue #20 adds a real repository checkout
and independent executable tests. Fixture adapters are deterministic orchestration
tests, have zero inference tokens, and must never be described as local-model
baseline evidence. Stochastic backends require at least five repeats with captured
seeds. The independent evaluator process and reference-oracle tests arrive in #6.
