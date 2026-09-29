# Context experiment v1

Twenty live steady-state samples and one separate cold-load run use the pinned
7B model on the same hardware/software and immutable fixtures.

| Task / policy | Mean input tokens, full → candidate | Full / candidate success | Decision |
| --- | --- | --- | --- |
| batch-sum / retained | 347 → 318 | 5/5 → 0/5 | Correctness regressed; keep experimental |
| retrieve-fact / relevance | 386 → 294 | 0/5 → 5/5 | Task-specific selection fixes truncation failures |

Full history stays the default. Structured retained state is sufficient for the
deterministic fixture but this model does not reliably follow its compact format.
Fact selection reduces unnecessary input and tool arguments without exposing the
evaluator. Retrieval improvement is not described as correctness parity because
the full reference failed. `report.json` preserves all metrics, variance, missing
counts and paired Pareto comparisons. Public trace/artifact hashes are checked by CI.
