# Guarded scaffolding improvement

The current agent proposes a single bounded configuration mutation through one
registered `propose` tool. Its input contains grouped development scores, token
counts, configurations and run provenance. The proposal interface rejects held-out
results and gives the agent no filesystem, fixture or evaluator tools. Candidate
materialization permits context selection, bounded prompt suffixes, retry counts
and output budgets. It cannot change model weights, fixture authority or tool
implementations. This initial surface can be extended only through reviewed code.

The experiment permits at most three planner attempts; each uses two steps, one
tool call and a 512-token response limit. All measured attempts have separate
configuration/schema identities and retained work. The planner then loses authority
before protected development and held-out runs begin. Five repeats per task use
identical hardware/software. Fixture and evaluator hashes are checked before and
after evaluation.

`decide` requires matching cohorts and refuses correctness regressions on any task,
including held-out tasks. Unknown correctness rejects the mutation. A development
correctness gain or total token reduction without regressions accepts it; neutral
changes remain archived. Per-metric Pareto comparisons, rejected results, proposal
work and the materialized configuration remain inspectable. Acceptance here means
retaining an experiment candidate, rather than a claim of broad model improvement.
The initial held-out suite is deliberately small; broader protection follows in the
archive milestone.

Run `uv run --locked python -m agent_lab.improvement_experiment --output .runs/new`.
The local evidence demonstrates an actual model proposal, independent evaluation
and the resulting decision. No expected held-out answer is sent to the proposer.
