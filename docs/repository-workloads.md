# Historical repository workloads

The first fixture snapshots agent-lab's real runtime at commit
`59fe67036e5a233931fd791c649033d5c090c912`, before the unknown-usage repair in
`ccd509fc658abc7816a7716160dc8f8c038af426`. The manifest identifies the source
repository/path/commit and every immutable fixture file. `AGENTS.md` restricts the
repair to adapter-request exception accounting and names the parent-owned check.

The controller resolves the live shared convention stack through coding-tooling,
reads every resolved file and records sourceRevision/hash references. It also reads
local AGENTS/CLAUDE instructions. Missing tooling/registry produces an explicit
setup error; shared text is never copied into the fixture or evidence. The fixture's
Python requirement matches the canonical pinned environment.

Agents get scoped read, bounded exact replacement and protected check tools. They
cannot edit instructions, tests or evaluator code. Before executing a candidate,
the parent checks that all AST outside the single reviewed exception-handling seam
is unchanged; the new handler permits only null token assignments and re-raise.
This intentionally narrow historical workload accepts a chained assignment or two
assignments, rather than arbitrary generated code. It prevents candidate code from
spoofing the behavioral observation worker.

An offline read-only sandbox observes first/later unknown completion, cancellation,
known successful response and known tool failure. The parent compares observations
to independently held expected properties. The original snapshot fails; the repair
passes. No candidate-defined test exit code is treated as correctness. The patch,
original-source provenance, live-policy references, permissions, verification and
resource results remain inspectable.

Run `uv run --locked python -m agent_lab.repository_experiment --output .runs/new`.
Five warm repeats plus separate cold runs compare a clearly labelled reference
fixture agent and the pinned local coding model. The retained reference succeeds;
the local model fails this workload. Public historical source is explicitly retained
in private trace sidecars for replay; replay substitutes recorded tool outputs,
while immutable patch artifacts and independent checks preserve the coding result.
Generated source fixtures/evidence are excluded from repository formatters to keep
the original Git snapshot and measured patches immutable.
