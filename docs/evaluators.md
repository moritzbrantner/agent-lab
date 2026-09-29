# Evaluator authority

The benchmark runner calls a stdlib-only child Python process with isolated import
mode, a sanitized environment and a finite timeout. The child resolves its trusted
fixture, verifies its hash, and consumes only a candidate output string. It never
imports or executes candidate code. Agents receive no evaluator-edit capability;
the declared benchmark tools have no filesystem/process/network authority.

This is a capability boundary, not an OS sandbox for arbitrary Python adapters.
Fixture adapters, the harness and registered tools are trusted repository code.
Self-improvement candidates are configuration/data, never evaluator or harness
code. Arbitrary generated helper execution requires the sandbox in #19.

Current correctness is exact deterministic equality, including ordering and
serialization. A pass is full credit; errors are not correctness results. No model
judges are used. Future semantic/model judging must have a distinct evaluator ID
and must never be labelled deterministic.

`IncrementalIndex` retains sorted keys across put/delete operations;
`reference_sequence` independently rebuilds and sorts complete state after every
operation. `differential_report` compares complete canonical serialized sequences,
including repeated replacement, equal ranks, removals and missing removals.
