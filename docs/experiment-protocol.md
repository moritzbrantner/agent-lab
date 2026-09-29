# Experiment protocol v1

A result conforms to `schemas/result-v1.json`. Correctness is evaluated by a
named independent evaluator; partial credit is allowed only when the task
specifies a decomposable score. An evaluator error has no score.

Capture exact model artifact digest, quantization, backend/version, sampling,
context limit, seed, agent policy, tool identities, fixture hash, hardware profile,
and software identity in each configuration. Comparisons reject fixture,
protocol, hardware, or software drift and enumerate configuration differences.
Use separate hardware strata rather than comparing unrelated machines.

Work counters are deterministic evidence when the adapter supplies them.
Unavailable values are null, never zero. Measurements require a source string
that identifies measured or estimated origin. Model-load/setup costs are separate
from steady-state inference and tools; wall time includes all phases.

Report success against generated tokens, wall time, and peak RAM separately.
`compare` returns dominance, equivalence, tradeoff, or unavailable per view.
Do not convert these into an arbitrary single score. Deterministic regression
gates reject correctness loss or work-budget excess. Wall-clock gates require
an explicit baseline, relative threshold, and absolute noise allowance.

One sample suffices for deterministic fixtures. Stochastic models require at
least five independently seeded repeats per task/configuration; performance
experiments require at least five repeats after a separately measured cold load.
Keep every sample, including failures. `summarize` reports mean, sample standard
deviation, sample count and missing count; a single sample has no variance estimate.
Never pool different task/configuration/hardware strata.
