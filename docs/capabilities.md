# Reusable capabilities

`discover_capabilities()` loads versioned repository-owned manifests. Each utility
has an ID, version, input/output JSON Schemas, implementation source hash, tests,
and an identity covering the manifest and implementation. Registered code is
trusted; arbitrary generated programs are not executed through this registry.
Generated helper execution must pass the sandbox boundary introduced in #19.

`Capability.invoke` validates inputs and outputs, records provenance, and optionally
uses a private atomic cache keyed by complete capability and input identity.
Corrupt output/provenance records are discarded and recomputed. Cache warmth does
not affect correctness. Cache provenance is caller-declared until independently
verified by the distillation pipeline; it is never presented as a proof by itself.

`CapabilityRunner` executes without model calls and participates in the same
benchmark/evaluator/trace contracts. The first utility totals integer batches.
Run `uv run --locked python -m agent_lab.capability_experiment --output .runs/helper-new`
to compare five live model runs with five utility runs. Cold-load cost is separate,
and the standalone utility profile unloads the model before resource measurement.

Discover with `uv run --locked python -m agent_lab.capabilities list`, then invoke
with `run --id batch-sum-v1 --input input.json`. The input file contains
`{"batches": [[1, 2], [3]]}`; an optional `--cache .runs/capability-cache` enables reuse.
