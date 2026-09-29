# Verified-experience distillation

`distill` recognizes the bounded batch-sum recipe from successful measured
model runs. It reloads each development fixture, validates the ordered source
trace/artifact hashes and independently regrades the recorded output. Failed,
held-out, fixture-only, duplicated or changed source experience is rejected.

The resulting artifact contains a trusted deterministic tool reference, a compact
input schema, a reusable procedure and verified input/output examples. Every example
retains its source run, model/configuration, fixture, result, trace and output hashes.
The recipe selects a reviewed implementation from the capability registry; it does
not execute generated Python. This demonstrates experience-driven reuse, not an
unrestricted program synthesizer or proof of generalization from six examples.

`verify_artifact` rebuilds that provenance and checks the implementation identity.
`DistilledRunner` binds execution to the artifact digest and records exercised
authority in the trace. New inputs still face the independent benchmark evaluator.

Run `uv run --locked python -m agent_lab.distillation_experiment --output .runs/new`.
It measures fresh model versus distilled execution on identical tasks/hardware/
software, with five warm repeats and a separate cold model run. Reported derivation
and source re-verification cost is separate from repeated execution cost; amortize
it explicitly when comparing deployment economics. Rejected/source work remains
in its original immutable evidence bundles. No training or held-out data leakage
is implied by this deterministic artifact.
