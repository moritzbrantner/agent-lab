# Generation and verification

`generate_verify` runs an explicit schedule of up to eight independently seeded
candidates. It verifies each completed candidate with the protected exact evaluator.
Failed batches receive bounded failure feedback, without the expected answer. At
most four candidates can run concurrently; owned tasks are cancelled together.
Selection uses schedule order after the whole batch completes. All launched work,
including rejected candidates, contributes to parent tokens, calls and durations.
Parent wall time, energy and peaks cover the full job, including verification.
Verifier wall time and calls are explicit; child CPU has a broader documented
scope that includes sampler commands. Unknown model work stays unknown.

All current tasks have executable deterministic checks. A model judge is therefore
unnecessary for this experiment. Tasks without an independent check remain outside
this accepted workload; a model opinion must not silently replace correctness.

Run `uv run --locked python -m agent_lab.generation_experiment --output .runs/new`
for fresh single-strong versus cheap/cheap/strong regeneration evidence. The bundle
retains separate cold runs, five warm repeats, parent/child traces and all rejected
results. Compare total parent cost and correctness to assess the Pareto frontier.
