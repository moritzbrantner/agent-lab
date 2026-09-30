# Fixed-hardware release thresholds

The machine-readable [profile](../configurations/release-thresholds-v1.json) defines
`useful-local-3060ti-v1`: Ryzen 7 5700X, 16 logical CPUs, 32,476,340,224 bytes RAM,
RTX 3060 Ti 8 GiB, and the exact recorded driver. Its initial configuration uses
the digest-pinned Qwen2.5 Coder 7B Q4 model and Ollama 0.30.11. These are explicit
engineering targets on small reproducible workloads, not a comparison with the
intelligence or broad coverage of hosted assistants.

| Requirement | Threshold |
| --- | --- |
| Chat/reasoning | At least 95% exact arithmetic replies on each development/held-out sum task |
| Coding | At least 95% independently verified source repair and historical repository repair |
| Tool reliability | At least 95% correct outputs with tool use on each batch/index/sort/retrieval task |
| Held-out correctness | At least 95% on each of three controller-owned regression tasks |
| Short context retention | At least 95% exact retrieval on each explicit retrieval fixture |
| Configured context | At least 4,096 tokens; repository workload uses 8,192 |
| Effective context | Correct independently checked retrieval from a single prompt of at least 2,048 tokens |
| Interactive latency | Warm end-to-end p95 at most 10 seconds per task |
| Repository latency | Warm end-to-end p95 at most 30 seconds |
| Peak memory | At most 24 GiB scoped RAM and 8 GiB scoped VRAM |
| Recovery | Completed actions not repeated, spent request budget retained, ambiguous writes refused |
| Durable memory | Owner-enabled memory survives controller recreation and can be cleared |
| Offline | Loopback inference, ungranted HTTP denied, pinned native process isolation |
| Energy | Optional p95 at most 500 measured RAPL package joules |

The threshold schema validates profile definitions. The reporter binds task and
fixture identity, evaluator, exact per-workload configuration, model digest,
backend version, hardware, software, and unique repeat IDs. Only seed varies
within a workload. The repository configuration explicitly differs in context,
output budget, and protected authority options. Five warm repeats per workload
are required; at this sample size the empirical nearest-rank p95 is the maximum,
and 95% correctness requires five successes. These are reproducible small-sample
milestones, not statistical confidence claims. Errors count as unsuccessful.

Every threshold reports `pass`, `fail`, or `unavailable`, with task-level values
and sample counts. Any missing value makes that task's resource threshold
unavailable; available samples cannot silently replace missing ones. Any failed
required threshold makes the release fail. Required unavailability prevents
readiness. Optional energy unavailability stays visible and does not prevent
readiness. Costs include failed attempts; unsupported energy is never zero.

```sh
# Uses only installed pinned models; no model download or training.
uv run --locked python -m agent_lab.release_experiment run \
  --profile configurations/release-thresholds-v1.json \
  --output .runs/release-candidate

# Recomputes and verifies retained evidence without contacting inference.
uv run --locked python -m agent_lab.release \
  --profile configurations/release-thresholds-v1.json \
  --bundle evidence/release/local-v1
```

Reporting exits 0 for pass, 1 for required unavailability, and 2 for failure.
The experiment runner retains completed samples incrementally; an interrupted
acquisition has no final verified manifest and cannot claim a release. The final
manifest hashes every retained input, result, trace, proof, and measured source.
Reporting regrades pass/fail benchmark outputs with the independent evaluator,
checks repository verification artifacts, and checks trace/measurement identity.
One cold-load sample per workload is retained separately and excluded from warm
latency; each workload begins with a confirmed unloaded, dedicated model backend.

To evaluate another agent/model, author a new versioned profile with its exact
configuration and per-workload budgets, retaining these explicit requirements,
then acquire a fresh cohort. Old evidence cannot be mixed with changed software,
hardware, fixtures, quantization, model weights, or authority. `release_experiment
profile --output NEW_FILE` writes the initial profile for review; it never
overwrites an existing profile. Profile changes are owner-controlled, outside
model tool authority.

The controlled recovery and memory proofs exercise the same pinned runtime
configuration with explicitly labelled fixture replies. They establish scaffold
behavior, not model reasoning. The native offline check establishes isolated
process networking and host-home exclusion; it does not establish a host-wide
airgap. Existing retrieval tasks are short. They do not establish the effective
2,048-token capacity requirement, which remains unavailable until an appropriate
versioned long-context workload is independently evaluated. The held-out tasks
are public controller-owned regressions, not a secret test corpus or evidence of
broad generalization.

The retained [report](../evidence/release/local-v1/report.json) records 50 warm
runs: 15 passes, 25 failures, and 10 errors, plus ten separate cold-load runs.
Batch summation, simple source repair, and held-out short retrieval pass all five
repeats each. Reasoning, protected repository repair, tool reliability across
the full suite, held-out correctness, and context retention fail. Latency,
sampled RAM/VRAM, configured window, and controlled recovery/memory/offline
checks pass. Effective long-context capacity and package energy are unavailable.
The release therefore reports `status: fail` and `ready: false`.

Research optimizations and learned candidates retain their own evidence and
rejection decisions; completing the implementation roadmap does not imply that
a useful assistant release passes.
