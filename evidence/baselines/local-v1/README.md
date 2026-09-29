# Local baseline evidence v1

Captured on 2026-09-29 using the exact profiles, hardware, software and source
identity in `manifest.json`. Includes 120 steady-state samples and four separately
retained cold-load task runs. Failures and malformed outputs retain consumed work.

| Profile | Passed | Generated tokens (all attempts) | Peak sampled RAM GiB | Peak sampled VRAM GiB |
| --- | --- | --- | --- | --- |
| medium-q4-gpu | 10/30 | 4090 | 0.99 | 4.59 |
| small-q4-cpu | 0/30 | 971 | 0.69 | 0.00 |
| small-q4-gpu | 0/30 | 1000 | 0.76 | 0.61 |
| small-q8-gpu | 0/30 | 1344 | 0.75 | 0.73 |

The medium model passes batch arithmetic and source repair. Exact serialized
ordering, retrieval and structured final replies expose failures; the 0.5B model
fails this protocol across all measured tasks. These are baselines, not assistant
capability claims. CPU offload disabled for model layers still initializes a small
GPU context in this backend, visible in sampled VRAM.

See `summary.json` for each task/configuration's mean, sample standard deviation,
missing counts and exact identities. All public outputs and metadata traces are
retained under `runs/`. No private trace payloads are retained.

Verify with `uv run --locked python -m agent_lab.baselines verify --output evidence/baselines/local-v1`.
The deterministic test gate also checks all artifact hashes. To reacquire, use a
new output directory and the declared profiles; do not overwrite this bundle.
