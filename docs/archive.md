# Evolutionary archive

`Archive` keeps immutable content-addressed configurations, ancestry, mutation
reasons, all result metrics and decisions. Rejected entries retain compact result
records; their original run IDs and artifact hashes point to the source evidence
bundle without duplicating every trace. Missing or changed entry/regression data
fails validation. The archive is local and single-writer.

Selection stratifies by the exact task/protocol/hardware/software cohort. It uses
one correctness objective per task plus total tokens, total latency and peak
RAM/VRAM. No weighted scalar combines those dimensions. Unknown objectives exclude
an entry from selection. Held-out regressions reject a child even when development
improves; broad regression failures also exclude it. Multiple lineages with genuine
cost tradeoffs can remain on the frontier.

Every three new entries by default, `evaluate_survivors` runs the supplied protected
broader suite. A cycle is recorded only after all scheduled evaluations complete;
a crash cannot mark a partly checked cycle complete. The protected held-out suite
now adds different retrieval facts and a variable-length batch workload. None of
these inputs is passed through the proposal interface.

Import measured improvement evidence with:

```sh
uv run --locked python -m agent_lab.archive --archive .runs/archive \
  --import-improvement evidence/improvement/local-v1
```

The retained local archive correctly has no eligible frontier: its baseline has
unknown correctness and its proposed child was rejected. Deterministic tests also
exercise two non-dominated lineages, rejection of development gains that lose
held-out correctness, persistence/reload and executable periodic regression.
