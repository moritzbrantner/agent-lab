# Local baseline v1

Acquire the declared models explicitly before running deterministic verification:

```
ollama pull qwen2.5-coder:0.5b-instruct-q4_K_M
ollama pull qwen2.5-coder:0.5b-instruct-q8_0
ollama pull qwen2.5-coder:7b
uv run --locked python -m agent_lab.baselines capture --output configurations/local-baselines-v1.json
uv run --locked python -m agent_lab.baselines run --manifest configurations/local-baselines-v1.json --output .runs/baseline-new
```

Capture is an explicit mutation recording model SHA-256 identity, quantization,
Ollama version, sampling/context/budgets, offload settings and hardware profile.
Execution rejects missing models, backend/quantization/digest drift and different
hardware. It never downloads models. Use a new profile version for intentional
upgrades and a new output directory for each acquisition.

Four points cover a 0.5B model on CPU and GPU, Q4_K_M and Q8_0 quantization, and a
7B Q4_K_M GPU model. Every point runs the same development suite five times with
seeds 0–4. Model unload before each point produces a separate cold-load sample in
`cold-results.jsonl`; five subsequent repeats per task retain steady-state/load
separation. Rejected/failed/error samples remain
in `results.jsonl` and summaries show successes plus variance/missing counts.

Committed baseline bundles are intentional immutable research evidence, an
exception to disposable generated output. They include canonical results,
configuration/hardware/software identity, a content digest and summaries. Run
`verify` to validate their integrity. No wall-clock assertion belongs in CI.

A baseline measures this compact suite and machine; it is not a claim of useful
assistant parity. Sampled memory scopes and unavailable telemetry remain visible.
