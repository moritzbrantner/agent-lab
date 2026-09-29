# Adapter efficiency experiments

Optional `negotiate` asks an adapter to apply bounded features and returns a status
for every request. Adapters without the interface preserve the original config.
The Ollama adapter supports context size and GPU layer requests. Quantization is
selected through exact installed model digests, rather than changing weights at
runtime. Core orchestration does not interpret backend options.

The [Ollama parameter reference](https://docs.ollama.com/modelfile) documents
context-size controls; its [FAQ](https://docs.ollama.com/faq) describes placement
and server caching/concurrency. This experiment pins Ollama 0.30.11 and the
installed Qwen models. Prefix/KV reuse, speculative decoding and batching have no
measured per-request control for this model setup: their status is unavailable.
Automatic server caching is not claimed as an independently measured improvement.

`uv run --locked python -m agent_lab.inference_experiment --output .runs/new`
compares Q4 CPU/GPU, Q4/Q8 GPU and 4096/2048 context on the same batch-sum task,
with five warm repeats and separate cold runs. The bundle records correctness,
all model work, inference/setup/tool/wall time and scoped memory/energy. Changes
may worsen correctness; the report keeps those outcomes. This is a workload
measurement, not a recommendation to change the default model.
