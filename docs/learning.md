# Gated learned specialization

Learning starts after independent positive/negative evaluator probes, the measured
baseline bundle and retained regression-protection evidence pass `entry_gate`.
The controller captures fixture/evaluator/selection hashes. The trainer receives
verified development examples; held-out input hashes are exclusion checks and no
held-out answers enter the examples.

The first recipe projects the proven batch-sum tool into synthetic standalone sum
responses. Each example retains the source artifact and model-run provenance and
is separately checked with exact rational arithmetic. Content/schema/template
validation prevents arbitrary prompt substitution. The data contains 32 bounded
examples, including the visible development sum; this is a narrow specialization
experiment, not evidence of general arithmetic learning.

The optional locked `training` group pins Torch, Transformers, PEFT and Hub. The
[PEFT quick tour](https://huggingface.co/docs/peft/en/quicktour) describes the LoRA
approach; the [Qwen model](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
is pinned to a repository revision and independent LFS weight digest. Acquisition
is explicit and sends no credential token:

```sh
uv sync --locked --group training
uv run --locked --group training python -m agent_lab.learning_acquire \
  --output .runs/models/qwen-coder-0.5b
uv run --locked --group training python -m agent_lab.learning_experiment \
  --base .runs/models/qwen-coder-0.5b --output .runs/learning-new
```

Training/evaluation use local files only, disable Hub network lookup and remote
code, and use safetensors. Rank-four Q/V LoRA trains two epochs with batch size one,
BF16, fixed seed, eager attention and deterministic Torch algorithms on the pinned
CUDA machine. Training loss, steps, tokens, parameters, model-load/optimization
wall time, scoped RAM/VRAM and optional energy are separate from inference. CUDA
events report an interval including host dispatch gaps, rather than pure GPU busy
time. Generic inference GPU busy time remains unavailable.

Five warm repeats plus separate cold runs compare untuned and learned models on
development and broader held-out tasks. Malformed replies retain consumed work.
The same regression guard decides eligibility; unknown correctness cannot promote
a candidate. Adapter weights, data, configuration, framework versions, model-file
hashes and trace artifacts are retained. Ordinary `make check` needs no ML packages,
model download or GPU. No model merging is required for this first experiment.
