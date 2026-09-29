# Resource evidence

The runner records end-to-end `perf_counter` time (including evaluation and
measurement overhead), harness CPU time, tool-boundary duration, and backend
reported prompt/generation/model-load durations. Failed or unsupported inference
timing remains null; it is never assumed zero. Work counters remain independent.

RAM is the maximum sampled sum of harness and all visible Ollama process-tree RSS.
VRAM is the maximum sampled NVIDIA allocation for that Ollama process tree.
Samples occur at start/end and every 200ms, so short peaks may be missed. Sources
explicitly identify this as sampled evidence, including scope; shared RSS pages
may be counted twice. Use a quiescent dedicated backend for baseline experiments.
Other GPU processes are excluded. Unsupported drivers/platforms report null.
Harness CPU time excludes model-server CPU. GPU execution time and utilization
remain null until a reliable provider is available rather than estimating them
from wall-clock time. Backend load cost is reported separately from inference.

Energy is available only with a readable RAPL package counter and its wrap range.
Its source names the package-only scope (not whole-system power); sampling must
span fewer than a full counter range. Unsupported energy is null. No estimated
energy values are currently produced. Deterministic tests use fake clocks and
counters, never assert wall-clock thresholds.

Hardware profiles capture CPU, architecture, memory, GPUs and driver version.
Software identity records Python, platform and a hash of runtime source. Run IDs
include fixture, configuration, repeat and both environment identities. Measurements
always have explicit source strings, while unavailable values have no source.
