# agent-lab

Local agent experiments that compare independently verified capability and compute.

Implementation follows the [research roadmap](https://github.com/moritzbrantner/agent-lab/issues/1).

Requires Python 3.14.4 and uv 0.11.21. Install declared dependencies with
`uv sync --locked`. Run `make check` for the canonical deterministic gate and
`make format` for explicit formatting. No inference server is needed for tests.

`uv run --locked python -m agent_lab validate result.json`
validates the [result contract](schemas/result-v1.json). Commands `compare` and
`summarize` produce JSON; see the [experiment protocol](docs/experiment-protocol.md).

Start a [local workspace](docs/workspace.md) with
`uv run --locked python -m agent_lab.workspace --help`. Sessions retain project
context, attachments, permission decisions, tool activity, and recoverable runs.
