# Local workspace

The first surface is a keyboard-first terminal application. It uses the runtime's
checkpoints, budgets, tool permissions, traces, and measurements as its authority.
It works in a terminal or over SSH, including a mobile terminal. This is an
experimental interface; measured model failures remain visible.

```sh
uv run --locked python -m agent_lab.workspace --help
uv run --locked python -m agent_lab.workspace start work /absolute/project/path
uv run --locked python -m agent_lab.workspace attach SESSION /path/to/context.txt
uv run --locked python -m agent_lab.workspace run SESSION 'Read the attached context'
uv run --locked python -m agent_lab.workspace inspect SESSION
uv run --locked python -m agent_lab.workspace export SESSION > configuration.json
```

`start --profile` selects a pinned profile from
`configurations/local-baselines-v1.json`. Model digest, quantization, and backend
version must match the installed local Ollama inventory. The default is
`medium-q4-gpu`. `fixture-demo` is a deterministic attachment/read walkthrough
that needs no inference server and makes no model capability claim. Each run
exports the exact configuration, project, attachment hashes, permissions,
hardware, runtime identity, and language. A new completed turn retains conversation
messages and gets a fresh declared budget; resuming an interrupted turn retains
all spent attempts, tokens, and tools.

Press Ctrl+C to interrupt; `cancel SESSION` from another terminal signals only
the recorded owned Linux process. `resume SESSION` loads the original checkpoint.
Changed runtime identity or permissions prevent resume. A completed tool is not
repeated. An interrupted non-idempotent action with unknown completion requires
owner inspection rather than automatic replay. `abandon SESSION` retains the
failed run and explicitly starts a fresh conversation on the next `run`.

`sessions`, `history SESSION`, and `inspect SESSION` expose state and activity.
`--json` gives structured values; `export` always returns JSON. The exported
configuration identifies a run; project files and model availability must also
be preserved to reproduce it. History stays under
`$XDG_DATA_HOME/agent-lab` or `~/.local/share/agent-lab`; `--root` selects a new,
dedicated directory. Session/checkpoint/attachment files have owner-only access.
Private checkpoints necessarily contain conversation and tool results. Public
trace metadata retains tool names, hashes, and counters without tool contents.

Project access initially permits bounded text reads and top-level listings;
hidden paths and the private history directory are excluded. Attachments are
explicit owner-selected UTF-8 snapshots limited to 16 KiB. `permissions SESSION`
shows grants. `grant SESSION SCOPE RESOURCE OPERATION --reason REASON` records an
owner approval for future turns. The available tools remain a fixed registry:
project and attachment reads, integer sum, memory, and scoped offline process
execution. A grant does not add a new tool. Process execution requires an exact
executable grant and the pinned native sandbox; it has no host network or home
access. See [permissions](permissions.md) for sandbox limits.

`memory SESSION on|off|clear` controls app-owned durable memory. It is disabled
initially; enablement grants bounded writes and reads only in that session's
memory directory. `forget SESSION` explicitly deletes inactive session history,
attachments, and memory. It never deletes the selected project.

`settings --set-language en|de|es --theme light|dark|system` persists localization
and terminal theme preferences. `--language` overrides the interface language for
one command. Stable command IDs and JSON keys stay unchanged. Model output uses
the captured session language; tool activity uses localized labels.

Reproduce the public interruption walkthrough with:

```sh
uv run --locked python -m agent_lab.workspace_fixture --output .runs/workspace-demo
```

The retained [walkthrough](../evidence/workspace/local-v1/report.json) includes the
interrupted state, resumed checkpoint, trace, exact exported configuration, and
unchanged single tool count. Process cancellation and localized discovery also
run in the deterministic test gate.

The [actual model pilot](../evidence/workspace/local-v1/model-pilot.json) used the
pinned 7B local model, called `sum([1,2,3])` once, and returned `6`. This single UI
walkthrough is separate from repeated capability and release measurements.
