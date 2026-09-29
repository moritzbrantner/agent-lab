# Local privacy and tool authority

A session starts with no grants. Immutable owner-supplied `Grant` values authorize
one operation on one resource in seven separate scopes: filesystem, process,
network, secrets, repositories, documents and durable memory. Repository/document/
memory roots do not imply each other's authority. `Policy.elevate` is a trusted
control-plane API requiring an owner approval identity and reason; it is never a
model tool. The session persists the exact grant configuration and its digest.

File tools require a granted root and relative path. Directory descriptors and
no-follow opens prevent traversal and symlink swaps from following outside the
scope. Ordinary single-link files are required. UTF-8 reads/writes have a 1 MiB
budget; writes replace atomically and fsync. Deletion is separately granted.
Credential values stay in the parent's private store; tools return opaque handles.
This initial implementation does not hand those values to child processes or HTTP
requests. Future native connectors must implement their own approved handle use.

HTTP GET requires an exact URL grant, disables ambient proxies/redirects, and has
socket and response-size bounds. A grant for HTTP does not enable process networking.
The pinned Linux bubblewrap backend always uses an isolated network/PID namespace,
cleared environment, dropped capabilities, read-only runtime and explicitly granted
read roots under `/scopes/N`. Persistent writes use the native scoped file tool.
Process argv, output and time are bounded; timeout/cancellation kills and reaps the
owned process. No unsandboxed fallback exists. Native availability is explicit;
other machines need a reviewed binary pin before process execution becomes usable.

Every tool action is attributed through runtime request/result events and authority
checks. Public authority metadata hashes resources and never includes credential
values. Private context/checkpoints require explicit local retention and 0600 files.
Reads/opaque handle requests are declared idempotent. Writes, deletes, processes and
HTTP calls are conservatively non-idempotent: uncertain completion must stop
recovery unless a tool later provides a stronger transactional guarantee. Bind the
policy/tool identity to `CheckpointStore` when using recoverable sessions.

Run `uv run --locked python -m agent_lab.permission_fixture --output .runs/new`.
Retained local evidence demonstrates allow/deny/owner elevation, secret-free traces,
no host home access and offline child execution. The canonical gate tests scoped
filesystem authority on every host and runs native sandbox checks where the exact
pin is available; unsupported native checks are explicitly skipped, never passed.
