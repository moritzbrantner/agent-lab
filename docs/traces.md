# Trace v1

Trace metadata is ordered by a monotonic per-run sequence and stable run:sequence
IDs. Run/task/configuration and optional parent-run identities do not depend on
backend. Runtime hooks record transitions, attempts, tools, retries and failures;
artifact, evaluator and measurement producers append their own events.

Default retention stores metadata only. Prompts, responses, tool arguments,
results and exception messages are omitted. Public metadata permits only explicit
status/category fields and caller-sanitized artifact/evaluator/measurement/
authority records. Never put credentials in configuration identity or metadata.

Explicit `retain_content=True` writes a separate `.content.json` sidecar with
mode 0600 and a content integrity digest. Keep private traces in ignored `.runs/`
and share only reviewed metadata. Stable opaque references disclose no prompt
content digest. Saving uses atomic replacement; readers reject an inconsistent
sidecar. Artifact SHA-256 values detect output drift within a supplied root.

`Replay` substitutes both model and tool results, verifies each request in order,
and requires complete consumption. It uses no live inference or real tool effects.
Metadata-only traces cannot replay and fail explicitly. Retained private content
is required for deterministic replay; a digest alone cannot reconstruct it.
