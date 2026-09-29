# Historical runtime repair

Only runtime.py may change. Repair unknown usage accounting when an adapter request
fails or is cancelled. Preserve completed response usage and all existing tool,
retry, budget and completion behavior. This fixture permits adding only the bounded
exception handling needed for that request failure; imports and other code are protected.

The trusted controller resolves and reads the live shared convention stack before
execution and retains its sourceRevision/file hashes. Do not copy shared policy
into this fixture. Use the parent-owned check_patch tool as the deterministic gate.
