# Checkpoint and recovery

`run_checkpointed` writes a private 0600 atomic checkpoint before requests/actions
and after completed responses/actions. File and directory fsync make replacement
durable. It contains the task/configuration identity, outstanding actions, retained
messages/working memory, counters, tool authority, artifact hashes and ordered trace
continuation. Checkpoints contain sensitive context; storing them is an explicit
choice separate from public trace retention. Delete private checkpoints when their
session is no longer needed.

Load through the same `CheckpointStore` authority and configuration. It rejects
identity, content, counter, trace or artifact drift. Resume executes outstanding
work before requesting another model response. Completed sessions return directly.
A lost inference completion can be requested again: spent requests remain counted
and unknown token work remains null. Budgets are never reset.

If an action was started but its completion was not durably recorded, recovery
refuses non-idempotent actions. Only explicitly declared idempotent tools may be
retried. A checkpoint cannot guarantee exactly-once external side effects; those
need tool-owned transactional/idempotency mechanisms. Sync tools must be finite;
external process/network tools own their cancellation bounds.

The canonical gate interrupts fixtures after a response, after a tool and after
completion, then checks the same final state with exactly two model requests and
one tool call. It also rejects ambiguous writes and changed artifacts. Runtime
state and checkpoint storage remain independent of inference backends and UIs.
