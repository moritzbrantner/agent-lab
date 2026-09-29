# Reusable batch utility evidence

Five live model references and five deterministic utility runs all pass the same
independent batch-sum evaluator. The model averages 125 generated tokens; the
utility uses zero model calls and zero generated tokens. Inputs, outputs, utility
identity, cache provenance, sampled resources and artifact hashes are retained.
The cold-load run is separate. This demonstrates one reusable computation rather
than a general capability gain. The standalone utility profile unloads the model
before memory measurement; the report exposes that configuration change.
