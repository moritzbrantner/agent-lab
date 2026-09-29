"""Schema-checked reusable utilities with identity, provenance and transparent cache."""

import argparse
import copy
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path

from jsonschema import Draft202012Validator

from agent_lab.experiments import ROOT, canonical, digest
from agent_lab.runtime import RuntimeFailure
from agent_lab.trace import atomic_json


def batch_sum(value):
    return str(sum(sum(batch) for batch in value["batches"]))


IMPLEMENTATIONS = {"batch_sum": batch_sum}


class Capability:
    def __init__(self, manifest):
        if (
            manifest.get("schema_version") != 1
            or manifest.get("implementation") not in IMPLEMENTATIONS
        ):
            raise ValueError("Unsupported capability manifest")
        self.manifest = copy.deepcopy(manifest)
        self.id = manifest["id"]
        self.function = IMPLEMENTATIONS[manifest["implementation"]]
        self.identity = digest(
            {
                "manifest": manifest,
                "implementation_source": inspect.getsource(self.function),
            }
        )
        for name in ("input_schema", "output_schema"):
            Draft202012Validator.check_schema(manifest[name])

    def _validate(self, name, value):
        errors = list(Draft202012Validator(self.manifest[name]).iter_errors(value))
        if errors:
            raise ValueError("Invalid capability " + name + ": " + errors[0].message)

    def invoke(self, inputs, *, cache=None, source_runs=()):
        self._validate("input_schema", inputs)
        input_hash = digest(inputs)
        key = digest({"capability": self.identity, "input_hash": input_hash})
        path = Path(cache) / (key + ".json") if cache is not None else None
        if path is not None and path.exists():
            try:
                record = json.loads(path.read_text())
                if not isinstance(record, dict):
                    raise ValueError("Invalid disposable cache record")
                self._validate("output_schema", record["output"])
                if (
                    record["capability_hash"] == self.identity
                    and record["input_hash"] == input_hash
                    and digest(record["output"]) == record["output_hash"]
                    and digest({k: v for k, v in record.items() if k != "record_hash"})
                    == record["record_hash"]
                ):
                    return {**record, "cached": True}
            except ValueError, KeyError:
                pass  # Invalid disposable cache is recomputed, never used as evidence.
        output = self.function(copy.deepcopy(inputs))
        self._validate("output_schema", output)
        record = {
            "schema_version": 1,
            "capability_id": self.id,
            "capability_hash": self.identity,
            "input_hash": input_hash,
            "output": output,
            "output_hash": digest(output),
            "source_runs": sorted(set(source_runs)),
            "cached": False,
        }
        record["record_hash"] = digest(record)
        if path is not None:
            atomic_json(path, record)
        return record


def discover_capabilities():
    return [
        Capability(json.loads(path.read_text()))
        for path in sorted((ROOT / "capabilities").glob("*.json"))
    ]


class CapabilityRunner:
    """A trusted execution strategy; this utility never calls a model adapter."""

    def __init__(self, capability, inputs, *, cache=None, source_runs=()):
        self.capability, self.inputs = capability, copy.deepcopy(inputs)
        self.cache, self.source_runs = cache, source_runs

    async def __call__(self, adapter, tools, config, state, emit):
        if (
            config.backend != "capability"
            or config.options.get("capability") != self.capability.id
            or config.model_digest != self.capability.identity
            or config.max_tool_calls < 1
        ):
            raise RuntimeFailure("Capability identity/budget mismatch")
        state.status = "running"
        emit("state", {"status": state.status, "configuration": asdict(config)})
        emit(
            "tool_request",
            {
                "name": self.capability.id,
                "arguments": self.inputs,
                "capability_hash": self.capability.identity,
            },
        )
        result = self.capability.invoke(
            self.inputs, cache=self.cache, source_runs=self.source_runs
        )
        state.tool_calls += 1
        state.steps += 1
        state.output = result["output"]
        state.status = "completed"
        emit(
            "tool_result",
            {
                "name": self.capability.id,
                "value": state.output,
                "provenance": {k: v for k, v in result.items() if k != "output"},
            },
        )
        emit("state", {"status": state.status})
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "run"])
    parser.add_argument("--id")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    try:
        capabilities = discover_capabilities()
        if args.command == "list":
            print(
                canonical(
                    [
                        {
                            "id": c.id,
                            "version": c.manifest["version"],
                            "kind": c.manifest["kind"],
                            "hash": c.identity,
                        }
                        for c in capabilities
                    ]
                )
            )
        else:
            capability = next((c for c in capabilities if c.id == args.id), None)
            if capability is None or args.input is None:
                raise ValueError("run requires a known --id and --input JSON file")
            print(
                canonical(
                    capability.invoke(
                        json.loads(args.input.read_text()), cache=args.cache
                    )
                )
            )
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
