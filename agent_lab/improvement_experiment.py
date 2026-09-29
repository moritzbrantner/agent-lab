"""A complete local-model proposed, protected-evaluation scaffolding mutation."""

import argparse
import asyncio
import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from agent_lab.backends import RESPONSE_SCHEMA, OllamaAdapter
from agent_lab.baselines import inventory, unload_experiment_models, validate_manifest
from agent_lab.benchmarks import benchmark, discover, load_task
from agent_lab.experiments import ROOT, canonical, digest
from agent_lab.improvement import decide, materialize, propose
from agent_lab.runtime import AgentConfig
from agent_lab.telemetry import hardware_profile, software_identity
from agent_lab.trace import atomic_json


def authority():
    paths = [
        *sorted((ROOT / "benchmarks").rglob("*.json")),
        ROOT / "agent_lab/evaluators.py",
        ROOT / "agent_lab/evaluator_worker.py",
        ROOT / "schemas/task-v1.json",
    ]
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


async def experiment(output):
    if output.exists():
        raise ValueError("Choose a new evidence directory")
    manifest = json.loads((ROOT / "configurations/local-baselines-v1.json").read_text())
    baseline = AgentConfig(**manifest["profiles"]["medium-q4-gpu"])
    validate_manifest(asdict(baseline), *inventory())
    hardware, software, protected = hardware_profile(), software_identity(), authority()
    if hardware != manifest["hardware"]:
        raise ValueError("Hardware drift")
    output.mkdir(parents=True)
    evidence = [
        json.loads(line)
        for line in (ROOT / "evidence/context/local-v1/results.jsonl")
        .read_text()
        .splitlines()
    ]
    schema = copy.deepcopy(RESPONSE_SCHEMA)
    schema["properties"]["calls"]["maxItems"] = 1
    schema["properties"]["calls"]["items"]["properties"] = {
        "name": {"const": "propose"},
        "arguments": {
            "type": "object",
            "additionalProperties": False,
            "required": ["changes", "reason"],
            "properties": {
                "reason": {"type": "string", "minLength": 1, "maxLength": 1280},
                "changes": {
                    "type": "object",
                    "additionalProperties": False,
                    "minProperties": 1,
                    "properties": {
                        "context_policy": {"enum": ["full", "retained", "relevance"]},
                        "feedback": {"type": "string", "maxLength": 1280},
                        "max_retries": {"type": "integer", "minimum": 0, "maximum": 3},
                        "max_output_tokens": {
                            "type": "integer",
                            "minimum": 16,
                            "maximum": 512,
                        },
                    },
                },
            },
        },
    }
    attempts = []
    mutation = None
    for attempt in range(3):
        planner = replace(baseline, seed=42 + attempt, max_output_tokens=512)
        mutation, trace, proposal_work = await propose(
            OllamaAdapter(response_schema=schema), planner, evidence
        )
        attempt_root = output / "proposal-attempts" / str(attempt)
        atomic_json(
            attempt_root / "attempt.json",
            {
                "mutation": mutation,
                "work": proposal_work,
                "configuration": asdict(planner),
                "response_schema_sha256": digest(schema),
            },
        )
        trace.artifact("attempt.json", attempt_root / "attempt.json")
        trace.save(attempt_root / "trace.json")
        attempts.append(proposal_work)
        if mutation is not None:
            break
    if mutation is None:
        raise ValueError(
            "Three proposal attempts produced no valid mutation; evidence retained"
        )
    candidate = materialize(baseline, mutation)
    atomic_json(
        output / "proposal.json",
        {
            "mutation": mutation,
            "work": proposal_work,
            "all_attempts": attempts,
            "configuration": asdict(candidate),
        },
    )
    atomic_json(
        output / "attempt.json", json.loads((attempt_root / "attempt.json").read_text())
    )
    trace.artifact("proposal.json", output / "proposal.json")
    trace.save(output / "proposal-trace.json")
    allowed = {p["model"] for p in manifest["profiles"].values()}
    groups, cold = {}, []
    for label, profile in {"baseline": baseline, "candidate": candidate}.items():
        await asyncio.to_thread(unload_experiment_models, allowed)
        cold.append(
            await benchmark(
                load_task("development", "batch-sum"),
                OllamaAdapter(),
                profile,
                repeat=0,
                hardware=hardware,
                software=software,
                output_root=output / "cold",
            )
        )
        groups[label] = []
        for task in [*discover("development"), *discover("held-out")]:
            for repeat in range(1, 6):
                result = await benchmark(
                    task,
                    OllamaAdapter(),
                    replace(profile, seed=repeat),
                    repeat=repeat,
                    hardware=hardware,
                    software=software,
                    output_root=output / "runs",
                )
                groups[label].append(result)
                print(
                    canonical(
                        {
                            "configuration": label,
                            "task": task["id"],
                            "repeat": repeat,
                            "status": result["correctness"]["status"],
                        }
                    ),
                    flush=True,
                )
    if authority() != protected:
        raise ValueError("Protected evaluation authority changed")
    decision = decide(groups["baseline"], groups["candidate"])
    results = [r for group in groups.values() for r in group]
    (output / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in results))
    atomic_json(output / "cold-results.json", cold)
    atomic_json(output / "decision.json", decision)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "results_sha256": digest(results),
            "cold_sha256": digest(cold),
            "protected_authority": protected,
            "proposal_sha256": digest(mutation),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(experiment(parser.parse_args().output))


if __name__ == "__main__":
    main()
