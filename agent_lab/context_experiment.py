"""Repeat full/reference and retained/relevance policies on identical workloads."""

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from agent_lab.backends import OllamaAdapter
from agent_lab.baselines import inventory, unload_experiment_models, validate_manifest
from agent_lab.benchmarks import benchmark, load_task
from agent_lab.experiments import ROOT, canonical, compare, digest, summarize
from agent_lab.runtime import AgentConfig
from agent_lab.telemetry import hardware_profile, software_identity
from agent_lab.trace import atomic_json


def context_report(results):
    report = []
    for task_id, policy in (("batch-sum", "retained"), ("retrieve-fact", "relevance")):
        reference = [
            r
            for r in results
            if r["task"]["id"] == task_id
            and r["configuration"]["context_policy"] == "full"
        ]
        candidate = [
            r
            for r in results
            if r["task"]["id"] == task_id
            and r["configuration"]["context_policy"] == policy
        ]
        if len(reference) != 5 or len(candidate) != 5:
            raise ValueError("Context experiment requires five paired samples")
        reference.sort(key=lambda r: r["protocol"]["repeat"])
        candidate.sort(key=lambda r: r["protocol"]["repeat"])
        parity = all(
            r["correctness"]["status"] == "pass" for r in reference + candidate
        )
        summaries = summarize(reference), summarize(candidate)
        inputs = [s["input_tokens"]["mean"] for s in summaries]
        report.append(
            {
                "task": task_id,
                "policy": policy,
                "correctness_parity": parity,
                "input_tokens_reduced": all(v is not None for v in inputs)
                and inputs[1] < inputs[0],
                "reference": summaries[0],
                "candidate": summaries[1],
                "comparisons": [
                    compare(a, b) for a, b in zip(reference, candidate, strict=True)
                ],
            }
        )
    return report


async def experiment(manifest, output):
    if output.exists():
        raise ValueError("Choose a new evidence directory")
    values = manifest["profiles"]["medium-q4-gpu"]
    config = AgentConfig(**values)
    validate_manifest(values, *inventory())
    hardware, software = hardware_profile(), software_identity()
    if hardware != manifest["hardware"]:
        raise ValueError("Hardware profile drift")
    await asyncio.to_thread(
        unload_experiment_models, {p["model"] for p in manifest["profiles"].values()}
    )
    output.mkdir(parents=True)
    results = []
    cold = await benchmark(
        load_task("development", "batch-sum"),
        OllamaAdapter(),
        config,
        repeat=5,
        hardware=hardware,
        software=software,
        output_root=output / "runs",
    )
    for task_id, policy in (
        ("batch-sum", "full"),
        ("batch-sum", "retained"),
        ("retrieve-fact", "full"),
        ("retrieve-fact", "relevance"),
    ):
        for repeat in range(5):
            result = await benchmark(
                load_task("development", task_id),
                OllamaAdapter(),
                replace(config, seed=repeat, context_policy=policy),
                repeat=repeat,
                hardware=hardware,
                software=software,
                output_root=output / "runs",
            )
            results.append(result)
            print(
                canonical(
                    {
                        "task": task_id,
                        "policy": policy,
                        "repeat": repeat,
                        "status": result["correctness"]["status"],
                    }
                ),
                flush=True,
            )
    validate_manifest(values, *inventory())
    (output / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in results))
    atomic_json(output / "cold-result.json", cold)
    atomic_json(output / "report.json", context_report(results))
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "results_sha256": digest(results),
            "hardware": hardware,
            "software": software,
            "conventions_source_revision": "e6acb5310afaf15c0cba24f87108f5f4ad1bedc3",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configurations/local-baselines-v1.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(experiment(json.loads(args.manifest.read_text()), args.output))


if __name__ == "__main__":
    main()
