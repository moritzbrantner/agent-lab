"""Measure bounded candidates, deterministic checking and rejected work."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from agent_lab.backends import OllamaAdapter
from agent_lab.baselines import inventory, unload_experiment_models, validate_manifest
from agent_lab.benchmarks import load_task
from agent_lab.experiments import ROOT, canonical, compare, digest, summarize
from agent_lab.generation import generate_verify
from agent_lab.runtime import AgentConfig
from agent_lab.telemetry import hardware_profile, software_identity
from agent_lab.trace import atomic_json


async def experiment(output):
    if output.exists():
        raise ValueError("Choose a new evidence directory")
    manifest = json.loads((ROOT / "configurations/local-baselines-v1.json").read_text())
    cheap = AgentConfig(**manifest["profiles"]["small-q4-cpu"])
    strong = AgentConfig(**manifest["profiles"]["medium-q4-gpu"])
    for config in (cheap, strong):
        validate_manifest(asdict(config), *inventory())
    hardware, software = hardware_profile(), software_identity()
    if hardware != manifest["hardware"]:
        raise ValueError("Hardware drift")
    allowed = {p["model"] for p in manifest["profiles"].values()}
    task = load_task("development", "batch-sum")
    groups, cold = {}, []
    output.mkdir(parents=True)
    for label, schedule in {
        "single": [strong],
        "regenerate": [cheap, cheap, strong],
    }.items():
        await asyncio.to_thread(unload_experiment_models, allowed)
        groups[label] = []
        for repeat in range(6):
            report = await generate_verify(
                task,
                schedule,
                lambda config: OllamaAdapter(),
                repeat=repeat,
                hardware=hardware,
                software=software,
                output_root=output / label / str(repeat),
            )
            result = report["result"]
            (cold if repeat == 0 else groups[label]).append(result)
            print(
                canonical(
                    {
                        "configuration": label,
                        "repeat": repeat,
                        "status": result["correctness"]["status"],
                    }
                ),
                flush=True,
            )
    report = {
        label: {
            "passed": sum(r["correctness"]["status"] == "pass" for r in group),
            "summary": summarize(group),
        }
        for label, group in groups.items()
    }
    report["comparisons"] = [
        compare(a, b)
        for a, b in zip(groups["single"], groups["regenerate"], strict=True)
    ]
    results = [r for group in groups.values() for r in group]
    (output / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in results))
    atomic_json(output / "cold-results.json", cold)
    atomic_json(output / "report.json", report)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "results_sha256": digest(results),
            "cold_sha256": digest(cold),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(experiment(parser.parse_args().output))


if __name__ == "__main__":
    main()
