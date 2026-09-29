"""Compare repeated model work with a deterministic capability."""

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from agent_lab.backends import OllamaAdapter
from agent_lab.baselines import inventory, unload_experiment_models, validate_manifest
from agent_lab.benchmarks import benchmark, load_task
from agent_lab.capabilities import CapabilityRunner, discover_capabilities
from agent_lab.experiments import ROOT, canonical, compare, digest, summarize
from agent_lab.runtime import AgentConfig
from agent_lab.telemetry import hardware_profile, software_identity
from agent_lab.trace import atomic_json


async def experiment(output):
    if output.exists():
        raise ValueError("Choose a new evidence directory")
    manifest = json.loads((ROOT / "configurations/local-baselines-v1.json").read_text())
    values = manifest["profiles"]["medium-q4-gpu"]
    validate_manifest(values, *inventory())
    config = AgentConfig(**values)
    hardware, software = hardware_profile(), software_identity()
    if hardware != manifest["hardware"]:
        raise ValueError("Hardware drift")
    allowed = {p["model"] for p in manifest["profiles"].values()}
    await asyncio.to_thread(unload_experiment_models, allowed)
    task = load_task("development", "batch-sum")
    output.mkdir(parents=True)
    cold = await benchmark(
        task,
        OllamaAdapter(),
        config,
        repeat=5,
        hardware=hardware,
        software=software,
        output_root=output / "runs",
    )
    reference = []
    for repeat in range(5):
        reference.append(
            await benchmark(
                task,
                OllamaAdapter(),
                replace(config, seed=repeat),
                repeat=repeat,
                hardware=hardware,
                software=software,
                output_root=output / "runs",
            )
        )
    await asyncio.to_thread(unload_experiment_models, allowed)
    capability = discover_capabilities()[0]
    helper_config = AgentConfig(
        model="none",
        model_digest=capability.identity,
        backend="capability",
        options={"capability": capability.id},
    )
    candidates = []
    sources = [r["run_id"] for r in reference if r["correctness"]["status"] == "pass"]
    for repeat in range(5):
        candidates.append(
            await benchmark(
                task,
                None,
                replace(helper_config, seed=repeat),
                repeat=repeat,
                hardware=hardware,
                software=software,
                output_root=output / "runs",
                executor=CapabilityRunner(
                    capability,
                    task["input"],
                    cache=output / "cache",
                    source_runs=sources,
                ),
            )
        )
    report = {
        "correctness_parity": all(
            r["correctness"]["status"] == "pass" for r in reference + candidates
        ),
        "reference": summarize(reference),
        "capability": summarize(candidates),
        "comparisons": [
            compare(a, b) for a, b in zip(reference, candidates, strict=True)
        ],
    }
    results = reference + candidates
    (output / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in results))
    atomic_json(output / "cold-result.json", cold)
    atomic_json(output / "report.json", report)
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "results_sha256": digest(results),
            "capability_hash": capability.identity,
            "source_runs": sources,
        },
    )
    print(
        canonical(
            {
                "correctness_parity": report["correctness_parity"],
                "reference_generated_tokens": report["reference"]["generated_tokens"][
                    "mean"
                ],
                "helper_generated_tokens": report["capability"]["generated_tokens"][
                    "mean"
                ],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(experiment(parser.parse_args().output))


if __name__ == "__main__":
    main()
