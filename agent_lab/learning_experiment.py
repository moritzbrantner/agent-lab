"""Offline small-model LoRA candidate versus untuned development/held-out baseline."""

import argparse
import asyncio
import importlib.metadata
import json
import time
from dataclasses import replace
from pathlib import Path

from agent_lab.baselines import unload_experiment_models
from agent_lab.benchmarks import benchmark, discover, load_task
from agent_lab.experiments import ROOT, canonical, digest
from agent_lab.improvement import decide
from agent_lab.learning import build_data, entry_gate
from agent_lab.peft_backend import LocalPeftAdapter, environment, file_hashes, train
from agent_lab.runtime import AgentConfig
from agent_lab.telemetry import hardware_profile, software_identity
from agent_lab.trace import Trace, atomic_json


async def experiment(base, output):
    if output.exists():
        raise ValueError("Choose a new evidence directory")
    started = time.perf_counter()
    gate = entry_gate()
    plan = json.loads((ROOT / "configurations/learning-v1.json").read_text())
    artifact = json.loads(
        (ROOT / "evidence/distillation/local-v1/distilled-artifact.json").read_text()
    )
    data = build_data(artifact, count=plan["examples"], seed=plan["seed"])
    preparation_seconds = time.perf_counter() - started
    torch = environment(plan["seed"])
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "huggingface-hub")
    }
    hardware, software = (
        hardware_profile(),
        {**software_identity(), "packages": packages, "cuda": torch.version.cuda},
    )
    pinned = json.loads((ROOT / "configurations/local-baselines-v1.json").read_text())
    if hardware != pinned["hardware"]:
        raise ValueError("Hardware drift")
    await asyncio.to_thread(
        unload_experiment_models, {p["model"] for p in pinned["profiles"].values()}
    )
    output.mkdir(parents=True)
    atomic_json(output / "gate.json", gate)
    atomic_json(output / "training-data.json", data)
    atomic_json(
        output / "plan.json",
        {**plan, "base_files": file_hashes(base), "packages": packages},
    )
    trace = Trace(
        digest({"data": data["data_sha256"], "plan": plan})[:24],
        "learned-specialization",
        digest(plan),
    )
    trace.emit(
        "authority",
        {
            "scope": "verified-development-training",
            "gate_sha256": digest(gate),
            "data_sha256": data["data_sha256"],
        },
    )
    groups, cold = {}, []
    training = None
    tasks = [
        load_task("development", "sum"),
        load_task("development", "batch-sum"),
        *discover("held-out"),
    ]
    for label in ("untuned", "lora"):
        if label == "lora":
            training = await asyncio.to_thread(
                train, base, data, plan, output / "adapter"
            )
            atomic_json(output / "training-cost.json", training)
        adapter = LocalPeftAdapter(
            base, plan, adapter=output / "adapter" if label == "lora" else None
        )
        config = AgentConfig(
            model=plan["model"],
            model_digest=adapter.identity,
            backend="transformers-peft",
            backend_version=canonical(packages),
            quantization="BF16",
            seed=plan["seed"],
            max_output_tokens=64,
            options={
                "base_revision": plan["revision"],
                "device": plan["device"],
                "attention": plan["attention"],
                "adapter": label,
                "training_data_sha256": data["data_sha256"]
                if label == "lora"
                else None,
            },
        )
        cold.append(
            await benchmark(
                tasks[0],
                adapter,
                replace(config, seed=0),
                repeat=0,
                hardware=hardware,
                software=software,
                output_root=output / "cold",
            )
        )
        groups[label] = []
        try:
            for task in tasks:
                for repeat in range(1, 6):
                    result = await benchmark(
                        task,
                        adapter,
                        replace(config, seed=repeat),
                        repeat=repeat,
                        hardware=hardware,
                        software=software,
                        output_root=output / "runs",
                        parent_run_id=trace.identity["run_id"],
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
        finally:
            adapter.close()
    decision = decide(groups["untuned"], groups["lora"])
    results = groups["untuned"] + groups["lora"]
    (output / "results.jsonl").write_text("".join(canonical(r) + "\n" for r in results))
    atomic_json(output / "cold-results.json", cold)
    report = {
        "schema_version": 1,
        "preparation_seconds": preparation_seconds,
        "training_cost": training,
        "decision": decision,
        "passed": {
            label: {
                task["id"]: sum(
                    r["correctness"]["status"] == "pass"
                    for r in runs
                    if r["task"]["id"] == task["id"]
                )
                for task in tasks
            }
            for label, runs in groups.items()
        },
    }
    atomic_json(output / "report.json", report)
    for name in (
        "gate.json",
        "training-data.json",
        "plan.json",
        "training-cost.json",
        "report.json",
    ):
        trace.artifact(name, output / name)
    for name in file_hashes(output / "adapter"):
        trace.artifact("adapter/" + name, output / "adapter" / name)
    trace.save(output / "trace.json")
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "results_sha256": digest(results),
            "cold_sha256": digest(cold),
            "sourceRevision": plan["sourceRevision"],
        },
    )
    print(
        canonical({"passed": report["passed"], "decision": decision["status"]}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(experiment(args.base, args.output))


if __name__ == "__main__":
    main()
