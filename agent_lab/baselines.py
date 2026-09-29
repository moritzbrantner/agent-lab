"""Pinned local-model baseline acquisition, measurement and immutable evidence."""

import argparse
import asyncio
import json
import re
import sys
import time
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path

from agent_lab.backends import OllamaAdapter
from agent_lab.benchmarks import benchmark, discover
from agent_lab.experiments import canonical, digest, summarize, validate_result
from agent_lab.runtime import AgentConfig
from agent_lab.telemetry import hardware_profile, software_identity
from agent_lab.trace import atomic_json


def inventory():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    values = []
    for name in ("tags", "version"):
        with opener.open(f"http://127.0.0.1:11434/api/{name}", timeout=15) as response:
            values.append(json.load(response))
    return values[0], values[1]["version"]


def validate_manifest(config, tags, backend_version):
    model = next(
        (item for item in tags["models"] if item["name"] == config["model"]), None
    )
    if model is None or model["digest"] != config["model_digest"]:
        raise ValueError("Pinned model unavailable or changed")
    if backend_version != config["backend_version"]:
        raise ValueError("Backend version drift")
    if model["details"]["quantization_level"] != config["quantization"]:
        raise ValueError("Quantization drift")


def baseline_report(runs):
    groups = {}
    for run in runs:
        validate_result(run)
        config = run["configuration"]
        if (
            config.get("backend") != "ollama"
            or not re.fullmatch(r"[a-f0-9]{64}", config.get("model_digest", ""))
            or not config.get("quantization")
        ):
            raise ValueError(
                "Real baselines require pinned model identity and quantization"
            )
        family = {key: value for key, value in config.items() if key != "seed"}
        key = digest(
            {
                "configuration": family,
                "task": run["task"],
                "hardware": run["hardware"],
                "software": run["software"],
                "split": run["protocol"]["split"],
            }
        )
        groups.setdefault(key, []).append(run)
    report = []
    for key, samples in sorted(groups.items()):
        if len(samples) < 5:
            raise ValueError("Baseline requires at least five samples per stratum")
        report.append(
            {
                "identity": key,
                "task": samples[0]["task"],
                "configuration": {
                    k: v for k, v in samples[0]["configuration"].items() if k != "seed"
                },
                "hardware": samples[0]["hardware"],
                "software": samples[0]["software"],
                "samples": len(samples),
                "passed": sum(r["correctness"]["status"] == "pass" for r in samples),
                "failed": sum(r["correctness"]["status"] == "fail" for r in samples),
                "errors": sum(r["correctness"]["status"] == "error" for r in samples),
                "summary": summarize(samples),
            }
        )
    return report


def capture():
    tags, version = inventory()
    profiles = {}
    for label, name, gpu in (
        ("small-q4-cpu", "qwen2.5-coder:0.5b-instruct-q4_K_M", 0),
        ("small-q4-gpu", "qwen2.5-coder:0.5b-instruct-q4_K_M", 99),
        ("small-q8-gpu", "qwen2.5-coder:0.5b-instruct-q8_0", 99),
        ("medium-q4-gpu", "qwen2.5-coder:7b", 99),
    ):
        model = next((m for m in tags["models"] if m["name"] == name), None)
        if model is None:
            raise ValueError(f"Acquire declared model first: {name}")
        profiles[label] = asdict(
            AgentConfig(
                model=name,
                model_digest=model["digest"],
                backend="ollama",
                backend_version=version,
                quantization=model["details"]["quantization_level"],
                options={"num_gpu": gpu},
            )
        )
    return {"schema_version": 1, "hardware": hardware_profile(), "profiles": profiles}


def unload(model):
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=30) as response:
        json.load(response)


def unload_experiment_models(allowed):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:11434/api/ps", timeout=15) as response:
        loaded = json.load(response)["models"]
    for model in loaded:
        if model["name"] not in allowed:
            raise ValueError("Unrelated model loaded; use a dedicated idle backend")
    for model in loaded:
        unload(model["name"])
    deadline = time.perf_counter() + 10
    while True:
        with opener.open("http://127.0.0.1:11434/api/ps", timeout=15) as response:
            remaining = json.load(response)["models"]
        if not remaining:
            return
        if time.perf_counter() >= deadline:
            raise ValueError("Backend did not unload; baseline isolation unavailable")
        time.sleep(0.2)


async def run_baselines(
    manifest, output, *, repeats=5, split="development", task_ids=None
):
    if (
        manifest.get("schema_version") != 1
        or not manifest.get("profiles")
        or repeats < 5
    ):
        raise ValueError("Invalid baseline manifest/repeat policy")
    if output.exists():
        raise ValueError("Baseline output must be new; evidence is immutable")
    hardware, software = hardware_profile(), software_identity()
    if hardware != manifest["hardware"]:
        raise ValueError("Hardware differs from declared profile")
    output.mkdir(parents=True)
    results = []
    cold_results = []
    tasks = [task for task in discover(split) if not task_ids or task["id"] in task_ids]
    if not tasks:
        raise ValueError("No matching workloads")
    for label, values in sorted(manifest["profiles"].items()):
        config = AgentConfig(**values)
        validate_manifest(values, *inventory())
        await asyncio.to_thread(
            unload_experiment_models,
            {values["model"] for values in manifest["profiles"].values()},
        )
        cold = await benchmark(
            tasks[0],
            OllamaAdapter(),
            replace(config, seed=99),
            repeat=repeats,
            hardware=hardware,
            software=software,
            output_root=output / "runs",
        )
        cold_results.append(cold)
        print(
            canonical(
                {
                    "profile": label,
                    "phase": "cold-load",
                    "status": cold["correctness"]["status"],
                }
            ),
            flush=True,
        )
        for task in tasks:
            for repeat in range(repeats):
                sample = await benchmark(
                    task,
                    OllamaAdapter(),
                    replace(config, seed=repeat),
                    repeat=repeat,
                    hardware=hardware,
                    software=software,
                    output_root=output / "runs",
                )
                results.append(sample)
                print(
                    canonical(
                        {
                            "profile": label,
                            "task": task["id"],
                            "repeat": repeat,
                            "status": sample["correctness"]["status"],
                        }
                    ),
                    flush=True,
                )
        validate_manifest(values, *inventory())
    text = "".join(canonical(result) + "\n" for result in results)
    (output / "results.jsonl").write_text(text)
    (output / "cold-results.jsonl").write_text(
        "".join(canonical(result) + "\n" for result in cold_results)
    )
    atomic_json(output / "summary.json", baseline_report(results))
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "input": manifest,
            "results_sha256": digest(results),
            "cold_results_sha256": digest(cold_results),
            "software": software,
            "repeats": repeats,
            "tasks": [
                {"id": task["id"], "fixture_hash": digest(task)} for task in tasks
            ],
            "conventions_source_revision": "e6acb5310afaf15c0cba24f87108f5f4ad1bedc3",
        },
    )
    return results


def verify_bundle(output):
    manifest = json.loads((output / "manifest.json").read_text())
    results = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    if digest(results) != manifest["results_sha256"]:
        raise ValueError("Baseline evidence drift")
    cold = [
        json.loads(line)
        for line in (output / "cold-results.jsonl").read_text().splitlines()
    ]
    for run in cold:
        validate_result(run)
    if digest(cold) != manifest["cold_results_sha256"]:
        raise ValueError("Cold-load evidence drift")
    if len(cold) != len(manifest["input"]["profiles"]):
        raise ValueError("Incomplete cold-load evidence")
    expected_count = (
        len(manifest["input"]["profiles"])
        * len(manifest["tasks"])
        * manifest["repeats"]
    )
    if len(results) != expected_count:
        raise ValueError("Incomplete baseline evidence")
    if baseline_report(results) != json.loads((output / "summary.json").read_text()):
        raise ValueError("Baseline summary drift")
    return {"valid": True, "samples": len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["capture", "run", "verify"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--task", action="append")
    args = parser.parse_args()
    try:
        if args.command == "capture":
            if args.output.exists():
                raise ValueError("Choose a new manifest path for explicit recapture")
            atomic_json(args.output, capture())
        elif args.command == "verify":
            print(canonical(verify_bundle(args.output)))
        else:
            if args.manifest is None:
                raise ValueError("run requires --manifest")
            asyncio.run(
                run_baselines(
                    json.loads(args.manifest.read_text()),
                    args.output,
                    repeats=args.repeats,
                    task_ids=args.task,
                )
            )
    except (ValueError, OSError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
