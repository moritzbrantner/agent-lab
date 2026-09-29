"""Run a versioned suite through a selected local backend."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agent_lab.backends import OllamaAdapter
from agent_lab.benchmarks import benchmark, discover, fixture_adapter
from agent_lab.experiments import canonical
from agent_lab.runtime import AgentConfig


async def execute(args):
    config = (
        AgentConfig(**json.loads(args.configuration.read_text()))
        if args.configuration
        else AgentConfig()
    )
    if config.backend not in ("fixture", "ollama"):
        raise ValueError("Unsupported backend")
    repeats = args.repeats or (1 if config.backend == "fixture" else 5)
    if repeats < 1 or (config.backend != "fixture" and repeats < 5):
        raise ValueError("Stochastic backends require at least five repeats")
    results = []
    for task in discover(args.split):
        if args.task and task["id"] not in args.task:
            continue
        for repeat in range(repeats):
            from dataclasses import replace

            sampled = replace(config, seed=config.seed + repeat)
            adapter = (
                fixture_adapter(task)
                if config.backend == "fixture"
                else OllamaAdapter()
            )
            result = await benchmark(
                task,
                adapter,
                sampled,
                repeat=repeat,
                output_root=args.output,
                retain_content=args.retain_content,
            )
            print(canonical(result), flush=True)
            results.append(result)
    if not results:
        raise ValueError("No matching tasks")
    return 0 if all(r["correctness"]["status"] == "pass" for r in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument(
        "--split", choices=["development", "held-out"], default="development"
    )
    parser.add_argument("--task", action="append")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output", type=Path, default=Path(".runs"))
    parser.add_argument("--retain-content", action="store_true")
    args = parser.parse_args()
    try:
        return asyncio.run(execute(args))
    except (ValueError, OSError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
