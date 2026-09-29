"""Immutable lineages, rejected evidence and stratified Pareto selection."""

import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from agent_lab.benchmarks import benchmark
from agent_lab.experiments import digest, validate_result
from agent_lab.improvement import decide
from agent_lab.runtime import AgentConfig
from agent_lab.trace import atomic_json


def evaluation_identity(results):
    return digest(
        sorted(
            (
                {
                    "task": r["task"],
                    "protocol": r["protocol"],
                    "hardware": r["hardware"],
                    "software": r["software"],
                }
                for r in results
            ),
            key=digest,
        )
    )


def objectives(results):
    groups = defaultdict(list)
    for result in results:
        validate_result(result)
        groups[(result["protocol"]["split"], result["task"]["id"])].append(result)
    scores = [
        [r["correctness"]["score"] for r in groups[key]] for key in sorted(groups)
    ]
    costs = {
        key: [r[group][key] for r in results]
        for key, group in (
            ("generated_tokens", "work"),
            ("wall_seconds", "measurements"),
            ("peak_ram_bytes", "measurements"),
            ("peak_vram_bytes", "measurements"),
        )
    }
    if any(None in values for values in [*scores, *costs.values()]):
        return None
    return [
        *[sum(values) / len(values) for values in scores],
        -sum(costs["generated_tokens"]),
        -sum(costs["wall_seconds"]),
        -max(costs["peak_ram_bytes"]),
        -max(costs["peak_vram_bytes"]),
    ]


class Archive:
    """Single-writer archive; no rejected evidence or ancestry is overwritten."""

    def __init__(self, root):
        self.root = Path(root)

    def entries(self):
        return sorted(p.stem for p in (self.root / "entries").glob("*.json"))

    def read(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{64}", identifier):
            raise ValueError("Invalid archive identity")
        value = json.loads((self.root / "entries" / f"{identifier}.json").read_text())
        if digest(value) != identifier:
            raise ValueError("Archive evidence drift")
        return value

    def add(self, config, results, *, parents=(), mutation):
        if not results or not mutation or len(mutation) > 1280:
            raise ValueError("Archive requires bounded mutation and evidence")
        expected = {
            k: v
            for k, v in asdict(config).items()
            if k not in ("seed", "max_steps", "max_tool_calls")
        }
        for result in results:
            validate_result(result)
            actual = {
                k: v
                for k, v in result["configuration"].items()
                if k not in ("seed", "max_steps", "max_tool_calls")
            }
            if actual != expected:
                raise ValueError("Evidence configuration differs from candidate")
            for budget in ("max_steps", "max_tool_calls"):
                if result["configuration"][budget] > getattr(config, budget):
                    raise ValueError("Evidence exceeds candidate budget")
        parent_records = [self.read(parent) for parent in parents]
        if parent_records:
            decision = decide(parent_records[0]["results"], results)
        else:
            if {r["protocol"]["split"] for r in results} != {"development", "held-out"}:
                raise ValueError("Root requires development and held-out evidence")
            decision = {"status": "baseline"}
        value = {
            "schema_version": 1,
            "configuration": asdict(config),
            "parents": list(parents),
            "mutation": mutation,
            "decision": decision,
            "results": results,
            "evaluation_id": evaluation_identity(results),
            "ordinal": len(self.entries()) + 1,
        }
        identifier = digest(value)
        atomic_json(self.root / "entries" / f"{identifier}.json", value)
        return identifier

    def _regressions(self):
        records = []
        for path in (self.root / "regressions").glob("*.json"):
            value = json.loads(path.read_text())
            if digest(value) != path.stem:
                raise ValueError("Regression evidence drift")
            records.append(value)
        return records

    def regression_due(self, *, interval=3):
        if type(interval) is not int or interval < 1:
            raise ValueError("Regression interval must be positive")
        path = self.root / "cycle.json"
        checked = 0
        if path.exists():
            value = json.loads(path.read_text())
            expected = value.pop("sha256")
            if digest(value) != expected:
                raise ValueError("Regression cycle drift")
            checked = value["archive_ordinal"]
        return len(self.entries()) - checked >= interval

    def record_regression(self, identifier, results, *, suite_id):
        entry = self.read(identifier)
        for result in results:
            validate_result(result)
            config = {k: v for k, v in result["configuration"].items() if k != "seed"}
            expected = {k: v for k, v in entry["configuration"].items() if k != "seed"}
            if config != expected:
                raise ValueError("Regression configuration drift")
        value = {
            "schema_version": 1,
            "candidate": identifier,
            "suite_id": suite_id,
            "results": results,
            "archive_ordinal": len(self.entries()),
            "passed": bool(results)
            and all(r["correctness"]["status"] == "pass" for r in results),
        }
        atomic_json(self.root / "regressions" / f"{digest(value)}.json", value)
        return value

    def frontier(self):
        groups = defaultdict(list)
        regressions = self._regressions()
        failed = {r["candidate"] for r in regressions if not r["passed"]}
        for identifier in self.entries():
            entry = self.read(identifier)
            groups[entry["evaluation_id"]].append((identifier, entry))
        reports = []
        for key, entries in groups.items():
            eligible, excluded = {}, {}
            for identifier, entry in entries:
                point = objectives(entry["results"])
                if entry["decision"]["status"] == "reject" or identifier in failed:
                    excluded[identifier] = "rejected or broader regression failed"
                elif point is None:
                    excluded[identifier] = "unknown correctness or objective"
                else:
                    eligible[identifier] = point
            survivors = [
                identifier
                for identifier, point in eligible.items()
                if not any(
                    all(a >= b for a, b in zip(other, point, strict=True))
                    and other != point
                    for other in eligible.values()
                )
            ]
            reports.append(
                {"evaluation_id": key, "entries": survivors, "excluded": excluded}
            )
        return reports

    async def evaluate_survivors(
        self,
        adapter_factory,
        tasks,
        *,
        output_root,
        interval=3,
        hardware=None,
        software=None,
    ):
        if not self.regression_due(interval=interval):
            return []
        reports = []
        for group in self.frontier():
            for identifier in group["entries"]:
                config = AgentConfig(**self.read(identifier)["configuration"])
                results = [
                    await benchmark(
                        task,
                        adapter_factory(config, task),
                        config,
                        output_root=output_root,
                        hardware=hardware,
                        software=software,
                    )
                    for task in tasks
                ]
                reports.append(
                    self.record_regression(identifier, results, suite_id=digest(tasks))
                )
        cycle = {
            "archive_ordinal": len(self.entries()),
            "suite_id": digest(tasks),
            "regressions": [digest(report) for report in reports],
        }
        atomic_json(self.root / "cycle.json", {**cycle, "sha256": digest(cycle)})
        return reports


def import_improvement(root, bundle):
    """Retain measured rejected lineages without copying their trace artifacts."""
    archive = Archive(root)
    bundle = Path(bundle)
    results = [
        json.loads(line) for line in (bundle / "results.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((bundle / "manifest.json").read_text())
    if digest(results) != manifest["results_sha256"]:
        raise ValueError("Improvement bundle drift")
    midpoint = len(results) // 2
    baseline, candidate = results[:midpoint], results[midpoint:]
    if decide(baseline, candidate) != json.loads(
        (bundle / "decision.json").read_text()
    ):
        raise ValueError("Improvement decision drift")
    root_id = archive.add(
        AgentConfig(**baseline[0]["configuration"]),
        baseline,
        mutation="measured baseline",
    )
    proposal = json.loads((bundle / "proposal.json").read_text())
    child = archive.add(
        AgentConfig(**proposal["configuration"]),
        candidate,
        parents=[root_id],
        mutation=proposal["mutation"]["reason"],
    )
    report = {
        "schema_version": 1,
        "root": root_id,
        "candidate": child,
        "frontier": archive.frontier(),
        "source_bundle_sha256": digest(manifest),
    }
    atomic_json(Path(root) / "report.json", report)
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--import-improvement", type=Path)
    args = parser.parse_args()
    from agent_lab.experiments import canonical

    print(
        canonical(
            import_improvement(args.archive, args.import_improvement)
            if args.import_improvement
            else Archive(args.archive).frontier()
        )
    )


if __name__ == "__main__":
    main()
