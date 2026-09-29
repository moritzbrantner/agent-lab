"""Scoped, sourced measurements; unavailable telemetry is never fabricated."""

import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

from agent_lab.experiments import MEASUREMENTS, ROOT, digest


def _read_rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError, ValueError:
        return None
    return None


def memory_snapshot():
    own = _read_rss(os.getpid())
    if own is None:
        return None, None
    backend_pids = set()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / "cmdline").read_bytes().split(b"\0")
            if command and Path(os.fsdecode(command[0])).name == "ollama":
                backend_pids.add(int(path.name))
        except OSError:
            continue  # Processes may disappear between enumeration and reading.
    ram = own + sum(_read_rss(pid) or 0 for pid in backend_pids)
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ram, None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return ram, None
    if result.returncode:
        return ram, None
    try:
        allocations = [
            line.split(",") for line in result.stdout.splitlines() if line.strip()
        ]
        vram = sum(
            int(memory.strip()) * 1024 * 1024
            for pid, memory in allocations
            if int(pid.strip()) in backend_pids
        )
        return ram, vram
    except ValueError:
        return ram, None


class EnergyCounter:
    """One supported RAPL package counter, with explicit measurement scope."""

    def __init__(
        self, read_microjoules, *, max_microjoules, source="measured:RAPL/package"
    ):
        self.read = read_microjoules
        self.maximum = max_microjoules
        self.source = source
        self.initial = None

    def start(self):
        self.initial = self.read()

    def stop(self):
        final = self.read()
        if self.initial is None or final is None:
            return None
        return ((final - self.initial) % self.maximum) / 1e6

    @classmethod
    def discover(cls):
        for path in sorted(Path("/sys/class/powercap").glob("intel-rapl:[0-9]")):
            try:
                maximum = int((path / "max_energy_range_uj").read_text())
                int((path / "energy_uj").read_text())
            except OSError, ValueError:
                continue

            def read(path=path):
                try:
                    return int((path / "energy_uj").read_text())
                except OSError, ValueError:
                    return None

            return cls(
                read,
                max_microjoules=maximum,
                source=f"measured:RAPL/{path.name}/package-only",
            )
        return None


class Telemetry:
    def __init__(
        self,
        *,
        clock=time.perf_counter,
        cpu_clock=time.process_time,
        memory_probe=memory_snapshot,
        energy=None,
        periodic=True,
    ):
        self.clock, self.cpu_clock = clock, cpu_clock
        self.memory_probe, self.energy, self.periodic = memory_probe, energy, periodic
        self.values = dict.fromkeys(MEASUREMENTS)
        self.sources = {}
        self.request_started = None
        self.tool_started = None
        self.model_requests = 0
        self.model_responses = 0
        self.model_metrics_available = True
        self.inference = self.setup = self.tool_time = 0
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        ram, vram = self.memory_probe()
        for key, value in (("peak_ram_bytes", ram), ("peak_vram_bytes", vram)):
            if value is not None:
                self.values[key] = max(self.values[key] or 0, value)

    def _poll(self):
        while not self.stop_event.wait(0.2):
            self.sample()

    def start(self):
        self.started, self.cpu_started = self.clock(), self.cpu_clock()
        if self.energy:
            self.energy.start()
        self.sample()
        if self.periodic:
            self.thread = threading.Thread(target=self._poll)
            self.thread.start()

    def emit(self, kind, payload):
        if kind == "model_request":
            self.model_requests += 1
        elif kind == "model_response":
            self.model_responses += 1
            metrics = payload.get("measurements", {})
            if all(
                key in metrics
                for key in ("load_duration", "prompt_eval_duration", "eval_duration")
            ):
                self.setup += metrics["load_duration"]
                self.inference += (
                    metrics["prompt_eval_duration"] + metrics["eval_duration"]
                )
            else:
                self.model_metrics_available = False
        elif kind == "tool_request":
            self.tool_started = self.clock()
        elif kind == "tool_result" and self.tool_started is not None:
            self.tool_time += self.clock() - self.tool_started
            self.tool_started = None
        if not self.periodic:
            self.sample()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.sample()
        ended = self.clock()
        self.values.update(
            wall_seconds=ended - self.started,
            cpu_seconds=self.cpu_clock() - self.cpu_started,
            tool_seconds=self.tool_time
            + (ended - self.tool_started if self.tool_started is not None else 0),
        )
        self.sources.update(
            wall_seconds="measured:perf_counter/end-to-end",
            cpu_seconds="measured:process_time/harness-only",
            tool_seconds="measured:perf_counter/tool-boundaries",
        )
        if self.model_metrics_available and self.model_requests == self.model_responses:
            self.values.update(
                inference_seconds=self.inference, setup_seconds=self.setup
            )
            self.sources.update(
                inference_seconds="reported:backend/prompt+generation",
                setup_seconds="reported:backend/model-load",
            )
        for key, source in (
            (
                "peak_ram_bytes",
                "sampled:/proc/VmRSS/harness+all-ollama-processes/200ms",
            ),
            (
                "peak_vram_bytes",
                "sampled:nvidia-smi/all-ollama-process-allocations/200ms",
            ),
        ):
            if self.values[key] is not None:
                self.sources[key] = source
        if self.energy:
            self.values["energy_joules"] = self.energy.stop()
            if self.values["energy_joules"] is not None:
                self.sources["energy_joules"] = self.energy.source
        return self.values, self.sources


def hardware_profile():
    profile = {
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "cpu": None,
        "ram_bytes": None,
        "gpus": None,
    }
    if Path("/proc/cpuinfo").exists():
        profile["cpu"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
        profile["ram_bytes"] = next(
            (
                int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal:")
            ),
            None,
        )
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                profile["gpus"] = result.stdout.strip().splitlines()
        except OSError, subprocess.TimeoutExpired:
            profile["gpus"] = None
    return {"profile": digest(profile), **profile}


def software_identity():
    files = {
        path.relative_to(ROOT).as_posix(): path.read_text()
        for path in sorted((ROOT / "agent_lab").glob("*.py"))
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_hash": digest(files),
    }
