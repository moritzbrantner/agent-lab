"""Authoritative local session/run storage; terminal surfaces only reference it."""

import asyncio
import copy
import fcntl
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import signal
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from agent_lab.backends import OllamaAdapter
from agent_lab.baselines import inventory, validate_manifest
from agent_lab.checkpoints import CheckpointStore, run_checkpointed
from agent_lab.experiments import ROOT, canonical, digest
from agent_lab.permissions import Grant, LocalTools, PermissionDenied, Policy
from agent_lab.runtime import AgentConfig, AgentState, FunctionAdapter, Reply, ToolCall
from agent_lab.telemetry import EnergyCounter, Telemetry, hardware_profile
from agent_lab.trace import Trace, atomic_json, load_trace

CORE = (
    "workspace_store.py",
    "runtime.py",
    "context.py",
    "checkpoints.py",
    "permissions.py",
    "backends.py",
    "trace.py",
)


def core_identity():
    return {
        "python": platform.python_version(),
        "runtime_source_sha256": digest(
            {name: (ROOT / "agent_lab" / name).read_text() for name in CORE}
        ),
    }


def _read(path):
    return json.loads(Path(path).read_text())


def _tick(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _initialize(self):
        marker = self.root / ".agent-lab-workspace.json"
        if self.root.exists() and any(self.root.iterdir()) and not marker.exists():
            raise ValueError("Workspace history requires an empty dedicated directory")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        if not marker.exists():
            atomic_json(marker, {"schema_version": 1, "application": "agent-lab"})

    def _directory(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{16}", identifier):
            raise ValueError("Invalid session identity")
        path = self.root / "sessions" / identifier
        if path.is_symlink():
            raise ValueError("Session directory cannot be a symlink")
        return path

    def session(self, identifier):
        value = _read(self._directory(identifier) / "session.json")
        if value["schema_version"] != 1 or value["id"] != identifier:
            raise ValueError("Session identity drift")
        return value

    def _save_session(self, value):
        atomic_json(self._directory(value["id"]) / "session.json", value)

    def sessions(self):
        return [
            self.session(p.name)
            for p in sorted((self.root / "sessions").glob("*"))
            if p.is_dir()
        ]

    def start(self, name, project, *, profile="medium-q4-gpu"):
        project = Path(project).resolve(strict=True)
        if not project.is_dir() or project.is_relative_to(self.root):
            raise ValueError("Select a project outside private session storage")
        if not isinstance(name, str) or not 1 <= len(name) <= 128:
            raise ValueError("Session name must be bounded")
        if profile == "fixture-demo":
            config = AgentConfig(model="fixture-demo", request_timeout=15)
        else:
            manifest = _read(ROOT / "configurations/local-baselines-v1.json")
            if profile not in manifest["profiles"]:
                raise ValueError("Unknown pinned model profile")
            config = replace(
                AgentConfig(**manifest["profiles"][profile]), request_timeout=15
            )
            validate_manifest(asdict(config), *inventory())
        self._initialize()
        identifier = secrets.token_hex(8)
        directory = self._directory(identifier)
        directory.mkdir(parents=True, mode=0o700)
        grants = [asdict(Grant("repositories", str(project), "read"))]
        value = {
            "schema_version": 1,
            "id": identifier,
            "name": name,
            "project": str(project),
            "profile": profile,
            "configuration": asdict(config),
            "permissions": grants,
            "attachments": [],
            "memory_enabled": False,
            "runs": [],
            "latest_run": None,
            "owner_actions": [
                {
                    "action": "select_project",
                    "approval_id": "owner-start:" + identifier,
                    "owner_uid": os.getuid(),
                    "resource_sha256": digest(str(project)),
                }
            ],
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._save_session(value)
        return value

    def attach(self, identifier, source):
        with self._lease(identifier):
            return self._attach(identifier, source)

    def grant(self, identifier, grant, *, reason):
        with self._lease(identifier):
            return self._grant(identifier, grant, reason=reason)

    def memory(self, identifier, **kwargs):
        with self._lease(identifier):
            return self._memory(identifier, **kwargs)

    def forget(self, identifier):
        with self._lease(identifier):
            return self._forget(identifier)

    def _attach(self, identifier, source):
        session = self.session(identifier)
        source = Path(source).resolve(strict=True)
        native = LocalTools(Policy([Grant("documents", str(source.parent), "read")]))
        text = native.read(
            {"scope": "documents", "root": str(source.parent), "path": source.name}
        )
        if len(text.encode()) > 16384:
            raise ValueError("Text attachments are limited to 16 KiB")
        attachment_id = hashlib.sha256(text.encode()).hexdigest()
        directory = self._directory(identifier) / "attachments"
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / attachment_id
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        attachment = {"id": attachment_id, "name": source.name, "sha256": attachment_id}
        session.setdefault("owner_actions", []).append(
            {
                "action": "attach_document",
                "approval_id": "owner-attach:" + secrets.token_hex(8),
                "owner_uid": os.getuid(),
                "resource_sha256": attachment_id,
            }
        )
        session["attachments"] = [
            a for a in session["attachments"] if a["id"] != attachment_id
        ] + [attachment]
        grant = asdict(Grant("documents", str(directory), "read"))
        if grant not in session["permissions"]:
            session["permissions"].append(grant)
        self._save_session(session)
        return attachment

    def preferences(self, *, language=None, theme=None):
        path = self.root / "preferences.json"
        values = _read(path) if path.exists() else {"language": "en", "theme": "system"}
        if language is not None:
            if language not in ("en", "de", "es"):
                raise ValueError("Unknown language")
            values["language"] = language
        if theme is not None:
            if theme not in ("light", "dark", "system"):
                raise ValueError("Unknown theme")
            values["theme"] = theme
        if language is not None or theme is not None:
            self._initialize()
            atomic_json(path, values)
        return values

    def _grant(self, identifier, grant, *, reason):
        session = self.session(identifier)
        policy = Policy(Grant(**g) for g in session["permissions"])
        approval_id = "owner-cli:" + secrets.token_hex(8)
        elevated = policy.elevate(grant, reason=reason, approval_id=approval_id)
        session["permissions"] = elevated.document()
        session.setdefault("owner_actions", []).append(
            {
                "action": "grant",
                "grant": asdict(grant),
                "reason": reason,
                "approval_id": approval_id,
                "owner_uid": os.getuid(),
            }
        )
        self._save_session(session)
        return elevated.document()

    def _memory(self, identifier, *, enabled=None, clear=False):
        session = self.session(identifier)
        directory = self._directory(identifier) / "memory"
        if clear and directory.exists():
            shutil.rmtree(directory)
        if clear and session["memory_enabled"]:
            directory.mkdir(mode=0o700, exist_ok=True)
        if enabled is not None:
            session.setdefault("owner_actions", []).append(
                {
                    "action": "memory_control",
                    "approval_id": "owner-memory:" + secrets.token_hex(8),
                    "owner_uid": os.getuid(),
                    "resource_sha256": digest(str(directory)),
                    "enabled": bool(enabled),
                }
            )
            session["memory_enabled"] = bool(enabled)
            session["permissions"] = [
                g for g in session["permissions"] if g["scope"] != "memory"
            ]
            if enabled:
                directory.mkdir(mode=0o700, exist_ok=True)
                session["permissions"] += [
                    asdict(Grant("memory", str(directory), op))
                    for op in ("read", "write")
                ]
            self._save_session(session)
        return {"enabled": session["memory_enabled"], "cleared": clear}

    def _adapter(self, configuration):
        config = AgentConfig(**configuration["agent"])
        if config.backend == "fixture":
            attachments = configuration["attachments"]

            def respond(messages, _):
                if messages[-1]["role"] == "tool":
                    return Reply(
                        str(messages[-1]["content"]), input_tokens=0, output_tokens=0
                    )
                call = (
                    ToolCall("read_attachment", {"id": attachments[0]["id"]})
                    if attachments
                    else ToolCall("read_project", {"path": "README.md"})
                )
                return Reply(calls=(call,), input_tokens=0, output_tokens=0)

            return FunctionAdapter(respond)
        validate_manifest(asdict(config), *inventory())
        return OllamaAdapter()

    def _tools(self, configuration, run_root, emit):
        policy = Policy([Grant(**g) for g in configuration["permissions"]], emit)
        native = LocalTools(policy)
        project = Path(configuration["project"])
        attachments = {a["id"]: a for a in configuration["attachments"]}

        def read_project(args):
            path = Path(args["path"])
            target = (project / path).resolve()
            if any(
                part.startswith(".") or part in ("node_modules", "__pycache__")
                for part in path.parts
            ) or target.is_relative_to(self.root):
                raise PermissionDenied("Private project path unavailable")
            text = native.read(
                {"scope": "repositories", "root": str(project), "path": str(path)}
            )
            if len(text.encode()) > 16384:
                raise ValueError("Project text read exceeds 16 KiB")
            return text

        def list_project(_):
            policy.require("repositories", str(project), "read")
            return sorted(
                p.name
                for p in project.iterdir()
                if not p.name.startswith(".") and not p.is_symlink()
            )[:100]

        def read_attachment(args):
            identifier = args["id"]
            if identifier not in attachments:
                raise PermissionDenied("Attachment not in run configuration")
            root = Path(configuration["attachment_root"])
            text = native.read(
                {"scope": "documents", "root": str(root), "path": identifier}
            )
            if (
                hashlib.sha256(text.encode()).hexdigest()
                != attachments[identifier]["sha256"]
            ):
                raise ValueError("Attachment drift")
            return text

        def remember(args):
            if not configuration["memory_enabled"] or not re.fullmatch(
                r"[a-z0-9-]{1,64}", args["key"]
            ):
                raise PermissionDenied("Memory disabled or key invalid")
            return native.write(
                {
                    "scope": "memory",
                    "root": configuration["memory_root"],
                    "path": args["key"],
                    "content": args["value"],
                }
            )

        def recall(args):
            if not configuration["memory_enabled"] or not re.fullmatch(
                r"[a-z0-9-]{1,64}", args["key"]
            ):
                raise PermissionDenied("Memory disabled or key invalid")
            return native.read(
                {
                    "scope": "memory",
                    "root": configuration["memory_root"],
                    "path": args["key"],
                }
            )

        def add(args):
            numbers = args["numbers"]
            if (
                not isinstance(numbers, list)
                or len(numbers) > 1024
                or any(type(n) is not int for n in numbers)
            ):
                raise ValueError("sum requires at most 1024 integers")
            return sum(numbers)

        registry = {
            "list_project": list_project,
            "read_project": read_project,
            "read_attachment": read_attachment,
            "remember": remember,
            "recall": recall,
            "sum": add,
            "run_process": native.process,
        }
        return registry, {
            "list_project",
            "read_project",
            "read_attachment",
            "recall",
            "sum",
        }

    def _store(self, run_root, configuration, config):
        return CheckpointStore(
            run_root / "checkpoint.json",
            configuration["task_sha256"],
            config,
            idempotent_tools={
                "list_project",
                "read_project",
                "read_attachment",
                "recall",
                "sum",
            },
            artifact_root=run_root,
            tool_identity={
                "permissions": configuration["permissions"],
                "runtime": configuration["software"],
            },
        )

    def _load_latest(self, identifier):
        session = self.session(identifier)
        if session["latest_run"] is None:
            raise ValueError("Session has no run")
        root = self._directory(identifier) / "runs" / session["latest_run"]
        configuration = _read(root / "configuration.json")
        config = AgentConfig(**configuration["agent"])
        state, trace = self._store(root, configuration, config).load()
        return session, root, configuration, config, state, trace

    @contextmanager
    def _lease(self, identifier):
        self.session(identifier)
        descriptor = os.open(
            self._directory(identifier) / "lease", os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Session already has an active operation") from None
            yield
        finally:
            os.close(descriptor)

    async def execute(self, identifier, prompt, **kwargs):
        with self._lease(identifier):
            return await self._execute(identifier, prompt, **kwargs)

    async def resume(self, identifier, **kwargs):
        with self._lease(identifier):
            return await self._resume(identifier, **kwargs)

    def abandon(self, identifier):
        with self._lease(identifier):
            session = self.session(identifier)
            if session["latest_run"] is None:
                raise ValueError("No run to abandon")
            session.setdefault("closed_runs", []).append(session["latest_run"])
            session.setdefault("owner_actions", []).append(
                {"action": "abandon", "run_id": session["latest_run"]}
            )
            self._save_session(session)
            return {"abandoned": session["latest_run"]}

    async def _execute(self, identifier, prompt, *, after_save=None, on_event=None):
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 16384:
            raise ValueError("Enter a bounded message")
        session = self.session(identifier)
        if session["latest_run"]:
            _, _, _, _, previous, _ = self._load_latest(identifier)
            if previous.status != "completed" and session[
                "latest_run"
            ] not in session.get("closed_runs", []):
                raise ValueError(
                    "Resume or forget the unfinished run before starting a turn"
                )
            previous_messages = (
                copy.deepcopy(previous.messages)
                if previous.status == "completed"
                else []
            )
        else:
            previous_messages = []
        run_id = secrets.token_hex(12)
        root = self._directory(identifier) / "runs" / run_id
        root.mkdir(parents=True, mode=0o700)
        configuration = {
            "schema_version": 1,
            "agent": session["configuration"],
            "project": session["project"],
            "permissions": session["permissions"],
            "owner_actions": session.get("owner_actions", []),
            "attachments": session["attachments"],
            "attachment_root": str(self._directory(identifier) / "attachments"),
            "memory_root": str(self._directory(identifier) / "memory"),
            "memory_enabled": session["memory_enabled"],
            "response_language": self.preferences()["language"],
            "software": core_identity(),
            "hardware": hardware_profile(),
        }
        task = (
            "Return JSON replies with content and calls. Tools: list_project({}), "
            "read_project(path: relative file), read_attachment(id: attachment id), "
            "sum(numbers: array), remember(key,value), recall(key), run_process(argv). "
            "Only declared permissions apply. Memory is "
            + str(session["memory_enabled"])
            + ". Attachments: "
            + canonical(session["attachments"])
            + "\nUser: "
            + prompt
        )
        task += "\nReply in " + configuration["response_language"] + "."
        for name in ("AGENTS.md", "CLAUDE.md"):
            path = Path(session["project"]) / name
            if (
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_size <= 16384
            ):
                task += "\nRepository instructions " + name + ":\n" + path.read_text()
        configuration["task_sha256"] = digest(task)
        atomic_json(root / "configuration.json", configuration)
        trace = Trace(
            run_id,
            "workspace-session",
            digest(configuration),
            parent_run_id=session["latest_run"],
        )
        trace.artifact("configuration.json", root / "configuration.json")
        for action in configuration["owner_actions"]:
            trace.emit(
                "authority",
                {
                    "action": action["action"],
                    "approval_id": action.get("approval_id"),
                    "owner_uid": action.get("owner_uid"),
                    "resource_sha256": action.get(
                        "resource_sha256", digest(action.get("grant", {}))
                    ),
                },
            )
        state = AgentState(
            task,
            messages=previous_messages + [{"role": "user", "content": task}]
            if previous_messages
            else [],
        )
        session["latest_run"] = run_id
        session["runs"].append(run_id)
        self._save_session(session)
        config = AgentConfig(**configuration["agent"])
        self._store(root, configuration, config).save(state, "pending", trace)
        return await self._drive(
            identifier,
            root,
            configuration,
            config,
            state,
            trace,
            after_save=after_save,
            on_event=on_event,
        )

    async def _resume(self, identifier, *, after_save=None, on_event=None):
        session, root, configuration, config, state, trace = self._load_latest(
            identifier
        )
        if (
            configuration["software"] != core_identity()
            or configuration["permissions"] != session["permissions"]
        ):
            raise ValueError("Resume requires original runtime and permissions")
        return await self._drive(
            identifier,
            root,
            configuration,
            config,
            state,
            trace,
            after_save=after_save,
            on_event=on_event,
        )

    async def _drive(
        self,
        identifier,
        root,
        configuration,
        config,
        state,
        trace,
        *,
        after_save=None,
        on_event=None,
    ):
        active = self._directory(identifier) / "active.json"
        if active.exists():
            recorded = _read(active)
            try:
                live = _tick(recorded["pid"]) == recorded["tick"]
            except OSError:
                live = False
            if live:
                raise ValueError("Session already has an active run")
            active.unlink()
        meter = Telemetry(energy=EnergyCounter.discover())

        raw_emit = trace.emit

        def emit(kind, payload):
            raw_emit(kind, payload)
            meter.emit(kind, payload)
            if on_event:
                on_event(kind, payload)

        trace.emit = emit
        failure = None
        try:
            meter.start()
            tools, _ = self._tools(configuration, root, emit)
            adapter = self._adapter(configuration)
            atomic_json(
                active,
                {
                    "pid": os.getpid(),
                    "tick": _tick(os.getpid()),
                    "run_id": trace.identity["run_id"],
                },
            )
            await run_checkpointed(
                adapter,
                tools,
                config,
                state,
                trace,
                self._store(root, configuration, config),
                after_save=after_save,
            )
        except asyncio.CancelledError, KeyboardInterrupt:
            if state.status != "completed":
                state.status = "cancelled"
            raise
        except Exception as error:
            failure = type(error).__name__
            state.status = "failed"
            raw_emit("failure", {"category": failure})
        finally:
            trace.emit = raw_emit
            measurements, sources = meter.stop()
            trace.emit("measurements", {"values": measurements, "sources": sources})
            self._store(root, configuration, config).save(
                state, "session_boundary", trace
            )
            trace.save(root / "trace.json")
            atomic_json(
                root / "summary.json",
                {
                    "status": state.status,
                    "steps": state.steps,
                    "model_calls": state.model_calls,
                    "tool_calls": state.tool_calls,
                    "failure": failure,
                    "measurements": measurements,
                    "measurement_sources": sources,
                },
            )
            active.unlink(missing_ok=True)
        return asdict(state)

    def inspect(self, identifier):
        _, root, configuration, _, state, _ = self._load_latest(identifier)
        trace = (
            load_trace(root / "trace.json", artifact_root=root)
            if (root / "trace.json").exists()
            else None
        )
        return {
            "configuration": configuration,
            "state": asdict(state),
            "trace": trace,
            "summary": _read(root / "summary.json")
            if (root / "summary.json").exists()
            else None,
        }

    def export_config(self, identifier):
        return self.inspect(identifier)["configuration"]

    def history(self, identifier):
        session = self.session(identifier)
        values = []
        for run_id in session["runs"]:
            path = self._directory(identifier) / "runs" / run_id / "summary.json"
            values.append(
                {
                    "run_id": run_id,
                    **(_read(path) if path.exists() else {"status": "pending"}),
                }
            )
        return values

    def cancel(self, identifier):
        active = _read(self._directory(identifier) / "active.json")
        if not hasattr(os, "pidfd_open"):
            raise ValueError("Use Ctrl+C on this platform")
        descriptor = os.pidfd_open(active["pid"])
        try:
            if _tick(active["pid"]) != active["tick"]:
                raise ValueError("Active process identity expired")
            signal.pidfd_send_signal(descriptor, signal.SIGINT)
        finally:
            os.close(descriptor)
        return {"cancel_requested": True}

    def _forget(self, identifier):
        directory = self._directory(identifier)
        self.session(identifier)
        if (directory / "active.json").exists():
            raise ValueError("Cancel the active run before forgetting history")
        shutil.rmtree(directory)
        return {"forgotten": identifier}
