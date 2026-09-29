"""Explicit local authority, safe file traversal and owned offline process sandbox."""

import asyncio
import hashlib
import os
import secrets as secure_random
import shutil
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from agent_lab.experiments import ROOT, digest
from agent_lab.runtime import RuntimeFailure

SCOPES = {
    "filesystem",
    "process",
    "network",
    "secrets",
    "repositories",
    "documents",
    "memory",
}
FILE_SCOPES = {"filesystem", "repositories", "documents", "memory"}
LIMIT = 1024 * 1024


class PermissionDenied(RuntimeFailure):
    """The supplied owner policy does not authorize this action."""


class SandboxUnavailable(RuntimeFailure):
    """Required native isolation is unavailable; never run unsandboxed."""


@dataclass(frozen=True)
class Grant:
    scope: str
    resource: str
    operation: str

    def __post_init__(self):
        operations = {"process": {"execute"}, "network": {"get"}, "secrets": {"handle"}}
        if (
            self.scope not in SCOPES
            or self.operation
            not in operations.get(self.scope, {"read", "write", "delete"})
            or not isinstance(self.resource, str)
            or not self.resource
        ):
            raise ValueError("Invalid explicit permission grant")
        if self.scope in FILE_SCOPES | {"process"}:
            if not Path(self.resource).is_absolute():
                raise ValueError("File/process grant needs an absolute resource")
            object.__setattr__(self, "resource", str(Path(self.resource).resolve()))


class Policy:
    def __init__(self, grants=(), emit=lambda *_: None):
        self.grants = tuple(
            sorted(set(grants), key=lambda g: (g.scope, g.resource, g.operation))
        )
        self.emit = emit
        self.identity = digest([asdict(g) for g in self.grants])

    def document(self):
        return [asdict(g) for g in self.grants]

    def require(self, scope, resource, operation):
        allowed = Grant(scope, resource, operation) in self.grants
        self.emit(
            "authority",
            {
                "action": "check",
                "scope": scope,
                "operation": operation,
                "resource_sha256": digest(resource),
                "policy_sha256": self.identity,
                "allowed": allowed,
            },
        )
        if not allowed:
            raise PermissionDenied(f"Denied {scope}/{operation}")

    def elevate(self, grant, *, reason, approval_id):
        if (
            not reason
            or len(reason) > 1280
            or not approval_id
            or len(approval_id) > 128
        ):
            raise ValueError(
                "Owner elevation needs bounded reason and approval identity"
            )
        self.emit(
            "authority",
            {
                "action": "elevation",
                "scope": grant.scope,
                "operation": grant.operation,
                "resource_sha256": digest(grant.resource),
                "reason": reason,
                "approval_id": approval_id,
            },
        )
        return Policy([*self.grants, grant], self.emit)


def _parent(root, relative):
    path = Path(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(p in ("..", ".") for p in path.parts)
    ):
        raise PermissionDenied("File path must be contained and relative")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor, path.name
    except OSError:
        os.close(descriptor)
        raise PermissionDenied("Unsafe file traversal") from None


def _regular(descriptor):
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PermissionDenied("Only ordinary unshared files are permitted")


async def _bounded(stream):
    data = bytearray()
    while chunk := await stream.read(65536):
        data.extend(chunk)
        if len(data) > LIMIT:
            raise RuntimeFailure("Sandbox output budget exceeded")
    return bytes(data).decode("utf-8", errors="replace")


class LocalTools:
    def __init__(self, policy, *, secrets=None):
        self.policy = policy
        self._secrets = dict(secrets or {})
        self._handles = {
            name: "credential:" + secure_random.token_hex(16) for name in self._secrets
        }

    def _file(self, args, operation):
        scope, root = args["scope"], str(Path(args["root"]).resolve())
        if scope not in FILE_SCOPES:
            raise PermissionDenied("Unknown file authority")
        self.policy.require(scope, root, operation)
        return _parent(root, args["path"])

    def read(self, args):
        parent, name = self._file(args, "read")
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(descriptor, "rb") as stream:
                _regular(stream.fileno())
                value = stream.read(LIMIT + 1)
                if len(value) > LIMIT:
                    raise RuntimeFailure("File read budget exceeded")
                return value.decode("utf-8")
        except OSError:
            raise PermissionDenied("Unsafe or unavailable file") from None
        finally:
            os.close(parent)

    def write(self, args):
        content = args["content"]
        if not isinstance(content, str) or len(content.encode()) > LIMIT:
            raise ValueError("Write requires bounded UTF-8 text")
        parent, name = self._file(args, "write")
        temporary = ".agent-lab-" + secure_random.token_hex(16)
        try:
            try:
                existing = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                try:
                    _regular(existing)
                finally:
                    os.close(existing)
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            with os.fdopen(descriptor, "w") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            return {"sha256": hashlib.sha256(content.encode()).hexdigest()}
        except OSError:
            raise PermissionDenied("Unsafe or unavailable write") from None
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    def delete(self, args):
        parent, name = self._file(args, "delete")
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                _regular(descriptor)
            finally:
                os.close(descriptor)
            os.unlink(name, dir_fd=parent)
            os.fsync(parent)
            return {"deleted": True}
        finally:
            os.close(parent)

    def secret_handle(self, args):
        name = args["name"]
        self.policy.require("secrets", name, "handle")
        if name not in self._handles:
            raise PermissionDenied("Credential unavailable")
        return self._handles[name]

    async def fetch(self, args):
        url = args["url"]
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "http") or parsed.username or parsed.password:
            raise PermissionDenied("Unsupported network target")
        self.policy.require("network", url, "get")

        def request():
            import urllib.request

            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    raise PermissionDenied("Redirect requires separate authority")

            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}), NoRedirect()
            )
            with opener.open(url, timeout=5) as response:
                body = response.read(LIMIT + 1)
            if len(body) > LIMIT:
                raise RuntimeFailure("Network read budget exceeded")
            return body.decode("utf-8")

        return await asyncio.to_thread(request)

    async def process(self, args):
        import json

        argv = args["argv"]
        if (
            not isinstance(argv, list)
            or not 1 <= len(argv) <= 32
            or any(not isinstance(a, str) for a in argv)
            or sum(len(a) for a in argv) > 65536
        ):
            raise ValueError("Process needs bounded argv")
        executable = str(Path(argv[0]).resolve())
        self.policy.require("process", executable, "execute")
        timeout = args.get("timeout", 5)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 60
        ):
            raise ValueError("Process timeout must be 0..60 seconds")
        pin = json.loads((ROOT / "configurations/sandbox-v1.json").read_text())
        binary = shutil.which("bwrap")
        if (
            binary is None
            or hashlib.sha256(Path(binary).read_bytes()).hexdigest()
            != pin["bubblewrap_sha256"]
        ):
            raise SandboxUnavailable("Pinned bubblewrap binary unavailable")
        command = [
            binary,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--ro-bind",
            "/usr",
            "/usr",
        ]
        for path in ("/lib", "/lib64"):
            if Path(path).exists():
                command += ["--ro-bind", path, path]
        command += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "--setenv",
            "TZ",
            "UTC",
            "--chdir",
            "/tmp",
        ]
        roots = sorted(
            {
                g.resource
                for g in self.policy.grants
                if g.scope in FILE_SCOPES and g.operation == "read"
            }
        )
        for index, root in enumerate(roots):
            command += ["--ro-bind", root, f"/scopes/{index}"]
        # Subprocesses always have offline read-only persistent mounts. Writes use
        # the separately scoped native file tool, never inherited process authority.
        command += ["--", executable, *argv[1:]]
        launch = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        )
        process = None
        try:
            process = await asyncio.shield(launch)
            async with asyncio.timeout(timeout):
                async with asyncio.TaskGroup() as group:
                    stdout = group.create_task(_bounded(process.stdout))
                    stderr = group.create_task(_bounded(process.stderr))
                    group.create_task(process.wait())
            result = {
                "returncode": process.returncode,
                "stdout": stdout.result(),
                "stderr": stderr.result(),
            }
            if process.returncode and ("bwrap:" in result["stderr"]):
                raise SandboxUnavailable("Required namespace isolation unavailable")
            return result
        finally:
            if process is None:
                process = await launch
            if process.returncode is None:
                process.kill()
            await process.wait()

    def registry(self):
        return {
            "read_file": self.read,
            "write_file": self.write,
            "delete_file": self.delete,
            "credential_handle": self.secret_handle,
            "http_get": self.fetch,
            "run_process": self.process,
        }

    @property
    def idempotent_tools(self):
        return {"read_file", "credential_handle"}
