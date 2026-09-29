"""Historical repository fixtures with live policy and protected executable grading."""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from agent_lab.experiments import ROOT
from agent_lab.permissions import Grant, LocalTools, Policy
from agent_lab.runtime import RuntimeFailure

FIXTURE = ROOT / "benchmarks/repositories/unknown-usage-v1"
NEEDLE = "                    await asyncio.sleep(config.retry_delay * 2**attempt)\n"
ADDITION = (
    "                except Exception, asyncio.CancelledError:\n"
    "                    state.input_tokens = state.output_tokens = None\n"
    "                    raise\n"
)


class RepositorySetupUnavailable(RuntimeFailure):
    """Live policy cannot be resolved; never invent substitute conventions."""


def fixture_manifest():
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Historical fixture drift")
    return manifest


def reference_patch(source):
    if source.count(NEEDLE) != 1:
        raise ValueError("Historical request seam changed")
    return source.replace(NEEDLE, NEEDLE + ADDITION)


def _request_try(tree):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Try)
            and any(isinstance(statement, ast.AsyncWith) for statement in node.body)
            and any(
                isinstance(child, ast.Await)
                and isinstance(child.value, ast.Call)
                and isinstance(child.value.func, ast.Attribute)
                and child.value.func.attr == "complete"
                for child in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            )
        ):
            return node
    raise ValueError("Adapter request boundary absent")


def safe_patch(original, candidate):
    """Allow only side-effect-free null accounting plus re-raise at one seam."""
    try:
        before, after = ast.parse(original), ast.parse(candidate)
        initial, changed = _request_try(before), _request_try(after)
        if len(changed.handlers) != len(initial.handlers) + 1:
            return False
        handler = changed.handlers.pop()
        expected_type = (
            ast.parse("try:\n pass\nexcept Exception, asyncio.CancelledError:\n pass\n")
            .body[0]
            .handlers[0]
            .type
        )
        if (
            ast.dump(handler.type) != ast.dump(expected_type)
            or handler.name is not None
            or not handler.body
        ):
            return False
        fields = []
        for statement in handler.body[:-1]:
            if (
                not isinstance(statement, ast.Assign)
                or not isinstance(statement.value, ast.Constant)
                or statement.value.value is not None
            ):
                return False
            for target in statement.targets:
                if (
                    not isinstance(target, ast.Attribute)
                    or not isinstance(target.value, ast.Name)
                    or target.value.id != "state"
                ):
                    return False
                fields.append(target.attr)
        if sorted(fields) != ["input_tokens", "output_tokens"]:
            return False
        last = handler.body[-1]
        if (
            not isinstance(last, ast.Raise)
            or last.exc is not None
            or last.cause is not None
        ):
            return False
        return ast.dump(before) == ast.dump(after)
    except SyntaxError, ValueError:
        return False


def resolve_conventions(workspace):
    cli = Path("/home/moenarch/moritzbrantner/coding-tooling/src/cli.ts")
    registry = Path("/home/moenarch/.config/moenarch/environment.toml")
    bun = shutil.which("bun")
    if not bun or not cli.is_file() or not registry.is_file():
        raise RepositorySetupUnavailable(
            "Live shared convention tooling/registry unavailable"
        )
    response = subprocess.run(
        [
            bun,
            str(cli),
            "conventions",
            "resolve",
            "--root",
            str(workspace),
            "--registry",
            str(registry),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    document = json.loads(response.stdout)
    if response.returncode or document["status"] != "passed":
        raise RepositorySetupUnavailable("Shared convention resolution failed")
    stack = document["data"]
    files = []
    for entry in stack["files"]:
        content = Path(entry["absolutePath"]).read_text()
        files.append(
            {
                "path": entry["path"],
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        )
    local = {
        name: (Path(workspace) / name).read_text()
        for name in ("AGENTS.md", "CLAUDE.md")
        if (Path(workspace) / name).exists()
    }
    return {
        "sourceRevision": stack["sourceRevision"],
        "technologies": stack["technologies"],
        "files": files,
        "local_instructions": local,
    }


def _expected():
    def value(inputs, outputs, calls, tools, status, output=""):
        return {
            "input_tokens": inputs,
            "output_tokens": outputs,
            "model_calls": calls,
            "tool_calls": tools,
            "status": status,
            "output": output,
        }

    return {
        "first_unknown": value(None, None, 1, 0, "failed"),
        "cancel_unknown": value(None, None, 1, 0, "cancelled"),
        "known_response": value(2, 3, 1, 0, "completed", "done"),
        "known_tool_failure": value(2, 3, 1, 0, "failed"),
        "later_unknown": value(None, None, 2, 1, "failed"),
    }


async def check_patch(workspace, *, allow_baseline=False):
    workspace = Path(workspace).resolve()
    original = (FIXTURE / "runtime.py").read_text()
    manifest = fixture_manifest()
    for name, expected in manifest["files"].items():
        if (
            name != "runtime.py"
            and hashlib.sha256((workspace / name).read_bytes()).hexdigest() != expected
        ):
            return {
                "status": "fail",
                "score": 0,
                "evaluator": manifest["checker"],
                "reason": "repository authority changed",
            }
    candidate = (workspace / "runtime.py").read_text()
    if not (allow_baseline and candidate == original) and not safe_patch(
        original, candidate
    ):
        return {
            "status": "fail",
            "score": 0,
            "evaluator": manifest["checker"],
            "reason": "patch outside reviewed historical seam",
        }
    policy = Policy(
        [
            Grant("process", "/usr/bin/python3.14", "execute"),
            Grant("repositories", str(workspace), "read"),
        ]
    )
    source = (ROOT / "agent_lab/repository_checker.py").read_text()
    # Trusted worker observes behavior; expected values remain in the parent.
    result = await LocalTools(policy).process(
        {
            "argv": [
                "/usr/bin/python3.14",
                "-I",
                "-S",
                "-c",
                source,
                "/scopes/0/runtime.py",
            ],
            "timeout": 10,
        }
    )
    if result["returncode"]:
        return {
            "status": "error",
            "score": None,
            "evaluator": manifest["checker"],
            "reason": "isolated observation failed",
        }
    observed = json.loads(result["stdout"])
    expected = _expected()
    passed = observed == expected
    return {
        "status": "pass" if passed else "fail",
        "score": int(passed),
        "evaluator": manifest["checker"],
        "observations": observed,
        "failed_criteria": [
            name for name in expected if observed.get(name) != expected[name]
        ],
    }


class RepositoryTools:
    def __init__(self, workspace, emit):
        self.workspace = Path(workspace).resolve()
        self.policy = Policy(
            [
                Grant("repositories", str(self.workspace), operation)
                for operation in ("read", "write")
            ],
            emit,
        )
        self.native = LocalTools(self.policy)

    def read(self, args):
        name = args["path"]
        if name not in ("runtime.py", "AGENTS.md", "pyproject.toml"):
            raise ValueError("Repository fixture read outside declared files")
        return self.native.read(
            {"scope": "repositories", "root": str(self.workspace), "path": name}
        )

    def replace(self, args):
        old, new = args["old"], args["new"]
        if (
            not isinstance(old, str)
            or not isinstance(new, str)
            or not old
            or max(len(old), len(new)) > 8192
        ):
            raise ValueError("Edit must be a bounded exact replacement")
        text = self.read({"path": "runtime.py"})
        if text.count(old) != 1:
            raise ValueError("Replacement must match exactly once")
        return self.native.write(
            {
                "scope": "repositories",
                "root": str(self.workspace),
                "path": "runtime.py",
                "content": text.replace(old, new),
            }
        )

    async def check(self, args):
        result = await check_patch(self.workspace)
        return {k: v for k, v in result.items() if k != "observations"}

    def registry(self):
        return {
            "read_source": self.read,
            "replace_text": self.replace,
            "check_patch": self.check,
        }
