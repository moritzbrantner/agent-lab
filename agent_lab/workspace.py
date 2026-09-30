"""Keyboard-first local assistant surface backed by runtime checkpoints."""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

from agent_lab.experiments import canonical
from agent_lab.permissions import SCOPES, Grant
from agent_lab.workspace_locale import translator
from agent_lab.workspace_store import Workspace

COMMANDS = (
    "start",
    "sessions",
    "attach",
    "run",
    "resume",
    "cancel",
    "inspect",
    "export",
    "history",
    "permissions",
    "grant",
    "memory",
    "settings",
    "abandon",
    "forget",
)


def default_root():
    return (
        Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
        / "agent-lab"
    )


def parser_for(translate):
    class Formatter(argparse.HelpFormatter):
        def _format_usage(self, usage, actions, groups, prefix):
            return super()._format_usage(
                usage, actions, groups, prefix or translate("usage") + " "
            )

    class Parser(argparse.ArgumentParser):
        def __init__(self, *args, **kwargs):
            kwargs.update(add_help=False, formatter_class=Formatter)
            super().__init__(*args, **kwargs)
            self._positionals.title = translate("arguments")
            self._optionals.title = translate("options")
            self.add_argument("-h", "--help", action="help", help=translate("help"))

        def error(self, message):
            self.print_usage(sys.stderr)
            self.exit(2, translate("invalid") + "\n")

    parser = Parser(description=translate("title"))
    parser.add_argument(
        "--root", type=Path, default=default_root(), help=translate("root")
    )
    parser.add_argument(
        "--language", choices=("en", "de", "es"), help=translate("language")
    )
    parser.add_argument("--json", action="store_true", help=translate("json"))
    commands = parser.add_subparsers(
        dest="command", required=True, title=translate("commands")
    )
    for command in COMMANDS:
        child = commands.add_parser(
            command, help=translate(command), description=translate(command)
        )
        if command not in ("start", "sessions", "settings"):
            child.add_argument("session", help=translate("session"))
        if command == "start":
            child.add_argument("name", help=translate("name"))
            child.add_argument("project", type=Path, help=translate("project"))
            child.add_argument(
                "--profile", default="medium-q4-gpu", help=translate("profile")
            )
        elif command == "attach":
            child.add_argument("file", type=Path, help=translate("file"))
        elif command == "run":
            child.add_argument("prompt", help=translate("prompt"))
        elif command == "grant":
            child.add_argument("scope", choices=sorted(SCOPES), help=translate("scope"))
            child.add_argument("resource", help=translate("resource"))
            child.add_argument("operation", help=translate("operation"))
            child.add_argument("--reason", required=True, help=translate("reason"))
        elif command == "memory":
            child.add_argument(
                "state", choices=("on", "off", "clear"), help=translate("state")
            )
        elif command == "settings":
            child.add_argument(
                "--set-language", choices=("en", "de", "es"), help=translate("language")
            )
            child.add_argument(
                "--theme", choices=("light", "dark", "system"), help=translate("theme")
            )
    return parser


def _safe(value):
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "?", str(value))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--root", type=Path, default=default_root())
    early.add_argument("--language")
    hints, _ = early.parse_known_args(argv)
    workspace = Workspace(hints.root)
    language = hints.language or workspace.preferences()["language"]
    translate = translator(language)
    args = parser_for(translate).parse_args(argv)
    preferences = workspace.preferences()

    def show(message):
        value = _safe(message)
        if sys.stdout.isatty():
            token = {
                "light": "\033[30;47m",
                "dark": "\033[97;40m",
                "system": "\033[36m",
            }.get(preferences["theme"], "\033[36m")
            print(token + value + "\033[0m")
        else:
            print(value)

    def event(kind, payload):
        if kind == "tool_request" and not args.json:
            show(translate("tool", name=translate(payload["name"])))

    try:
        command = args.command
        if command == "start":
            value = workspace.start(args.name, args.project, profile=args.profile)
        elif command == "sessions":
            value = workspace.sessions()
        elif command == "attach":
            value = workspace.attach(args.session, args.file)
        elif command == "run":
            value = asyncio.run(
                workspace.execute(args.session, args.prompt, on_event=event)
            )
        elif command == "resume":
            value = asyncio.run(workspace.resume(args.session, on_event=event))
        elif command == "cancel":
            value = workspace.cancel(args.session)
        elif command == "inspect":
            value = workspace.inspect(args.session)
        elif command == "export":
            value = workspace.export_config(args.session)
        elif command == "history":
            value = workspace.history(args.session)
        elif command == "permissions":
            value = workspace.session(args.session)["permissions"]
        elif command == "grant":
            value = workspace.grant(
                args.session,
                Grant(args.scope, args.resource, args.operation),
                reason=args.reason,
            )
        elif command == "memory":
            value = workspace.memory(
                args.session,
                enabled=None if args.state == "clear" else args.state == "on",
                clear=args.state == "clear",
            )
        elif command == "settings":
            value = workspace.preferences(language=args.set_language, theme=args.theme)
        elif command == "abandon":
            value = workspace.abandon(args.session)
        else:
            value = workspace.forget(args.session)
        if args.json or command == "export":
            print(canonical(value))
        elif command == "start":
            show(translate("started", name=value["name"], id=value["id"]))
        elif command == "attach":
            show(translate("attached", name=value["name"], id=value["id"]))
        elif command == "sessions":
            for session in value:
                show(session["id"] + "  " + session["name"] + "  " + session["profile"])
            if not value:
                show(translate("empty"))
        elif command in ("run", "resume", "inspect"):
            state = value["state"] if command == "inspect" else value
            status = translate(
                "status_failed" if state["status"] == "failed" else state["status"]
            )
            show(
                translate(
                    "summary",
                    status=status,
                    models=state["model_calls"],
                    tools=state["tool_calls"],
                )
            )
            if state["output"]:
                show(state["output"])
            if command == "inspect" and value["trace"]:
                for entry in value["trace"]["events"]:
                    if entry["kind"] == "tool_request" and "name" in entry["metadata"]:
                        show(
                            translate("tool", name=translate(entry["metadata"]["name"]))
                        )
        elif command in ("permissions", "history"):
            for entry in value:
                show(canonical(entry))
        else:
            show(translate("done"))
        return (
            1 if command in ("run", "resume") and value["status"] != "completed" else 0
        )
    except KeyboardInterrupt:
        show(translate("interrupted"))
        return 130
    except Exception as error:
        show(translate("error", category=type(error).__name__))
        return 1


if __name__ == "__main__":
    sys.exit(main())
