"""Machine-readable experiment CLI."""

import argparse
import json
import sys
from pathlib import Path

from agent_lab.experiments import canonical, compare, summarize, validate_result


def main():
    parser = argparse.ArgumentParser(prog="agent-lab")
    parser.add_argument("command", choices=["validate", "compare", "summarize"])
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        values = [json.loads(path.read_text()) for path in args.files]
        if args.command == "validate":
            for value in values:
                validate_result(value)
            result = {"valid": True, "count": len(values)}
        elif args.command == "compare":
            if len(values) != 2:
                raise ValueError("compare requires exactly two results")
            result = compare(*values)
        else:
            result = summarize(values)
        print(canonical(result))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
