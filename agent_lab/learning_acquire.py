"""Explicit pinned model acquisition; learning/evaluation never downloads files."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from agent_lab.experiments import ROOT
from agent_lab.trace import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    config = json.loads((ROOT / "configurations/learning-v1.json").read_text())
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=config["model"],
        revision=config["revision"],
        local_dir=output,
        token=False,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"],
    )
    weights = hashlib.sha256((output / "model.safetensors").read_bytes()).hexdigest()
    if weights != config["weights_sha256"]:
        raise ValueError("Acquired model weights differ from pinned LFS digest")
    atomic_json(
        output / "acquisition.json",
        {
            "model": config["model"],
            "revision": config["revision"],
            "weights_sha256": weights,
        },
    )


if __name__ == "__main__":
    main()
