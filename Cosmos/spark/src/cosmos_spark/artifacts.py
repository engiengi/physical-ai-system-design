"""Experiment artifacts with stable IDs and explicit result categories."""
import hashlib
import json
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, default=json_default, allow_nan=False) + "\n")


def manifest(directory):
    directory = Path(directory)
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            entries.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                            "sha256": digest.hexdigest()})
    write_json(directory / "manifest.json", entries)


def summarize(results):
    evaluated = [r for r in results if r["status"] in ("success", "task_failure")]
    successes = sum(r["status"] == "success" for r in evaluated)
    return {"attempted": len(results), "evaluated": len(evaluated), "successes": successes,
            "task_failures": len(evaluated) - successes,
            "execution_errors": sum(r["status"] == "execution_error" for r in results),
            "diagnostics": sum(r["status"] == "diagnostic_complete" for r in results),
            "incomplete": sum(r["status"] == "incomplete" for r in results),
            "success_rate": successes / len(evaluated) if evaluated else None}
