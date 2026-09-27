"""Opt-in integration check against an ALREADY RUNNING simulator + web server.

Runs one manual pick/place through the same HTTP API as the lab UI. No model API
calls. Run on Spark: .venv/bin/python tests/gui_mode_smoke.py native|remote
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ROOT, read_json, write_json


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request("http://127.0.0.1:8765" + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def action(payload):
    job = request("/api/sim", payload)["job"]
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        result = request("/api/jobs/" + job)
        if result["status"] == "done":
            return result["result"]
        if result["status"] == "error":
            raise RuntimeError(result)
        time.sleep(1)
    raise TimeoutError(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["native", "remote"])
    args = parser.parse_args()
    action({"command": "reset", "scene": 0})
    obs = action({"command": "snapshot"})
    # Fixed image coordinates used in the previously validated scene-0 manual test.
    # No ground-truth object poses are passed to the controller.
    result = action({"command": "pick_place", "action": {
        "pick": [575, 664], "place": [472, 305], "observation_id": obs["observation_id"]}})
    metrics = read_json(ROOT / result["output"] / "metrics.json")
    report = {"mode": args.mode, "observation": obs, "result": result, "metrics": metrics}
    write_json(ROOT / "outputs/validation/gui_modes" / f"{args.mode}_smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not metrics.get("placement_within_tolerance"):
        raise RuntimeError("Placement check failed; review saved video")


if __name__ == "__main__":
    main()
