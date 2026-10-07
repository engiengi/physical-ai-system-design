"""Optional GPU integration: real simulation -> local test policy -> fault abort.

This is a transport diagnostic. It does not run Cosmos and must not be used as
a model-performance result. Run with scripts/spark-python tests/integration_policy_probe.py.
"""
import json
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

from test_client import policy_server


root = Path(__file__).resolve().parents[1]
run_id = "transport_diagnostic_" + uuid.uuid4().hex[:8]
count = 0


def respond(request):
    global count
    count += 1
    assert request["observation/image"].shape == (540, 640, 3)
    assert request["session_id"].startswith(run_id + "/episode_0000/")
    if count == 3:
        return {"type": "error", "message": "intentional transport diagnostic failure"}
    action = np.concatenate((request["observation/joint_position"], [0.0]))
    return {"actions": np.tile(action, (2, 1)), "inference_time_ms": 0.0}


with policy_server(respond) as (uri, requests):
    log_path = root / "logs" / f"{run_id}.log"
    with log_path.open("w") as log:
        completed = subprocess.run([sys.executable, "-u", "-m", "cosmos_spark.runner", "--mode", "policy",
                                    "--uri", uri, "--headless", "--max-steps", "10", "--run-id", run_id],
                                   cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=240)
result = json.loads((root / "runs" / run_id / "episode_0000" / "episode.json").read_text())
assert completed.returncode != 0, "Infrastructure failure must propagate as a failing exit code"
assert result["status"] == "execution_error"
assert result["steps"] == 4, result
assert len(requests) == 3, len(requests)
assert result["rtt_ms"]["count"] == 3
print(json.dumps({"test": "simulation_websocket_fault_abort", "passed": True,
                  "run_id": run_id, "applied_steps": result["steps"],
                  "requests": len(requests), "exit_code": completed.returncode}, indent=2))
