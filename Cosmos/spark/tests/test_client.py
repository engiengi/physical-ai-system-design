"""Exercise real OpenPI WebSocket exchanges and failure handling without a GPU."""
from contextlib import contextmanager
import json
import threading
import time

import numpy as np
import pytest
import torch
from websockets.sync.server import serve

from robolab.eval.websocket_transport import MsgPackNumpy
from cosmos_spark.client import PolicyError, RecordedCosmosClient
from cosmos_spark.artifacts import summarize


@contextmanager
def policy_server(respond):
    codec = MsgPackNumpy()
    requests = []
    def handler(socket):
        socket.send(codec.pack({"model": "test-only", "action_dim": 8}))
        try:
            for message in socket:
                request = codec.unpack(message)
                requests.append(request)
                response = respond(request)
                socket.send(codec.pack(response))
        except Exception:
            pass  # A timeout test deliberately closes before the response.
    server = serve(handler, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}", requests
    finally:
        server.shutdown()
        thread.join(timeout=3)


def observation():
    return {"image_obs": {name: torch.full((1, 360, 640, 3), value, dtype=torch.uint8)
                           for name, value in [("wrist_cam", 50), ("over_shoulder_left_camera", 100),
                                               ("over_shoulder_right_camera", 150)]},
            "proprio_obs": {"arm_joint_pos": torch.zeros(1, 7), "gripper_pos": torch.zeros(1, 1)}}


def test_short_chunks_replan_and_preserve_wire_format(tmp_path):
    actions = np.zeros((2, 8), dtype=np.float32)
    actions[1, 0] = 0.1
    actions[1, -1] = 0.8
    with policy_server(lambda request: {"actions": actions}) as (uri, requests):
        client = RecordedCosmosClient(uri, tmp_path, "run-a", "episode-0", execute_horizon=16)
        try:
            outputs = [client.infer(observation(), "Put banana in bowl")["action"] for _ in range(3)]
            assert len(requests) == 2  # Shorter than requested horizon, no indexing past end.
            assert requests[0]["session_id"] == "run-a/episode-0/env-0"
            image = requests[0]["observation/image"]
            assert image.shape == (540, 640, 3)
            assert image[0, 0, 0] == 50 and image[-1, 0, 0] == 100 and image[-1, -1, 0] == 150
            assert outputs[1][-1] == 1  # Official gripper threshold.
            assert outputs[2][0] == 0
            client.reset()
            client.infer(observation(), "Put banana in bowl")
            assert len(requests) == 3
        finally:
            client.close()
    assert len(list(tmp_path.glob("*_observation.npz"))) == 3
    trace = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert all(entry["status"] == "ok" and entry["rtt_ms"] >= 0 for entry in trace)


@pytest.mark.parametrize("response", [
    {"actions": np.zeros((2, 7))}, {"actions": np.zeros((0, 8))},
    {"actions": np.full((1, 8), np.nan)}, {"actions": np.full((1, 8), np.inf)},
    {"message": "no action"},
    {"type": "error", "message": "out of memory"},
])
def test_invalid_responses_abort_without_action(tmp_path, response):
    with policy_server(lambda request: response) as (uri, _):
        client = RecordedCosmosClient(uri, tmp_path, "run", "episode")
        try:
            with pytest.raises(PolicyError):
                client.infer(observation(), "test")
            assert not client._chunks
        finally:
            client.close()
    assert json.loads((tmp_path / "requests.jsonl").read_text())["status"] == "execution_error"


def test_raw_gripper_predictions_use_official_threshold_and_keep_evidence(tmp_path):
    actions = np.zeros((6, 8), dtype=np.float32)
    actions[:, 0] = np.arange(6) * 0.1
    actions[:, -1] = [-0.1, 0.0, 0.5, 0.5001, 1.0, 1.1]
    response = {"actions": actions, "run_id": "thor-run", "request_id": "thor-request",
                "session_id": "shared-session", "inference_time_ms": 123.0,
                "api_token": "must-not-be-recorded"}
    with policy_server(lambda request: response) as (uri, requests):
        client = RecordedCosmosClient(uri, tmp_path, "spark-run", "episode")
        try:
            outputs = np.asarray([client.infer(observation(), "test")["action"] for _ in range(6)])
        finally:
            client.close()
        assert len(requests) == 1
    np.testing.assert_array_equal(outputs[:, -1], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(outputs[:, :7], actions[:, :7])
    np.testing.assert_array_equal(np.load(tmp_path / "request_00000_actions.npy"), actions)
    raw_log = (tmp_path / "requests.jsonl").read_text()
    entry = json.loads(raw_log)
    assert entry["status"] == "ok"
    assert entry["raw_gripper_outside_unit_interval"] == 2
    assert entry["raw_gripper_min"] == pytest.approx(-0.1)
    assert entry["raw_gripper_max"] == pytest.approx(1.1)
    assert entry["server_ids"] == {k: response[k] for k in ("run_id", "request_id", "session_id")}
    assert entry["server_timing"] == {"inference_time_ms": 123.0}
    assert entry["run_id"] == "spark-run"
    assert "must-not-be-recorded" not in raw_log


def test_timeout_is_bounded_and_recorded(tmp_path):
    def slow(request):
        time.sleep(0.3)
        return {"actions": np.zeros((2, 8))}
    with policy_server(slow) as (uri, _):
        client = RecordedCosmosClient(uri, tmp_path, "run", "episode", timeout=0.05)
        try:
            start = time.perf_counter()
            with pytest.raises(PolicyError):
                client.infer(observation(), "test")
            assert time.perf_counter() - start < 2
        finally:
            client.close()
    assert json.loads((tmp_path / "requests.jsonl").read_text())["status"] == "execution_error"


def test_errors_and_diagnostics_are_excluded_from_success_rate():
    result = summarize([{"status": status} for status in
                        ["success", "task_failure", "execution_error", "diagnostic_complete", "incomplete"]])
    assert result["evaluated"] == 2 and result["success_rate"] == 0.5
    assert result["execution_errors"] == 1 and result["incomplete"] == 1
