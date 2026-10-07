"""Official Cosmos preprocessing with bounded transport and trace recording."""
import os
import time
from pathlib import Path

import numpy as np

from policies.cosmos3.client import Cosmos3Client
from robolab.eval.base_client import InferenceClient
from robolab.eval.websocket_transport import MsgPackWebSocketTransport

from .artifacts import append_jsonl, write_json


class PolicyError(RuntimeError):
    """Transport/protocol failure, never counted as task failure."""


def validate_actions(response):
    if isinstance(response, dict):
        if response.get("type") == "error":
            raise PolicyError(str(response.get("message", "Policy server error")))
        response = response.get("action", response.get("actions"))
    try:
        values = np.asarray(response, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise PolicyError("Action response is not numeric") from exc
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 8:
        raise PolicyError(f"Expected nonempty [T,8] actions, received {values.shape}")
    if not np.isfinite(values).all():
        raise PolicyError("Action response contains NaN or infinity")
    # These are raw policy predictions, not simulator gripper commands.
    # The official Cosmos3Client._postprocess_chunk thresholds finite gripper
    # predictions at > 0.5 into 0=open / 1=closed, including values outside [0,1].
    return values


class RecordedCosmosClient(Cosmos3Client):
    def __init__(self, uri, episode_dir, run_id, episode_id, *, execute_horizon=16,
                 timeout=30.0, connect_timeout=5.0, policy_seed=None, prediction_receiver=None, scenario_context=None):
        # Avoid the upstream constructor's unbounded connection retry loop.
        InferenceClient.__init__(self)
        if execute_horizon < 1 or timeout <= 0 or connect_timeout <= 0:
            raise ValueError("Horizons and timeouts must be positive")
        self._image_w, self._image_h = self.IMAGE_W, self.IMAGE_H
        self.open_loop_horizon = execute_horizon
        self.timeout = timeout
        if policy_seed is not None and (type(policy_seed) is not int or not 0 <= policy_seed < 2**31):
            raise ValueError("policy_seed must be an integer in [0, 2**31)")
        self.policy_seed = policy_seed
        self.prediction_receiver = prediction_receiver
        self.scenario_context = dict(scenario_context or {})
        if self.scenario_context:
            import re
            if (set(self.scenario_context)!={'scenario_id','experiment_track','source_type'} or
                not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',str(self.scenario_context.get('scenario_id',''))) or
                self.scenario_context.get('experiment_track') not in ('new_standalone','banana_link') or
                self.scenario_context.get('source_type')!='simulation'):
                raise ValueError('Invalid simulator scenario context')
        self.episode_dir = Path(episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.run_id, self.episode_id = run_id, episode_id
        self.request_index = 0
        self.step_index = 0
        self.rtt_ms = []
        self.last_request_id = None
        self.last_chunk_index = None
        self.client = MsgPackWebSocketTransport(
            uri, api_token=os.environ.get("COSMOS3_API_TOKEN"),
            connect_kwargs={"open_timeout": connect_timeout, "close_timeout": 2,
                            "max_size": 64 * 1024 * 1024},
            metadata_timeout=connect_timeout,
        )
        try:
            metadata = self.client.connect()
        except Exception as exc:
            self.client.close()
            raise PolicyError(f"Policy connection failed: {exc}") from exc
        # Metadata may include model settings; never persist authentication fields.
        safe = {k: metadata[k] for k in ("model", "model_name", "model_revision", "action_dim", "action_horizon",
                                       "run_id", "action_shape", "conditioning_fps", "decode_video", "capabilities", "protocol_version", "policy_seed")
                if isinstance(metadata, dict) and k in metadata}
        write_json(self.episode_dir / "server.json", safe)

    def _extract_observation(self, raw_obs, *, env_id=0):
        extracted = super()._extract_observation(raw_obs, env_id=env_id)
        extracted["session_id"] = f"{self.run_id}/{self.episode_id}/env-{env_id}"
        return extracted

    def _query_server(self, request):
        request_id = f"request_{self.request_index:05d}"
        request.update(self.scenario_context)
        request["episode_id"] = self.episode_id
        request["episode_request_index"] = self.request_index
        if self.policy_seed is not None:
            request["policy_seed"] = (self.policy_seed + self.request_index) % (2**31)
        self.last_request_id = request_id
        self.request_index += 1
        np.savez_compressed(self.episode_dir / f"{request_id}_observation.npz",
                            image=request["observation/image"],
                            joint_position=request["observation/joint_position"],
                            gripper_position=request["observation/gripper_position"])
        entry = {"run_id": self.run_id, "episode_id": self.episode_id,
                 "request_id": request_id, "session_id": request["session_id"],
                 "step": self.step_index, "prompt": request["prompt"],
                 "wall_time_ns": time.time_ns()}
        if self.policy_seed is not None:
            entry["policy_seed_requested"] = request["policy_seed"]
        start = time.perf_counter()
        try:
            response = self.client.request(request, timeout=self.timeout)
            entry["rtt_ms"] = (time.perf_counter() - start) * 1000
            self.rtt_ms.append(entry["rtt_ms"])
            # Preserve server identifiers and timings even when actions fail
            # validation. Never serialize the entire response (may contain secrets).
            if isinstance(response, dict):
                entry["server_ids"] = {k: response[k] for k in ("run_id", "request_id", "session_id")
                                       if k in response}
                entry["server_timing"] = {k: response[k] for k in ("server_timing", "inference_time_ms", "inference_time_s")
                                          if k in response}
                entry["memory"] = response.get("memory")
                entry["client_context"] = response.get("client_context")
                entry["seed_rule"] = response.get("seed_rule")
            actions = validate_actions(response)
            # Save raw predictions; applied_actions.jsonl records the commands
            # after the inherited official gripper threshold is applied.
            np.save(self.episode_dir / f"{request_id}_actions.npy", actions)
            if self.policy_seed is not None:
                actual = response.get("policy_seed", response.get("seed")) if isinstance(response, dict) else None
                entry["policy_seed_used"] = actual
                if type(actual) is not int or actual != request["policy_seed"]:
                    raise PolicyError("Thor did not acknowledge the requested policy_seed; action withheld")
            if self.prediction_receiver is not None:
                # Download, verify and persist prediction BEFORE returning any action to the runner.
                entry["prediction"] = self.prediction_receiver(response, request_id, actions, request)
            entry.update(status="ok", returned_steps=len(actions),
                         executable_steps=min(len(actions), self.open_loop_horizon),
                         raw_gripper_min=float(actions[:, -1].min()),
                         raw_gripper_max=float(actions[:, -1].max()),
                         raw_gripper_outside_unit_interval=int(np.count_nonzero(
                             (actions[:, -1] < 0) | (actions[:, -1] > 1))),
                         gripper_postprocess="official_threshold_gt_0.5")
            return actions
        except Exception as exc:
            entry.update(status="execution_error", error_type=type(exc).__name__,
                         error=str(exc), elapsed_ms=(time.perf_counter() - start) * 1000)
            raise PolicyError(f"{request_id}: {exc}") from exc
        finally:
            append_jsonl(self.episode_dir / "requests.jsonl", entry)

    def _needs_refresh(self, env_id):
        return (env_id not in self._chunks or
                self._counters[env_id] >= min(self.open_loop_horizon, len(self._chunks[env_id])))

    def infer(self, obs, instruction, *, env_id=0):
        result = super().infer(obs, instruction, env_id=env_id)
        self.last_chunk_index = self._counters[env_id] - 1
        self.step_index += 1
        return result
