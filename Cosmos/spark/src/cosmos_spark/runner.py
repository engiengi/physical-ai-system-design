"""Single-environment, reproducible RoboLab runner for Spark and Thor."""
from .settings import URI
import argparse
from datetime import datetime
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import re
import os
import subprocess
import sys
import time
import traceback
import uuid

import cv2  # Must precede Isaac Lab imports.
import numpy as np

from .artifacts import append_jsonl, manifest, summarize, write_json


ROOT = Path(__file__).resolve().parents[2]


def provenance():
    packages = {}
    for name in ("isaacsim", "isaaclab", "torch", "websockets", "robolab"):
        packages[name] = importlib.metadata.version(name)
    commit = subprocess.check_output(["git", "-C", str(ROOT / "RoboLab"), "rev-parse", "HEAD"], text=True).strip()
    paths = [*sorted((ROOT / "src").rglob("*.py")), ROOT / "pyproject.toml", ROOT / "uv.lock"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip()
    return {"packages": packages, "robolab_commit": commit, "python": platform.python_version(),
            "machine": platform.machine(), "platform": platform.platform(), "gpu_driver": driver,
            "source_sha256": hashes}


def snapshot(env, path):
    """Save full simulator state without pickle for future controlled replay."""
    arrays = {}
    def visit(value, prefix):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{prefix}/{key}" if prefix else key)
        else:
            arrays[prefix] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    visit(env.scene.get_state(is_relative=True), "")
    np.savez_compressed(path, **arrays)


def load_snapshot(path, device):
    import torch
    state = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            cursor = state
            parts = key.split("/")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = torch.as_tensor(data[key], device=device)
    return state


def execute_episode(args, env, cfg, output, run_id, episode_id):
    import torch
    from robolab.core.environments.runtime import end_episode
    from robolab.core.logging.results import get_all_env_subtask_infos
    from robolab.core.utils.video_utils import VideoWriter
    from .client import RecordedCosmosClient

    result = {"run_id": run_id, "episode_id": episode_id, "mode": args.mode,
              "seed": cfg.seed, "status": "incomplete", "steps": 0}
    scenario_context = ({'scenario_id':args.scenario_id,'experiment_track':args.experiment_track,'source_type':'simulation'}
                        if args.scenario_id else {})
    result.update(scenario_context)
    client = None
    prediction = None
    writers = {}
    started = time.perf_counter()
    cleanup_errors = []
    try:
        env.reset_eval_state()
        obs, _ = env.reset()
        obs, _ = env.reset()  # Match the official RoboLab evaluation initialization.
        replay_actions = None
        if args.restore_episode:
            source = Path(args.restore_episode)
            obs, _ = env.reset_to(load_snapshot(source / "initial_state.npz", env.device),
                                  env_ids=None, seed=cfg.seed, is_relative=True)
            result["restored_from"] = str(source.resolve())
        if args.mode == "replay":
            source = Path(args.restore_episode)
            replay_actions = [json.loads(line)["action"] for line in (source / "applied_actions.jsonl").read_text().splitlines()]
            if not replay_actions:
                raise ValueError("Replay source contains no applied actions")
            result["replay_source"] = str(source.resolve())
        env.recorder_manager.set_hdf5_file("trajectory.hdf5")
        env.recorder_manager.set_episode_index(0, env_ids=[0])
        snapshot(env, output / "initial_state.npz")
        if args.pose_range_m and args.mode != "replay":
            offsets = {}
            for name in ("banana", "bowl"):
                asset = env.scene[name]
                delta = asset.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2] - asset.data.default_root_state[0, :2]
                offsets[name] = delta.cpu().numpy()
            result["initial_xy_offsets_m"] = offsets
            if all(np.allclose(delta, 0, atol=1e-6) for delta in offsets.values()):
                raise RuntimeError("Requested pose randomization did not change object positions")
        control_dt = cfg.sim.dt * cfg.decimation
        result.update(goal_variant=args.goal_variant,
                      task_provenance='official RoboLab task' if args.goal_variant=='official' else 'custom diagnostic derivative: single red hammer, declared 60s budget; scene change recorded in goal_variant and env_cfg')
        result.update(control_dt_s=control_dt, control_hz=1 / control_dt,
                      instruction=cfg.instruction, execute_horizon=args.execute_horizon,
                      configured_episode_limit_s=float(cfg.episode_length_s),
                      configured_episode_limit_steps=int(env.max_episode_length),
                      manual_step_cap=args.max_steps,
                      evaluation_scope='terminal_outcome' if args.mode=='policy' and not args.max_steps else 'diagnostic')
        write_json(output / "episode.json", result)
        if args.mode == "predict":
            from .prediction import PredictionReceiver
            prediction = PredictionReceiver(output, args.execute_horizon, 1 / control_dt,
                                            source=args.prediction_source)
        if args.mode in ("policy", "predict"):
            client = RecordedCosmosClient(args.uri, output, run_id, episode_id,
                                          execute_horizon=args.execute_horizon,
                                          timeout=args.timeout, policy_seed=args.policy_seed,
                                          prediction_receiver=prediction, scenario_context=scenario_context)
            client.begin_episode(0)
        # Video t=0 is the exact observation before the first applied action.
        def record_observation(observation):
            for group in ("image_obs", "viewport_cam"):
                for name, image in observation.get(group, {}).items():
                    frame = image[0].detach().cpu().numpy()
                    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                        continue
                    if not np.isfinite(frame).all() or frame.std() == 0:
                        raise RuntimeError(f"Invalid or blank camera observation: {name}")
                    if name not in writers:
                        writers[name] = VideoWriter(str(output / f"{name}.mp4"), 1 / control_dt)
                        cv2.imwrite(str(output / f"{name}_initial.png"), cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR))
                    writers[name].write(frame[..., :3])
            if prediction is not None:
                packed = client._pack_request(client._extract_observation(observation), cfg.instruction)
                if "policy_view" not in writers:
                    writers["policy_view"] = VideoWriter(str(output / "policy_view.mp4"), 1 / control_dt)
                writers["policy_view"].write(packed["observation/image"])
        record_observation(obs)
        required = {"wrist_cam", "over_shoulder_left_camera", "over_shoulder_right_camera"}
        if not required.issubset(writers):
            raise RuntimeError(f"Missing policy cameras: {required - writers.keys()}")
        held_joints = obs["proprio_obs"]["arm_joint_pos"][0].cpu().numpy().copy()
        steps = args.max_steps or env.max_episode_length
        if prediction is not None:
            steps = args.execute_horizon
        if replay_actions is not None:
            steps = min(steps, len(replay_actions))
        for step in range(steps):
            before = obs["proprio_obs"]
            if client is not None:
                action = client.infer(obs, cfg.instruction)["action"]
            elif replay_actions is not None:
                action = np.asarray(replay_actions[step], dtype=np.float32)
            else:
                # Diagnostic only: hold arm while opening/closing the gripper.
                action = np.concatenate((held_joints, [float((step // 15) % 2)]))
            entry = {"run_id": run_id, "episode_id": episode_id, "step": step,
                     "sim_time_before_s": step * control_dt,
                     "observation_frame": step, "result_frame": step + 1,
                     "action": action, "joint_position_before": before["arm_joint_pos"][0].cpu().numpy(),
                     "gripper_position_before": before["gripper_pos"][0].cpu().numpy(),
                     "request_id": client.last_request_id if client else None,
                     "chunk_index": client.last_chunk_index if client else None}
            entry["action_started_at_ns"] = time.time_ns()
            obs, _, term, trunc, _ = env.step(torch.as_tensor(action, device=env.device).reshape(1, 8))
            # Write applied actions only after a successful simulator step.
            entry.update(wall_time_ns=time.time_ns(), terminated=bool(term[0]), truncated=bool(trunc[0]))
            append_jsonl(output / "applied_actions.jsonl", entry)
            result["steps"] += 1
            append_jsonl(output / "subtasks.jsonl", {"step": step, "states": get_all_env_subtask_infos(env)})
            record_observation(obs)
            if env.all_terminated:
                break
        snapshot(env, output / "final_state.npz")
        result["robolab_result"] = env.get_env_results()[0]
        result['termination_evidence']={'environment_finished':bool(env.all_terminated),
            'terminated':bool(term[0]) if result['steps'] else False,
            'truncated':bool(trunc[0]) if result['steps'] else False,
            'sim_time_s':result['steps']*control_dt}
        if args.mode in ("smoke", "replay", "predict"):
            result["status"] = "diagnostic_complete"
        elif env.all_terminated:
            result["status"] = "success" if result["robolab_result"]["success"] else "task_failure"
        else:
            result["status"] = "incomplete"
        result['termination_reason']=('success_condition' if result['status']=='success' else
            'task_time_limit' if result['status']=='task_failure' and bool(trunc[0]) else
            'task_termination' if result['status']=='task_failure' else
            'manual_step_cap' if result['status']=='incomplete' and args.max_steps else result['status'])
        if replay_actions is not None and result["steps"] == len(replay_actions):
            with np.load(Path(args.replay_episode) / "final_state.npz") as expected, np.load(output / "final_state.npz") as actual:
                result["replay_final_max_abs_error"] = {key: float(np.max(np.abs(actual[key] - expected[key])))
                                                         for key in expected.files if expected[key].size}
        if args.post_terminal_seconds and env.all_terminated and args.mode != 'predict':
            result['task_outcome_before_post_observation']=result['status']
            from .settling import observe
            result['post_terminal_observation']=observe(env,cfg,obs,action,output/'post_terminal',args.post_terminal_seconds)
    except Exception as exc:
        result.update(status="execution_error", error_type=type(exc).__name__, error=str(exc))
        (output / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        for writer in writers.values():
            writer.release()
        if prediction is not None and result["status"] != "execution_error":
            try:
                result["prospective_comparison"] = prediction.finalize()
            except Exception as exc:
                result.update(status="execution_error", error_type=type(exc).__name__, error=str(exc))
                (output / "error.txt").write_text(traceback.format_exc())
        if client is not None:
            if client.rtt_ms:
                result["rtt_ms"] = {"count": len(client.rtt_ms), "mean": np.mean(client.rtt_ms),
                                    "p50": np.percentile(client.rtt_ms, 50),
                                    "p95": np.percentile(client.rtt_ms, 95), "max": max(client.rtt_ms)}
            client.close()
        try:
            end_episode(env)
        except Exception as exc:
            cleanup_errors.append(str(exc))
        result.update(wall_elapsed_s=time.perf_counter() - started,
                      simulated_elapsed_s=result["steps"] * cfg.sim.dt * cfg.decimation,
                      camera_names=sorted(writers))
        if cleanup_errors:
            result.update(status="execution_error", cleanup_errors=cleanup_errors)
        write_json(output / "episode.json", result)
    return result


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "policy", "replay", "predict"], default="smoke")
    parser.add_argument("--prediction-source", choices=["response", "legacy-files"], default="response",
                        help="response: Thor artifact metadata; legacy-files: explicit SSH adapter for existing decoded-video server")
    parser.add_argument("--replay-episode", help="Recorded episode directory (same code and simulator versions)")
    parser.add_argument("--restore-episode", help="Restore a recorded initial state/config for a fresh policy trial")
    parser.add_argument("--uri", default=URI)
    parser.add_argument("--task", default="BananaInBowlTask")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, help="Explicit per-episode policy seed; Thor must acknowledge it")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--execute-horizon", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--condition", default="default")
    parser.add_argument("--goal-variant", choices=['official','single-red-hammer-60s','single-red-hammer-no-drill-60s'], default='official')
    parser.add_argument("--post-terminal-seconds", type=float, default=0)
    parser.add_argument("--scenario-id")
    parser.add_argument("--experiment-track",choices=['new_standalone','banana_link'])
    parser.add_argument("--pose-range-m", type=float, default=0,
                        help="Uniform x/y perturbation of banana and bowl, with collision checking.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if bool(args.scenario_id) != bool(args.experiment_track):
        parser.error('Provide both --scenario-id and --experiment-track')
    if args.mode == "predict" and (args.episodes != 1 or args.max_steps):
        parser.error("Predict mode requires one episode; use --execute-horizon for its length")
    if args.replay_episode:
        if args.restore_episode and Path(args.replay_episode).resolve() != Path(args.restore_episode).resolve():
            parser.error("Conflicting restore sources")
        args.restore_episode = args.replay_episode
    if args.mode == "replay" and not args.restore_episode:
            parser.error("Replay requires --replay-episode")
    if args.restore_episode:
        if args.mode == "policy" and args.policy_seed is None:
            parser.error("Controlled policy retest requires --policy-seed acknowledged by Thor")
        source = Path(args.restore_episode).resolve()
        source_run = json.loads((source.parent / "run.json").read_text())
        source_episode = json.loads((source / "episode.json").read_text())
        current = provenance()
        for name in ("isaacsim", "isaaclab", "torch", "robolab"):
            if source_run["environment"]["packages"][name] != current["packages"][name]:
                parser.error(f"Replay requires matching {name} version")
        if source_run["environment"]["robolab_commit"] != current["robolab_commit"]:
            parser.error("Replay requires the recorded RoboLab commit")
        args.task = source_run["arguments"]["task"]
        args.goal_variant=source_run["arguments"].get("goal_variant","official")
        args.seed = source_episode["seed"]
        args.pose_range_m = source_run["arguments"]["pose_range_m"]
        args.replay_episode = str(source)
        args.restore_episode = str(source)
        args.episodes = 1
    if args.episodes < 1 or args.max_steps < 0 or args.pose_range_m < 0 or args.execute_horizon < 1 or args.timeout <= 0:
        parser.error("Invalid episode count, step count, pose range, horizon, or timeout")
    if not np.isfinite(args.post_terminal_seconds) or not 0 <= args.post_terminal_seconds <= 10:
        parser.error('Post-terminal observation must be between 0 and 10 seconds')
    if args.goal_variant != 'official' and (args.task != 'ToolOrganizationTask' or (args.restore_episode and args.mode!='replay')):
        parser.error('Single-hammer variant requires ToolOrganizationTask without restore')
    # Validate URI format before it enters persisted configuration.
    from urllib.parse import urlsplit
    uri = urlsplit(args.uri)
    if uri.scheme not in ("ws", "wss") or not uri.hostname or uri.username or uri.password or uri.query or uri.fragment:
        parser.error("Use a ws(s) URI without credentials; tokens belong in COSMOS3_API_TOKEN")
    if args.mode == "smoke" and args.max_steps == 0:
        args.max_steps = 60
    run_id = args.run_id or (datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z") + "_" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9_+.-]+", run_id) or run_id in (".", ".."):
        parser.error("Invalid run ID")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run.json", {"run_id": run_id, "arguments": vars(args), "environment": provenance()})
    args.enable_cameras = True
    results = []
    app = None
    run_error = None
    try:
        app = AppLauncher(args).app
        from robolab.constants import set_output_dir
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
        auto_register_droid_envs(task=args.task, cameras=WRIST_LEFT_RIGHT_HEAD)
        env_names = get_envs(task=args.task)
        if len(env_names) != 1:
            raise RuntimeError(f"Expected exactly one registered environment, got {env_names}")
        events = None
        if args.pose_range_m:
            from isaaclab.managers import EventTermCfg
            from robolab.core.events.reset_pose import reset_pose_uniform
            radius = args.pose_range_m
            # Replace the default reset: this function also resets all other assets.
            # Adding a separate event lets config ordering reset randomized poses back to defaults.
            events = {"reset": EventTermCfg(func=reset_pose_uniform, mode="reset", params={
                "pose_range": {"x": (-radius, radius), "y": (-radius, radius), "z": (0.0, 0.0)},
                "velocity_range": {}, "asset_cfg": ["banana", "bowl"],
                "reset_to_default_otherwise": True, "use_collision_check": True})}
        for index in range(args.episodes):
            episode_id = f"episode_{index:04d}"
            output = run_dir / episode_id
            output.mkdir()
            set_output_dir(str(output))
            env = None
            try:
                scene = env_names[0]
                if args.goal_variant in ('single-red-hammer-60s','single-red-hammer-no-drill-60s'):
                    from robolab.core.environments.config import parse_env_cfg
                    from robolab.core.task.conditionals import pick_and_place
                    scene=parse_env_cfg(env_names[0],device=args.device,seed=args.seed,num_envs=1,use_fabric=True)
                    scene.instruction={'default':'Put the red hammer in the left bin'}
                    scene.terminations.success.params['object']=['red_hammer']
                    scene.subtasks=[pick_and_place(object=['red_hammer'],container='left_bin',logical='all',score=1.0)]
                    scene.episode_length_s=60
                    if args.goal_variant == 'single-red-hammer-no-drill-60s':
                        scene.scene.cordless_drill.init_state.pos=(5.0,5.0,1.0)
                if args.restore_episode:
                    from robolab.core.environments.config import parse_env_cfg
                    from robolab.core.replay import apply_recorded_env_cfg
                    if isinstance(scene,str):
                        scene = parse_env_cfg(env_names[0], device=args.device, seed=args.seed,
                                              num_envs=1, use_fabric=True)
                    recorded = json.loads((Path(args.restore_episode) / "env_cfg.json").read_text())
                    # JSON stringifies functools.partial in nested condition lists.
                    # Verify the definitions and retain the live callables from the pinned source.
                    recorded_subtasks = recorded.pop("subtasks", None)
                    current_subtasks = scene.to_dict().get("subtasks")
                    def normalize_subtasks(value):
                        return re.sub(r"0x[0-9a-fA-F]+", "<address>", json.dumps(value, default=str, sort_keys=True))
                    if normalize_subtasks(recorded_subtasks) != normalize_subtasks(current_subtasks):
                        raise RuntimeError("Replay subtask definitions differ from the recording")
                    skipped = apply_recorded_env_cfg(scene, recorded)
                    # Private fields hold resolved metadata; all public config must match.
                    meaningful = [key for key in skipped if not key.startswith("/_")]
                    if meaningful:
                        raise RuntimeError(f"Replay config could not be restored: {meaningful}")
                if args.post_terminal_seconds:
                    from robolab.core.environments.config import parse_env_cfg
                    from robolab.core.sensors.contact_sensor_utils import create_contact_sensors
                    if isinstance(scene,str):
                        scene=parse_env_cfg(scene,device=args.device,seed=args.seed,num_envs=1,use_fabric=True)
                    # Supplemental right-finger sensor; preserve the official left-only goal predicate.
                    scene.contact_gripper={**scene.contact_gripper,
                        'audit_right':'{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/right_inner_finger'}
                    create_contact_sensors(scene)
                env, cfg = create_env(scene, device=args.device, seed=args.seed + index,
                                      num_envs=1, events=events, policy="cosmos3" if args.mode in ("policy", "predict") else "diagnostic")
                result = execute_episode(args, env, cfg, output, run_id, episode_id)
            except Exception as exc:
                result = {"run_id": run_id, "episode_id": episode_id, "status": "execution_error",
                          "error_type": type(exc).__name__, "error": str(exc)}
                (output / "error.txt").write_text(traceback.format_exc())
                write_json(output / "episode.json", result)
                traceback.print_exc()
            finally:
                if env is not None:
                    env.close()
            manifest(output)
            results.append(result)
            write_json(run_dir / "summary.json", summarize(results))
            # Stop on infrastructure failure, so a broken server doesn't consume a whole evaluation batch.
            if result["status"] == "execution_error":
                break
    except Exception as exc:
        run_error = str(exc)
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        write_json(run_dir / "summary.json", summarize(results))
        exit_code = int(run_error is not None or len(results) != args.episodes or
                        any(r["status"] in ("execution_error", "incomplete") for r in results))
        write_json(run_dir / "run_status.json", {"exit_code": exit_code, "error": run_error,
                                                "planned_episodes": args.episodes})
        print(f"Spark artifacts: {run_dir}", flush=True)
        if app is not None:
            app.close()
    if not results or any(r["status"] in ("execution_error", "incomplete") for r in results):
        raise SystemExit(1)


def supervised_main():
    """Kit may exit the process during close(); preserve the experiment exit status."""
    if os.environ.get("COSMOS_SPARK_WORKER") == "1" or "--help" in sys.argv or "-h" in sys.argv:
        main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id")
    parsed, _ = parser.parse_known_args()
    run_id = parsed.run_id or (datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z") + "_" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9_+.-]+", run_id) or run_id in (".", ".."):
        raise SystemExit("Invalid run ID")
    if (ROOT / "runs" / run_id).exists():
        raise SystemExit(f"Run already exists: {run_id}")
    command = [sys.executable, "-u", "-m", "cosmos_spark.runner", *sys.argv[1:]]
    if parsed.run_id is None:
        command.extend(["--run-id", run_id])
    completed = subprocess.run(command, env={**os.environ, "COSMOS_SPARK_WORKER": "1"})
    status_path = ROOT / "runs" / run_id / "run_status.json"
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if not status_path.exists():
        raise SystemExit("Simulator exited without a final experiment status")
    raise SystemExit(json.loads(status_path.read_text())["exit_code"])


if __name__ == "__main__":
    supervised_main()
