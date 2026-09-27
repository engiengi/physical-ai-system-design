"""Isaac Sim 6 standalone lab. Uses real contacts and NVIDIA's Franka controller.

Run with /isaac-sim/python.sh, never the host's virtualenv Python.
Only evaluator code reads object poses. Action targets come from RGB-D pixels.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
display_mode = parser.add_mutually_exclusive_group()
display_mode.add_argument("--stream", action="store_true")
display_mode.add_argument("--gui", action="store_true")
parser.add_argument("--stream-ip", default="")
args, kit_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + kit_args
if args.stream_ip:
    sys.argv.append(f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={args.stream_ip}")
if args.stream:
    sys.argv.extend(["--/app/livestream/allowResize=false",
                     "--/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=false",
                     "--/exts/omni.services.livestream.session/quitOnSessionEnded=false"])

from isaacsim import SimulationApp

settings = {"headless": not args.gui, "hide_ui": not (args.stream or args.gui), "width": 1280, "height": 720,
            "window_width": 1280, "window_height": 720, "renderer": "RaytracedLighting"}
experience = "/isaac-sim/apps/isaacsim.exp.full.streaming.kit" if args.stream else ""
app = SimulationApp(settings, experience=experience)

import json
import math
import traceback
import numpy as np
from PIL import Image
from pxr import UsdLux
import carb
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.sensors.camera import Camera
from common import ROOT, pixel, read_json, safe_child, uid, validate_action, write_json

if args.stream_ip:
    carb.settings.get_settings().set("/exts/omni.kit.livestream.app/primaryStream/publicIp", args.stream_ip)

runtime = ROOT / "runtime"
for path in (runtime / "queue", runtime / "replies", runtime / "observations", ROOT / "data/samples"):
    path.mkdir(parents=True, exist_ok=True)
# Fail abandoned requests explicitly; never execute old jobs on restart.
for path in (runtime / "queue").glob("*.json"):
    write_json(runtime / "replies" / path.name, {"error": "시뮬레이터 재시작으로 요청이 취소되었습니다."})
    path.unlink()
(runtime / "STOP").unlink(missing_ok=True)
write_json(runtime / "status.json", {"ready": False, "updated": time.time(), "note": "Isaac assets/cameras loading"})

world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/60)
world.scene.add_default_ground_plane(z_position=-0.06)
world.scene.add(FixedCuboid("/World/Table", name="table", position=np.array([0.4, 0, -0.025]),
                           scale=np.array([1.15, 1.05, 0.05]), color=np.array([0.62, 0.65, 0.7])))
robot = world.scene.add(Franka("/World/Franka", name="franka"))
light = UsdLux.DomeLight.Define(world.stage, "/World/Light")
light.CreateIntensityAttr(1400)
colors = {"red": [0.85, 0.045, 0.025], "blue": [0.035, 0.15, 0.85], "yellow": [0.95, 0.72, 0.025]}
SCENES = [
    [[0.40, -0.20, 0.026], [0.57, -0.19, 0.026], [0.55, -0.03, 0.026]],
    [[0.58, -0.23, 0.026], [0.39, -0.12, 0.026], [0.59, -0.03, 0.026]],
    [[0.37, -0.22, 0.026], [0.48, -0.15, 0.026], [0.61, -0.12, 0.026]]
]
cubes = {}
for (name, color), pos in zip(colors.items(), SCENES[0]):
    cubes[name] = world.scene.add(DynamicCuboid(f"/World/{name}", name=name, position=np.array(pos),
        size=0.05, color=np.array(color), mass=0.05))
tray_center = np.array([0.49, 0.235])
tray_parts = [world.scene.add(FixedCuboid("/World/TrayBase", name="tray_base", position=np.array([*tray_center, 0.005]),
                           scale=np.array([0.28, 0.20, 0.01]), color=np.array([0.08, 0.48, 0.13])))]
for i, (pos, scale) in enumerate([
    ([0.345, 0.235, 0.025], [0.01, 0.21, 0.05]), ([0.635, 0.235, 0.025], [0.01, 0.21, 0.05]),
    ([0.49, 0.13, 0.025], [0.30, 0.01, 0.05]), ([0.49, 0.34, 0.025], [0.30, 0.01, 0.05])]):
    tray_parts.append(world.scene.add(FixedCuboid(f"/World/TrayWall{i}", name=f"tray_wall_{i}", position=np.array(pos),
                               scale=np.array(scale), color=np.array([0.07, 0.38, 0.1]))))
tray_original = [part.get_world_pose()[0].copy() for part in tray_parts]

camera = Camera("/World/TopCamera", position=np.array([0.47, 0, 1.25]),
                orientation=euler_angles_to_quat(np.array([0, 90, 0]), degrees=True),
                resolution=(640, 480), frequency=60)
side = Camera("/World/SideCamera", resolution=(640, 480), frequency=60)
world.reset()
home_joints = robot.get_joint_positions().copy()
for cam in (camera, side):
    cam.initialize()
    cam.set_clipping_range(0.01, 10)
    # Camera API uses stage units here (USD tenths-of-stage-unit optics),
    # so 2.4 corresponds to a 24 mm lens, not 24.
    cam.set_focal_length(2.4)
    cam.set_horizontal_aperture(2.4)
camera.add_distance_to_image_plane_to_frame()
set_camera_view(eye=np.array([1.25, 1.1, 0.95]), target=np.array([0.4, 0, 0.08]), camera_prim_path=side.prim_path)
set_camera_view(eye=np.array([1.25, 1.1, 0.95]), target=np.array([0.4, 0, 0.08]))
controller = PickPlaceController(name="pick_place", gripper=robot.gripper, robot_articulation=robot,
                                 end_effector_initial_height=0.25)
art = robot.get_articulation_controller()
robot.gripper.set_default_state(robot.gripper.joint_opened_positions)
revision, scene, active, step = 0, 0, None, 0
recording = None
homing = None


def rgba(cam):
    value = np.asarray(cam.get_rgba())
    if value.ndim != 3 or value.shape[0] != 480:
        raise RuntimeError("Camera frame is not ready")
    return value[:, :, :3].astype(np.uint8)


def save_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(value).save(temp)
    os.replace(temp, path)


def snapshot():
    obs = uid()
    directory = runtime / "observations" / obs
    directory.mkdir()
    rgb = rgba(camera)
    depth = np.asarray(camera.get_depth())
    if depth.shape != (480, 640):
        raise RuntimeError("Depth frame is not ready")
    save_image(directory / "camera.png", rgb)
    np.save(directory / "depth.npy", depth)
    meta = {"observation_id": obs, "revision": revision, "created": time.time(), "simulation_time": world.current_time,
            "resolution": [640, 480], "intrinsics": camera.get_intrinsics_matrix().tolist(),
            "camera_pose": [v.tolist() for v in camera.get_world_pose()]}
    write_json(directory / "observation.json", meta)
    # Public immutable image. Depth remains in runtime, accessible only to controller.
    target = ROOT / "outputs/observations" / obs / "camera.png"
    save_image(target, rgb)
    write_json(target.parent / "observation.json", meta)
    return {"observation_id": obs, "image": str(target.relative_to(ROOT)), "revision": revision,
            "captured_wall_time": meta["created"], "simulation_time": meta["simulation_time"]}


def resolve_point(value, depth, kind):
    x, y = pixel(value, 640, 480)
    ix, iy = int(round(x)), int(round(y))
    window = depth[max(0, iy-1):iy+2, max(0, ix-1):ix+2]
    valid = window[np.isfinite(window) & (window > 0) & (window < 3)]
    if valid.size < 3:
        raise ValueError("선택한 위치의 깊이 값이 유효하지 않습니다.")
    xyz = camera.get_world_points_from_image_coords(np.array([[x, y]]), np.array([float(np.median(valid))]))[0]
    if not (0.25 <= xyz[0] <= 0.7 and -0.32 <= xyz[1] <= 0.36):
        raise ValueError("로봇 작업 범위 밖의 위치입니다.")
    if kind == "pick":
        if not 0.035 <= xyz[2] <= 0.10:
            raise ValueError("블록 윗면 중앙을 선택하세요. 테이블/로봇/트레이는 집기 대상에서 제외합니다.")
        xyz[2] -= 0.025  # Known 5 cm cube height; no simulator object pose lookup.
    else:
        if not -0.01 <= xyz[2] <= 0.03:
            raise ValueError("비어 있는 테이블 또는 트레이 바닥을 선택하세요.")
        xyz[2] += 0.025
    return xyz


def reply(request, value):
    write_json(runtime / "replies" / (request["id"] + ".json"), value)


def evaluator(before, pick, place):
    # Privileged data for saved evaluation only; never returned to the reasoner.
    after = {name: cube.get_world_pose()[0].tolist() for name, cube in cubes.items()}
    name = min(before, key=lambda k: np.linalg.norm(np.array(before[k]) - pick))
    distance = float(np.linalg.norm(np.array(after[name])[:2] - place[:2]))
    displacement = float(np.linalg.norm(np.array(after[name]) - np.array(before[name])))
    speed = float(np.linalg.norm(cubes[name].get_linear_velocity()))
    position_ok = distance < 0.055 and abs(after[name][2] - place[2]) < 0.035
    fingers_open = bool(np.min(robot.gripper.get_joint_positions()) > 0.025)
    return {"evaluation_only": True, "picked_candidate": name, "before": before, "after": after,
            "tray_center": tray_parts[0].get_world_pose()[0][:2].tolist(),
            "distance_to_requested_place_m": distance, "displacement_m": displacement,
            "object_speed_m_s": speed, "gripper_open": fingers_open,
            "placement_within_tolerance": bool(position_ok and displacement > 0.04 and speed < 0.03 and fingers_open),
            "instruction_success": "not_automatically_assessed",
            "criterion": "requested 3D place XY error <5.5cm, Z error <3.5cm, displacement >4cm, speed <3cm/s, open gripper after 1s settling"}


def finish(status):
    global active, revision
    state = active
    metrics = evaluator(state["before"], state["pick"], state["place"])
    metrics.update({"controller_status": status, "simulation_seconds": world.current_time - state["sim_start"],
                    "wall_seconds": time.monotonic() - state["wall_start"], "video_fps": 10,
                    "video_timebase": "simulation_time", "frames": state["frames"],
                    "pick_world_from_depth": state["pick"].tolist(), "place_world_from_depth": state["place"].tolist()})
    write_json(state["out"] / "metrics.json", metrics)
    save_image(state["out"] / "final.png", rgba(camera))
    reply(state["request"], {"controller_status": status, "metrics": str((state["out"] / "metrics.json").relative_to(ROOT))})
    revision += 1
    active = None


def perturb_red(source):
    """Explicit experimental teleport, never used to implement grasping."""
    global revision
    cube = cubes["red"]
    before = cube.get_world_pose()[0].tolist()
    cube.set_world_pose(np.array([0.60, -0.27, 0.027]), np.array([1, 0, 0, 0]))
    cube.set_linear_velocity(np.zeros(3))
    cube.set_angular_velocity(np.zeros(3))
    revision += 1
    event = {"type": "user_perturbation", "source": source, "object": "red",
             "wall_time": time.time(), "simulation_time": world.current_time,
             "evaluation_only": True, "before_world": before,
             "after_world": [0.60, -0.27, 0.027],
             "action_tick": active["ticks"] if active else None}
    event_id = uid()
    event_dir = active["out"] if active else ROOT / "outputs/perturbations"
    write_json(event_dir / (event_id + ".json"), event)
    if recording:
        write_json(recording["out"] / "events" / (event_id + ".json"), event)
    return event


def scenario_perturbation(kind):
    """Injection uses scene state; this privileged event is never sent to the API."""
    global revision
    if kind == "target_shift":
        return perturb_red("scheduled_recovery_experiment")
    event = {"type": kind, "source": "scheduled_recovery_experiment", "evaluation_only": True,
             "wall_time": time.time(), "simulation_time": world.current_time, "action_tick": active["ticks"]}
    if kind == "goal_shift":
        event["before_world"] = tray_parts[0].get_world_pose()[0].tolist()
        for part, original in zip(tray_parts, tray_original):
            part.set_world_pose(original + np.array([-0.18, 0, 0]))
        event["after_world"] = tray_parts[0].get_world_pose()[0].tolist()
        event["mechanism"] = "explicit relocation of empty receptacle before grasp"
    elif kind == "grasp_drop":
        # Release a genuinely lifted body. Gravity/contact determine where it lands.
        active["force_open"] = True
        event["red_at_release"] = cubes["red"].get_world_pose()[0].tolist()
        event["mechanism"] = "override gripper opening after red rises above 12 cm; no object teleport"
    else:
        raise ValueError("Unknown scenario")
    revision += 1
    event_id = uid()
    write_json(active["out"] / (event_id + ".json"), event)
    if recording:
        write_json(recording["out"] / "events" / (event_id + ".json"), event)
    return event


def process(request):
    global active, revision, scene, recording, homing
    command = request["command"]
    if time.time() - request["created"] > 300:
        raise ValueError("만료된 요청입니다.")
    if command == "stop":
        (runtime / "STOP").touch()
        reply(request, {"stopping": True})
        return
    if command == "hold":
        interrupted = active is not None or homing is not None
        if interrupted:
            art.apply_action(ArticulationAction(joint_positions=robot.get_joint_positions().copy()))
        if active:
            finish("interrupted_by_stream_monitor")
        if homing:
            reply(homing["request"], {"controller_status": "interrupted_by_stream_monitor"})
            homing = None
            revision += 1
        reply(request, {"interrupted": interrupted})
        return
    if (active or homing) and command not in ("shift", "record_stop", "snapshot"):
        raise ValueError("로봇이 움직이고 있습니다. 완료 후 다시 시도하세요.")
    if command == "snapshot":
        reply(request, snapshot())
    elif command == "return_home":
        if (runtime / "STOP").exists():
            raise ValueError("사용자 정지 상태에서는 복귀하지 않습니다.")
        homing = {"request": request, "start": robot.get_joint_positions().copy(), "ticks": 0}
    elif command == "record_start":
        if recording:
            raise ValueError("이미 전체 실험 녹화 중입니다.")
        out = safe_child(ROOT / "outputs", request["output"].removeprefix("outputs/"))
        recording = {"out": out, "frames": 0, "sim_start": world.current_time,
                     "wall_start": time.time()}
        reply(request, {"recording": True})
    elif command == "record_stop":
        if not recording:
            raise ValueError("진행 중인 전체 녹화가 없습니다.")
        summary = {"frames": recording["frames"], "fps": 10, "timebase": "simulation_time",
                   "start_simulation_time": recording["sim_start"], "start_wall_time": recording["wall_start"],
                   "simulation_seconds": world.current_time - recording["sim_start"],
                   "wall_seconds": time.time() - recording["wall_start"],
                   "camera": "/World/TopCamera", "mount": "fixed_overhead",
                   "note": "Full camera recording; consult protocol.json for actual model input transport."}
        write_json(recording["out"] / "recording.json", summary)
        recording = None
        reply(request, summary)
    elif command in ("reset", "sample"):
        scene = int(request.get("scene", 0))
        if scene not in range(3):
            raise ValueError("Invalid scene")
        world.reset()
        controller.reset()
        for part, original in zip(tray_parts, tray_original):
            part.set_world_pose(original)
        for cube, pos in zip(cubes.values(), SCENES[scene]):
            cube.set_world_pose(np.array(pos), np.array([1, 0, 0, 0]))
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        for _ in range(60):
            world.step(render=True)
        revision += 1
        (runtime / "STOP").unlink(missing_ok=True)
        obs = snapshot()
        if command == "sample":
            save_image(ROOT / f"data/samples/scene_{scene+1:02d}.png", rgba(camera))
            write_json(ROOT / f"data/samples/scene_{scene+1:02d}.json", {"source": "Isaac Sim rendered RGB", "scene": scene, "observation": obs,
                "instructions": ["Find all colored blocks and the green tray.", "Find the block to the right of the red block.", "Plan how to put all blocks into the tray."]})
        reply(request, obs)
    elif command == "shift":
        perturb_red("manual_button")
        reply(request, {"perturbation_applied": True})
    elif command == "pick_place":
        scenario = request.get("scenario", "target_shift")
        if scenario not in ("target_shift", "goal_shift", "grasp_drop"):
            raise ValueError("Unknown recovery scenario")
        perturb_tick = request.get("perturb_after_ticks")
        if perturb_tick is not None and (type(perturb_tick) is not int or perturb_tick != 60):
            raise ValueError("자동 방해는 첫 동작 60 tick 시점만 허용합니다.")
        action = validate_action(request["action"])
        obs_dir = runtime / "observations" / action["observation_id"]
        meta = read_json(obs_dir / "observation.json")
        if meta["revision"] != revision or time.time() - meta["created"] > 300:
            raise ValueError("장면이 바뀌었거나 관측이 만료되었습니다. 새 관측을 촬영하세요.")
        if not all(np.allclose(now, old, atol=1e-5) for now, old in zip(camera.get_world_pose(), meta["camera_pose"])):
            raise ValueError("카메라가 이동했습니다. 새 관측을 촬영하세요.")
        if not np.allclose(camera.get_intrinsics_matrix(), meta["intrinsics"], atol=1e-4):
            raise ValueError("카메라 보정값이 바뀌었습니다. 새 관측을 촬영하세요.")
        if (runtime / "STOP").exists():
            raise ValueError("정지 상태입니다. 장면 초기화 후 다시 실행하세요.")
        depth = np.load(obs_dir / "depth.npy")
        pick, place = resolve_point(action["pick"], depth, "pick"), resolve_point(action["place"], depth, "place")
        out = safe_child(ROOT / "outputs", request["output"].removeprefix("outputs/"))
        out.mkdir(parents=True, exist_ok=True)
        save_image(out / "input_images/observation.png", np.array(Image.open(obs_dir / "camera.png")))
        write_json(out / "input_images/observation.json", meta)
        np.save(out / "input_images/depth.npy", depth)
        controller.reset()
        active = {"request": request, "out": out, "pick": pick, "place": place,
                  "before": {name: cube.get_world_pose()[0].tolist() for name, cube in cubes.items()},
                  "ticks": 0, "frames": 0, "settle": 0, "return_ticks": 0,
                  "perturb_after_ticks": perturb_tick,
                  "scenario": scenario, "force_open": False,
                  "wall_start": time.monotonic(), "sim_start": world.current_time}
    else:
        raise ValueError("Unknown simulator command")


try:
    for _ in range(120):
        world.step(render=True)
    # Render three runnable, real simulator samples on first startup.
    for number in range(3):
        process({"id": uid(), "command": "sample", "created": time.time(), "scene": number})
    process({"id": uid(), "command": "reset", "created": time.time(), "scene": 0})
    print("GR_ER2_READY", flush=True)
    while app.is_running():
        started = time.monotonic()
        if homing:
            if (runtime / "STOP").exists():
                art.apply_action(ArticulationAction(joint_positions=robot.get_joint_positions().copy()))
                reply(homing["request"], {"controller_status": "stopped_by_user"})
                homing = None
            else:
                homing["ticks"] += 1
                blend = (1 - math.cos(math.pi * homing["ticks"] / 120)) / 2
                target = homing["start"] * (1 - blend) + home_joints * blend
                target[-2:] = robot.gripper.joint_opened_positions
                art.apply_action(ArticulationAction(joint_positions=target))
                if homing["ticks"] == 120:
                    reply(homing["request"], {"controller_status": "completed", "note": "Returned to observation joints without resetting objects"})
                    homing = None
                    revision += 1
        if active:
            trigger = (cubes["red"].get_world_pose()[0][2] > 0.12 if active["scenario"] == "grasp_drop"
                       else active["ticks"] == active["perturb_after_ticks"])
            if not (runtime / "STOP").exists() and active["perturb_after_ticks"] is not None and trigger:
                scenario_perturbation(active["scenario"])
                active["perturb_after_ticks"] = None
            if (runtime / "STOP").exists():
                finish("stopped_by_user")
            elif active["ticks"] > 2400:
                finish("controller_timeout")
            elif not controller.is_done():
                action = controller.forward(picking_position=active["pick"], placing_position=active["place"],
                    current_joint_positions=robot.get_joint_positions(), end_effector_offset=np.array([0, 0.005, 0]))
                art.apply_action(action)
            elif active["return_ticks"] < 120:
                if active["return_ticks"] == 0:
                    active["return_start"] = robot.get_joint_positions().copy()
                active["return_ticks"] += 1
                blend = (1 - math.cos(math.pi * active["return_ticks"] / 120)) / 2
                target = active["return_start"] * (1 - blend) + home_joints * blend
                target[-2:] = robot.gripper.joint_opened_positions
                art.apply_action(ArticulationAction(joint_positions=target))
            else:
                active["settle"] += 1
                if active["settle"] >= 60:
                    finish("completed")
            if active and active["force_open"]:
                # forward returns a full-articulation action (9 entries), whereas
                # gripper.apply_action expects only the two finger entries.
                art.apply_action(robot.gripper.forward("open"))
        world.step(render=True)
        step += 1
        if active:
            active["ticks"] += 1
            if active["ticks"] % 6 == 0:
                frame = active["frames"]
                save_image(active["out"] / "input_frames" / f"{frame:06d}.jpg", rgba(camera))
                save_image(active["out"] / "side_frames" / f"{frame:06d}.jpg", rgba(side))
                active["frames"] += 1
        if step % 6 == 0:
            if recording:
                frame = recording["frames"]
                save_image(recording["out"] / "input_frames" / f"{frame:06d}.jpg", rgba(camera))
                save_image(recording["out"] / "side_frames" / f"{frame:06d}.jpg", rgba(side))
                with (recording["out"] / "frame_timestamps.jsonl").open("a") as clock_file:
                    clock_file.write(json.dumps({"frame": frame, "wall_time": time.time(),
                                                 "simulation_time": world.current_time}) + "\n")
                with (recording["out"] / "evaluation_trace.jsonl").open("a") as trace_file:
                    trace_file.write(json.dumps({"frame": frame, "wall_time": time.time(), "evaluation_only": True,
                        "red": cubes["red"].get_world_pose()[0].tolist(),
                        "fingers": robot.gripper.get_joint_positions().tolist(),
                        "tray_center": tray_parts[0].get_world_pose()[0][:2].tolist()}) + "\n")
                recording["frames"] += 1
            save_image(runtime / "camera.jpg", rgba(camera))
            write_json(runtime / "status.json", {"ready": True, "busy": active is not None or homing is not None, "updated": time.time(),
                       "revision": revision, "scene": scene, "simulation_time": world.current_time, "streaming": args.stream})
            for path in sorted((runtime / "queue").glob("*.json")):
                try:
                    request = read_json(path)
                    process(request)
                except Exception as exc:
                    traceback.print_exc()
                    write_json(runtime / "replies" / path.name, {"error": str(exc)})
                finally:
                    path.unlink(missing_ok=True)
        # Real-time cap; record measured wall time when rendering runs slower.
        time.sleep(max(0, 1/60 - (time.monotonic() - started)))
finally:
    write_json(runtime / "status.json", {"ready": False, "updated": time.time(), "note": "simulator stopped"})
    app.close()
