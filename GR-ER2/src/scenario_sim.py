"""Isaac Sim cabinet manipulation and quadruped inspection.

Drawer motion uses calibrated RMPFlow phases; Spot uses a learned walking policy.
ER2 gets RGB, never this module's evaluation trace. Drawer control uses simulator state; navigation uses simulated
localization and a hand-authored route graph, with local raycast stopping.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", required=True, choices=["drawer", "patrol"])
parser.add_argument("--gui", action="store_true")
parser.add_argument("--stream", action="store_true")
parser.add_argument("--stream-ip", default="")
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown
if args.stream_ip:
    sys.argv += [f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={args.stream_ip}"]
from isaacsim import SimulationApp
app = SimulationApp({"headless": not args.gui, "hide_ui": not(args.gui or args.stream), "width": 1280,
                     "height": 720, "renderer": "RaytracedLighting"},
                    experience="/isaac-sim/apps/isaacsim.exp.full.streaming.kit" if args.stream else "")

import numpy as np
import warp as wp
from PIL import Image
from pxr import UsdLux
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from drawer_motion import DrawerMotion
from isaacsim.robot.policy.examples.robots.spot import SpotFlatTerrainPolicy
from isaacsim.core.deprecation_manager import import_module
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from omni.physx import get_physx_scene_query_interface
from common import ROOT, read_json, write_json, safe_child, uid
from scenario_rules import ROUTES, steering, skill_request
torch = import_module("torch")

runtime = ROOT / "runtime"
for name in ("queue", "replies", "observations"):
    (runtime / name).mkdir(parents=True, exist_ok=True)
for path in (runtime / "queue").glob("*.json"):
    write_json(runtime / "replies" / path.name, {"error": "Simulator restarted"})
    path.unlink()
(runtime / "STOP").unlink(missing_ok=True)
(runtime / "SHUTDOWN").unlink(missing_ok=True)
write_json(runtime / "status.json", {"ready": False, "updated": time.time(), "world_scenario": args.scenario})
# Drawer contact control uses 120 Hz; Spot matches its shipped 500 Hz env.yaml.
# Rendered and non-rendered steps advance equal dt.
HZ = 120 if args.scenario == "drawer" else 500
DT = 1/HZ
RENDER_EVERY = 2 if args.scenario == "drawer" else 10
RECORD_EVERY = HZ//10
world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
world.scene.add_default_ground_plane()
UsdLux.DomeLight.Define(world.stage, "/World/Light").CreateIntensityAttr(1800)
assets = get_assets_root_path()

def box(name, pos, scale, color):
    return world.scene.add(FixedCuboid("/World/"+name, name=name, position=np.array(pos),
                                    scale=np.array(scale), color=np.array(color)))

if args.scenario == "drawer":
    cabinet_path = assets + "/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
    add_reference_to_stage(cabinet_path, "/World/cabinet")
    cabinet = Articulation(paths="/World/cabinet", positions=[0.8, 0, 0.4], orientations=[0, 0, 0, 1])
    policy = DrawerMotion("/World/franka", cabinet=cabinet, assets=assets)
    eye, look = [-.6, -1.6, 1.3], [0.55, 0, 0.55]
    side_eye = [-1.2, 1.7, 1.4]
else:
    policy = SpotFlatTerrainPolicy("/World/Spot", position=[0, 0, 0.65])
    cabinet = None
    box("BackWall", [5.8, 0, 1], [.15, 6, 2], [.45, .48, .55])
    box("NorthWall", [2.5, 3.1, .7], [6.5, .15, 1.4], [.55, .60, .66])
    box("SouthWall", [2.5, -3.1, .7], [6.5, .15, 1.4], [.55, .60, .66])
    box("Station", [5.0, 0, .6], [.45, 1.0, 1.2], [.15, .25, .36])
    box("Indicator", [4.765, 0, .95], [.04, .28, .28], [1, .015, .015])
    box("Panel", [4.765, 0, .52], [.04, .65, .25], [.85, .85, .85])
    box("CenterRoute", [2.3, 0, .006], [4.8, .12, .012], [.05, .5, .9])
    box("RightRoute", [2.5, -1.8, .006], [3.0, .12, .012], [.95, .65, .03])
    barrier = box("Barrier", [2.5, 6, .5], [.4, 1.7, 1.0], [.95, .35, .03])
    eye, look, side_eye = [.6, 0, .9], [4.8, 0, .8], [5.5, -6.0, 5.0]

if not math.isclose(policy._dt, DT, rel_tol=1e-6):
    raise RuntimeError(f"Policy env.yaml dt changed: expected {DT}, got {policy._dt}")
camera = Camera("/World/AgentCamera", resolution=(640, 480))
side = Camera("/World/Overview", resolution=(640, 480))
set_camera_view(eye=np.array(eye), target=np.array(look), camera_prim_path=camera.prim_path)
set_camera_view(eye=np.array(side_eye), target=np.array(look if args.scenario == "drawer" else [2.5, 0, 0]), camera_prim_path=side.prim_path)
set_camera_view(eye=np.array(side_eye), target=np.array(look if args.scenario == "drawer" else [2.5, 0, 0]))
world.reset()
policy.initialize()
policy.post_reset()
robot_original_gains = [x.numpy().copy() for x in policy.robot.get_dof_gains()]
for cam in (camera, side):
    cam.initialize()
    cam.set_clipping_range(.03, 30)
    cam.set_focal_length(2.4)
    cam.set_horizontal_aperture(2.4)
if cabinet:
    drawer_index = cabinet.get_dof_indices("drawer_top_joint")
    cabinet_original_gains = [x.numpy().tolist() for x in cabinet.get_dof_gains(dof_indices=drawer_index)]
    # This lesson uses a passive drawer without a spring closer. The pretrained
    # task used stiffness=10, damping=1; declare this scene adaptation explicitly.
    cabinet.set_dof_gains(stiffnesses=[0.0], dampings=[10.0], dof_indices=drawer_index)
active, recording, injection = None, None, False
step = 0

def pose():
    pos, quat = policy.robot.get_world_poses()
    p, q = pos.numpy()[0], quat.numpy()[0]
    yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
    return p, yaw

def observation():
    ident = uid()
    target = ROOT / "outputs/observations" / ident / "camera.png"
    save_image(target, camera)
    return {"observation_id": ident, "image": str(target.relative_to(ROOT)),
            "captured_wall_time": time.time(), "simulation_time": world.current_time}

def save_image(path, cam):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(rgb).save(temporary)
    os.replace(temporary, path)

def event(kind, **data):
    if recording:
        write_json(recording["out"] / "events" / (uid()+".json"),
                   {"type": kind, "wall_time": time.time(), "simulation_time": world.current_time,
                    "evaluation_only": True, **data})

def finish(status):
    global active
    if cabinet and status != "completed":
        policy.stop(status)
    write_json(runtime / "replies" / (active["id"]+".json"), {"controller_status": status})
    event("skill_end", status=status, skill=active["skill"])
    active = None

def process(req):
    global active, recording, injection
    reply_path = runtime / "replies" / (req["id"]+".json")
    command = req["command"]
    if time.time()-req["created"] > 300:
        raise ValueError("Expired request")
    if command == "shutdown":
        (runtime/"SHUTDOWN").touch()
        write_json(reply_path, {"shutdown": True})
    elif command in ("hold", "stop"):
        if active:
            finish("stopped_by_user")
        if command == "stop":
            (runtime / "STOP").touch()
        write_json(reply_path, {"stopped": True})
    elif command == "snapshot":
        write_json(reply_path, observation())
    elif command == "reset":
        if active or recording:
            raise ValueError("Cannot reset during an experiment")
        policy.post_reset()
        policy.robot.set_dof_gains(stiffnesses=robot_original_gains[0], dampings=robot_original_gains[1])
        if cabinet:
            zeros = np.zeros_like(cabinet.get_dof_positions().numpy())
            cabinet.set_dof_positions(zeros)
            cabinet.set_dof_velocities(zeros)
        else:
            policy._policy_counter = 0
            policy._previous_action.zero_()
            policy.robot.set_world_poses(positions=[[0, 0, .65]], orientations=[[1, 0, 0, 0]])
            barrier.set_world_pose(np.array([2.5, 6, .5]))
        (runtime/"STOP").unlink(missing_ok=True)
        write_json(reply_path, {"reset": True})
    elif command == "record_start":
        if recording or active:
            raise ValueError("A run is already active")
        out = safe_child(ROOT/"outputs", req["output"].removeprefix("outputs/"))
        out.mkdir(parents=True, exist_ok=True)
        recording = {"out": out, "frames": 0, "wall_start": time.time(), "sim_start": world.current_time}
        if cabinet:
            policy.reset_quality()
        injection = bool(req.get("perturb", True)) if args.scenario == "patrol" else False
        write_json(out / "policy_sources.json", {"class": type(policy).__name__, "assets_root": assets,
            "physics_dt": DT, "trained_physics_dt": policy._dt, "policy_decimation": policy._decimation,
            "drawer_scene_adaptation": {"original_gains": cabinet_original_gains, "stiffness": 0, "damping": 10, "reason": "passive damped drawer; other joints unchanged"} if cabinet else None,
            "drawer_control": {"type": "RMPFlow + calibrated Cartesian phases", "learned_drawer_policy_used": False,
                "constant_robot_gains": [x.tolist() for x in robot_original_gains], "joint_command_speed_limit": .5,
                "quality_sampling_hz": HZ, "joint_hardware_velocity_limit": .7, "excess_speed_stop": 1.0} if cabinet else None,
            "low_level_observation": "privileged drawer/robot state" if cabinet else "proprioception + simulated localization + raycast safety",
            "model_input": "RGB only; no evaluation trace"})
        write_json(reply_path, {"recording": True})
    elif command == "record_stop":
        if not recording:
            raise ValueError("No active recording")
        result = {"frames": recording["frames"], "fps": 10, "timebase": "simulation_time",
                  "wall_seconds": time.time()-recording["wall_start"], "simulation_seconds": world.current_time-recording["sim_start"]}
        write_json(recording["out"] / "recording.json", result)
        recording = None
        write_json(reply_path, result)
    elif command == "perturb" and cabinet and not active:
        previous = float(cabinet.get_dof_positions(dof_indices=drawer_index).numpy().ravel()[0])
        if previous < .15 or np.linalg.norm(policy.tcp()-policy.handle_center()) < .07:
            raise ValueError("Refusing ineffective/unsafe closure: drawer must be open and gripper withdrawn")
        cabinet.set_dof_positions([[0.0]], dof_indices=drawer_index)
        cabinet.set_dof_velocities([[0.0]], dof_indices=drawer_index)
        event("drawer_closed", before_open_m=previous, mechanism="explicit external reset of drawer joint; robot state unchanged")
        write_json(reply_path, {"injected": True})
    elif command == "skill":
        if active or (runtime/"STOP").exists():
            raise ValueError("Busy or stopped")
        skill = skill_request(args.scenario, req)
        active = {**skill, "id": req["id"], "start": world.current_time, "waypoint": 0, "open_ticks": 0}
        if cabinet:
            policy.begin()
        event("skill_start", **skill)
    elif command == "metrics":
        # Only the offline evaluator/manual checks use this endpoint.
        write_json(reply_path, metrics())
    else:
        raise ValueError("Command is not supported in this scene")

def metrics():
    result = {"wall_time": time.time(), "simulation_time": world.current_time, "evaluation_only": True}
    if cabinet:
        result["drawer_open_m"] = float(cabinet.get_dof_positions(dof_indices=drawer_index).numpy().ravel()[0])
        result.update(policy.metrics(DT))
    else:
        p, yaw = pose()
        result.update(position=p.tolist(), yaw=yaw)
    return result

try:
    for _ in range(100):
        if cabinet:
            policy.tick(DT)
        else:
            policy.forward(DT, torch.zeros(3))
        world.step(render=True)
    write_json(runtime/"scene.json", {"scenario": args.scenario, "policy": type(policy).__name__})
    print("GR_ER2_READY " + args.scenario, flush=True)
    while app.is_running() and not (runtime/"SHUTDOWN").exists():
        started = time.monotonic()
        if (runtime/"STOP").exists() and active:
            finish("stopped_by_user")
        if cabinet:
            previous_phase = policy.phase
            policy.tick(DT)
            if policy.phase != previous_phase:
                event("control_phase", phase=policy.phase)
            if active and policy.result:
                finish(policy.result)
            if active and world.current_time-active["start"] >= active["seconds"]:
                finish("timeout")
        else:
            p, yaw = pose()
            command = [0.0, 0.0, 0.0]
            if active:
                route = ROUTES[active["route"]]
                target = route[active["waypoint"]]
                if active.get("align"):
                    command = [0, 0, max(-.8, min(.8, -yaw*1.8))]
                    if abs(yaw) < .12:
                        finish("arrived")
                elif np.linalg.norm(p[:2]-np.array(target)) < .23:
                    active["waypoint"] += 1
                    if active["waypoint"] == len(route):
                        active["waypoint"] = len(route)-1
                        active["align"] = True
                if active:
                    if not active.get("align"):
                        command = steering(p, yaw, route[active["waypoint"]])
                    # Forward collision guard is independent of API latency.
                    direction = (math.cos(yaw), math.sin(yaw), 0.0)
                    for offset in (-.22, 0, .22):
                        origin = (float(p[0]+.55*direction[0]-offset*direction[1]),
                                  float(p[1]+.55*direction[1]+offset*direction[0]), .55)
                        hit = get_physx_scene_query_interface().raycast_closest(origin, direction, .7)
                        if hit.get("hit") and "/Spot" not in str(hit.get("rigidBody", "")) and command[0] > .05:
                            event("obstacle_stop", hit_distance=hit.get("distance"))
                            finish("blocked")
                            command = [0, 0, 0]
                            break
                    if active and world.current_time-active["start"] > active["seconds"]:
                        finish("timeout")
                        command = [0, 0, 0]
                    if p[2] < .2:
                        if active:
                            finish("fallen")
                        command = [0, 0, 0]
            if injection and p[0] > .8:
                barrier.set_world_pose(np.array([2.5, 0, .5]))
                event("barrier_inserted", mechanism="explicit relocation of obstacle into center route")
                injection = False
            policy.forward(DT, torch.tensor(command, dtype=torch.float32))
            if step % RENDER_EVERY == 0:
                forward = np.array([math.cos(yaw), math.sin(yaw), 0])
                camera_pos = p + .5*forward + np.array([0, 0, .32])
                set_camera_view(eye=camera_pos, target=camera_pos+3*forward+np.array([0, 0, -.1]), camera_prim_path=camera.prim_path)
        world.step(render=step % RENDER_EVERY == 0, update_fabric=True)
        step += 1
        if step % RECORD_EVERY == 0:
            save_image(runtime/"camera.jpg", camera)
            if recording:
                frame = recording["frames"]
                for folder, cam in (("input_frames", camera), ("side_frames", side)):
                    save_image(recording["out"]/folder/f"{frame:06d}.jpg", cam)
                for filename, row in (("frame_timestamps.jsonl", {"frame": frame, "wall_time": time.time(), "simulation_time": world.current_time}),
                                      ("evaluation_trace.jsonl", {"frame": frame, **metrics()})):
                    with (recording["out"]/filename).open("a") as handle:
                        handle.write(json.dumps(row)+"\n")
                recording["frames"] += 1
            write_json(runtime/"status.json", {"ready": True, "busy": bool(active), "updated": time.time(),
                       "world_scenario": args.scenario, "simulation_time": world.current_time})
            for path in sorted((runtime/"queue").glob("*.json")):
                try:
                    process(read_json(path))
                except Exception as exc:
                    traceback.print_exc()
                    write_json(runtime/"replies"/path.name, {"error": str(exc)})
                finally:
                    path.unlink(missing_ok=True)
        time.sleep(max(0, DT-(time.monotonic()-started)))
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    write_json(runtime/"status.json", {"ready": False, "updated": time.time(), "world_scenario": args.scenario})
    app.close()
