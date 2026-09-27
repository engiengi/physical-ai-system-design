"""Pure, testable bounds and metrics for the drawer and patrol lessons."""
import math

STATION_WAYPOINT = (3.4, 0)
ROUTES = {"center": [(1.4, 0), STATION_WAYPOINT],
          "right": [(1.0, -1.8), (3.4, -1.8), STATION_WAYPOINT]}


def skill_request(scene, payload):
    if scene == "drawer" and payload.get("skill") == "open_drawer":
        return {"skill": "open_drawer", "seconds": 55.0}
    if scene == "patrol" and payload.get("skill") == "navigate":
        route = payload.get("route")
        if route not in ROUTES:
            raise ValueError("Route must be center or right")
        return {"skill": "navigate", "route": route, "seconds": 50.0}
    raise ValueError("Skill does not match loaded scene")


def steering(position, yaw, target):
    dx, dy = target[0]-position[0], target[1]-position[1]
    error = math.atan2(math.sin(math.atan2(dy, dx)-yaw), math.cos(math.atan2(dy, dx)-yaw))
    distance = math.hypot(dx, dy)
    speed = min(0.45, distance)*max(0, math.cos(error)) if abs(error) < .5 else 0.0
    return [speed, 0.0, max(-0.8, min(0.8, error*1.8))]


def evaluate(scene, trace, events, reports):
    final = trace[-1] if trace else {}
    if scene == "drawer":
        closures = [e for e in events if e.get("type") == "drawer_closed"]
        effective_closure = bool(closures and closures[0].get("before_open_m", 0) >= .15)
        reopened = bool(closures and any(r["wall_time"] > closures[0]["wall_time"] and r.get("drawer_open_m", 0) >= 0.15 for r in trace))
        recovery_actions = [e for e in events if closures and e.get("type") == "skill_start" and e["wall_time"] > closures[0]["wall_time"]]
        recovery_time = min((e["wall_time"] for e in recovery_actions), default=None)
        waiting = [r for r in trace if recovery_time and closures[0]["wall_time"] < r["wall_time"] <= recovery_time]
        closed_before_action = bool(waiting and waiting[-1].get("drawer_open_m", 1) <= .02)
        reopened_after_action = bool(recovery_time and any(r["wall_time"] > recovery_time and r.get("drawer_open_m", 0) >= .15 for r in trace))
        opening_met = final.get("drawer_open_m", 0) >= 0.15
        quality_available = bool(trace and all("arm_joint_speed_max_rad_s" in r and "drawer_bottom_open_m" in r for r in trace))
        bottom_max = max((max(abs(r.get("drawer_bottom_open_m", 0)), r.get("physics_peak_bottom_displacement_m", 0)) for r in trace), default=0) if quality_available else None
        doors_max = max([r.get("physics_peak_other_doors_rad", 0) for r in trace]+[abs(v) for r in trace for k, v in r.get("cabinet_joints", {}).items() if k.startswith("door_")], default=0) if quality_available else None
        arm_max = max((max(r.get("arm_joint_speed_max_rad_s", 0), r.get("physics_peak_arm_speed_rad_s", 0)) for r in trace), default=0) if quality_available else None
        terminal = [r for r in trace if r.get("simulation_time", 0) >= final.get("simulation_time", 0)-1.5]
        stable = quality_available and len(terminal) >= 10 and all(r.get("arm_joint_speed_max_rad_s", 999) < .15 and r.get("tcp_speed_m_s", 999) < .03 for r in terminal)
        released = bool(quality_available and final.get("finger_positions") and min(final["finger_positions"]) > .025)
        grasp_checked = any(r.get("grasp_confirmed") for r in trace)
        skill_completed = any(e.get("type") == "skill_end" and e.get("status") == "completed" for e in events)
        quality_passed = bool(quality_available and bottom_max <= .01 and doors_max <= .02 and arm_max <= 1.0 and stable and released and grasp_checked and skill_completed)
        success = opening_met and quality_passed
        return {"evaluation_only": True, "final_open_m": final.get("drawer_open_m"),
                "opening_threshold_met": opening_met, "motion_quality_available": quality_available,
                "motion_quality_passed": quality_passed, "bottom_drawer_max_displacement_m": bottom_max,
                "other_doors_max_angle_rad": doors_max,
                "arm_peak_speed_rad_s": arm_max, "final_arm_stable": stable, "gripper_released": released,
                "grasp_contact_confirmed": grasp_checked,
                "initial_open_observed": any(r.get("drawer_open_m", 0) >= 0.15 and (not closures or r["wall_time"] < closures[0]["wall_time"]) for r in trace),
                "closure_injected": bool(closures), "reopened_after_closure": reopened,
                "effective_closure": effective_closure,
                "closed_before_recovery_action": closed_before_action,
                "reopened_after_recovery_action": reopened_after_action,
                "task_success": success, "recovery_success": effective_closure and closed_before_action and reopened_after_action and success,
                "final_model_claim": reports[-1].get("status") if reports else None,
                "false_completion_claim": bool(reports and reports[-1].get("status") == "goal_complete" and (not opening_met or quality_available and not quality_passed)),
                "criterion": "Top >=15cm AND bottom displacement<=1cm AND arm speed<=1rad/s AND bilateral grasp evidence AND released gripper AND completed skill AND stable last1.5s. Local controller uses calibrated state; ER2 uses RGB."}
    position = final.get("position", [0, 0, 0])
    arrived = math.hypot(position[0]-STATION_WAYPOINT[0], position[1]-STATION_WAYPOINT[1]) <= 0.3
    colors = [r.get("indicator_color", "").lower() for r in reports if r.get("status") == "goal_complete"]
    fell = any(r.get("position", [0, 0, 1])[2] < 0.2 for r in trace)
    barriers = [e for e in events if e.get("type") == "barrier_inserted"]
    rerouted = bool(barriers and any(e.get("type") == "skill_start" and e.get("route") == "right" and e["wall_time"] > barriers[0]["wall_time"] for e in events))
    success = arrived and not fell and bool(colors and colors[-1] == "red")
    return {"evaluation_only": True, "arrived": arrived, "final_position": position,
            "barrier_injected": any(e.get("type") == "barrier_inserted" for e in events),
            "local_obstacle_stop": any(e.get("type") == "obstacle_stop" for e in events),
            "fell": fell, "reported_colors": colors, "indicator_correct": bool(colors and colors[-1] == "red"),
            "task_success": success, "rerouted_after_barrier": rerouted, "recovery_success": rerouted and success,
            "criterion": "Within 0.3m of station waypoint, no base-height fall, final visual report red. No general navigation claim."}
