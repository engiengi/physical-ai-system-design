"""Pure cooperative task protocol. No scene truth is used to choose model actions."""
import math

ROBOTS = ("A", "B")
TERMINAL = {"sequence_completed", "arrived", "blocked", "cancelled", "timeout", "fallen"}
TRAYS = {"A": (.46, -.72), "B": (.46, .72)}


def finite_vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"Expected {size} coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value):
        raise ValueError("Coordinates must be finite numbers")
    return list(map(float, value))


def validate_job(scene, robot, action, args):
    if scene=='transport' and robot=='Spot' and action=='navigate':
        if set(args)!={'destination','route'} or args['destination'] not in ('A','B') or args['route'] not in ('direct','detour'):
            raise ValueError('Need destination A/B and route direct/detour')
        return
    if robot not in ROBOTS:
        raise ValueError("robot_id must be A or B")
    if scene in ("dual_franka", "transport") and action == "pick_place":
        if set(args) != {"pick", "place", "observation_id"}:
            raise ValueError("Need pick, place and observation_id")
        for key in ("pick", "place"):
            v = finite_vector(args[key], 2)
            if any(not 0 <= x <= 1000 for x in v):
                raise ValueError("Coordinates are [y,x] normalized 0..1000")
        obs = args["observation_id"]
        if not isinstance(obs, str) or not obs or any(c not in '0123456789abcdef_' for c in obs):
            raise ValueError("Invalid observation_id")
    else:
        raise ValueError("Action unavailable for this scene")


def public_job(job):
    # Explicit whitelist: never return simulator object poses / evaluation labels.
    return {k: job[k] for k in ("job_id", "robot_id", "action", "arguments", "status",
                                "started_wall", "ended_wall") if k in job}


def inside_tray(position, robot):
    x, y = TRAYS[robot]
    return abs(position[0]-x) < .14 and abs(position[1]-y) < .09 and .015 < position[2] < .10


def evaluate_arm(objects):
    counts = {r: {"red": 0, "blue": 0} for r in ROBOTS}
    for name, pos in objects.items():
        for r in ROBOTS:
            if inside_tray(pos, r):
                counts[r][name.split('_')[0]] += 1
    satisfied = sum(min(1, n) for group in counts.values() for n in group.values())
    return {"tray_counts": counts, "satisfied_requirements": satisfied,
            "progress_fraction": satisfied/4, "physical_task_success": all(n == 1 for g in counts.values() for n in g.values())}


def overlap_seconds(jobs):
    total = 0.
    for a in jobs:
        for b in jobs:
            if a.get('robot_id') == 'A' and b.get('robot_id') == 'B' and 'ended_wall' in a and 'ended_wall' in b:
                total += max(0., min(a['ended_wall'], b['ended_wall'])-max(a['started_wall'], b['started_wall']))
    return total


def progress_bracket(fraction):
    return ("0-20", "20-40", "40-60", "60-80", "80-100")[min(4, int(max(0., fraction)*5))]
