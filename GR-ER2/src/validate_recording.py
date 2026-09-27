"""Offline integrity checks for streaming recordings; no API/GPU needed."""
import argparse
import json
from pathlib import Path
import subprocess
from common import read_json, write_json
from presentation import jsonl


def validate(out):
    out = Path(out)
    checks = {}
    clocks = jsonl(out / "frame_timestamps.jsonl")
    record = read_json(out / "recording.json")
    checks["frame_clock_count"] = len(clocks) == record["frames"] > 0
    checks["ordered_wall_clock"] = all(a["wall_time"] <= b["wall_time"] for a, b in zip(clocks, clocks[1:]))
    checks["sequential_frame_ids"] = [x["frame"] for x in clocks] == list(range(len(clocks)))
    checks["simulation_sampling_10fps"] = all(abs(b["simulation_time"]-a["simulation_time"]-.1) < .001 for a, b in zip(clocks, clocks[1:]))
    for name in ("input_frames", "side_frames"):
        checks[name] = len(list((out / name).glob("*.jpg"))) == len(clocks)
    frames = [read_json(p) for p in sorted((out / "sent_frames").glob("*.json"))]
    checks["jpeg_rate_at_most_one_fps"] = all(b["sent_wall_time"] - a["sent_wall_time"] >= 1 for a, b in zip(frames, frames[1:]))
    checks["all_transmitted_jpegs_saved"] = len(frames) > 0 and all((out / "sent_frames" / f"{f['frame']:06d}.jpg").is_file() for f in frames)
    videos = {}
    for name in ("input_video/camera.mp4", "rollout_video/side.mp4", "presentation/synchronized.mp4"):
        path = out / name
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,nb_frames,duration", "-of", "json", str(path)], check=True, capture_output=True, text=True)
        stream = json.loads(result.stdout)["streams"][0]
        expected = read_json(out / "presentation/manifest.json")["frames"] if name.startswith("presentation/") else len(clocks)
        checks[name] = int(stream["nb_frames"]) == expected and float(stream["duration"]) > 0
        videos[name] = stream
    scenario = read_json(out / "protocol.json")["scenario"]
    events = [read_json(p) for p in sorted((out / "events").glob("*.json"))]
    if scenario in ("drawer", "patrol"):
        interpretation = read_json(out / "evaluation.json")
    else:
        from live_recovery import evaluate_stream, drop_confirmed
        metrics = read_json(out / "recovery_metrics.json")
        assessment = evaluate_stream(None, None, False, {}, metrics, False)
        interpretation = {"red_already_inside_before_recovery_action": assessment["red_already_inside_before_recovery_action"]}
        if scenario == "grasp_drop":
            interpretation["physical_drop_confirmed"] = drop_confirmed(jsonl(out / "evaluation_trace.jsonl"), events[0] if events else None)
    report = {"passed": all(checks.values()), "checks": checks, "videos": videos, "interpretation": interpretation}
    write_json(out / "validation.json", report)
    if not report["passed"]:
        raise ValueError(report)
    print(str(out) + ": recording checks passed", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="+", type=Path)
    for directory in parser.parse_args().run:
        validate(directory)
