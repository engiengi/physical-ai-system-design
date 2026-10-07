"""Create review media and measured summaries from a completed simulation run."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from .artifacts import manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-name", default="content")
    args = parser.parse_args()
    run = args.run.resolve()
    if not args.output_name.replace('_', '').isalnum():
        parser.error("Use an alphanumeric output folder name")
    dest = run / args.output_name
    dest.mkdir(exist_ok=False)
    episode = run / "episode_0000"
    result = json.loads((episode / "episode.json").read_text())
    requests = [json.loads(x) for x in (episode / "requests.jsonl").read_text().splitlines()]
    actions = [json.loads(x) for x in (episode / "applied_actions.jsonl").read_text().splitlines()]
    # Validate what was applied against the original server response and official postprocessing.
    for row in actions:
        raw = np.load(episode / f"{row['request_id']}_actions.npy")[row['chunk_index']].copy()
        raw[-1] = raw[-1] > 0.5
        np.testing.assert_array_equal(raw, np.asarray(row['action'], dtype=np.float32))
    transitions = [i for i, x in enumerate(actions) if i and x['action'][-1] != actions[i-1]['action'][-1]]
    selected = sorted(set([0, result['steps'], *[min(result['steps'], i + 1) for i in transitions[:6]]]))
    camera = cv2.VideoCapture(str(episode / "egocentric_mirrored_camera.mp4"))
    for frame in selected:
        camera.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = camera.read()
        if not ok:
            raise RuntimeError(f"Missing video frame {frame}")
        cv2.imwrite(str(dest / f"scene_frame_{frame:04d}.png"), image)
    camera.release()
    specs = [("egocentric_mirrored_camera.mp4", "SCENE"), ("wrist_cam.mp4", "WRIST"),
             ("over_shoulder_left_camera.mp4", "LEFT"), ("over_shoulder_right_camera.mp4", "RIGHT")]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-filter_complex_threads", "2"]
    filters = []
    for i, (filename, label) in enumerate(specs):
        command += ["-i", str(episode / filename)]
        filters.append(f"[{i}:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,drawtext=text='{label} | SIM TIME 15 FPS':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[v{i}]")
    filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|640_0|0_360|640_360:shortest=1[v]")
    command += ["-filter_complex", ";".join(filters), "-map", "[v]", "-an", "-c:v", "libx264", "-threads", "4",
                "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest / "multiview.mp4")]
    subprocess.run(command, check=True)
    rtts = [r['rtt_ms'] for r in requests if r['status'] == 'ok']
    infer = [r['server_timing']['server_timing']['infer_ms'] for r in requests
             if r.get('server_timing', {}).get('server_timing', {}).get('infer_ms') is not None]
    stats = lambda values: ({"count": len(values), "mean_ms": float(np.mean(values)),
                             "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))} if values else None)
    write_json(dest / "review.json", {"run_id": run.name, "episode": result, "request_rtt": stats(rtts),
        "thor_inference": stats(infer), "applied_actions_match_server": True,
        "gripper_command_transition_steps": transitions, "selected_video_frames": selected,
        "human_review": None, "note": "Simulator task result; human review pending. Gripper transitions do not prove a grasp. Camera MP4 time is simulation time; GUI recording is wall time."})
    manifest(dest)
    manifest(run)
    print(dest)


if __name__ == "__main__":
    main()
