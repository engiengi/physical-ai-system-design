"""Causal, wall-clock synchronized recording replay; no extra model calls."""
import argparse
import bisect
import json
import math
from pathlib import Path
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont
from common import read_json, write_json


def latest_index(times, now):
    """Never show an input or response before its recorded arrival time."""
    return bisect.bisect_right(times, now) - 1


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def render(out):
    out = Path(out)
    clocks = jsonl(out / "frame_timestamps.jsonl")
    if not clocks:
        raise ValueError("Exact frame timestamps required; refusing approximate synchronization")
    sent = [read_json(p) for p in sorted((out / "sent_frames").glob("*.json"))]
    events = jsonl(out / "api_timeline.jsonl")
    control = {row["frame"]: row.get("controller_phase") for row in jsonl(out/"evaluation_trace.jsonl") if "controller_phase" in row}
    times = [c["wall_time"] for c in clocks]
    sent_times = [c["sent_wall_time"] for c in sent]
    event_times = [c["wall_time"] for c in events]
    start, end = times[0], times[-1]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 24)
    font = ImageFont.truetype(font_path, 18)
    small = ImageFont.truetype(font_path, 16)
    dest = out / "presentation"
    dest.mkdir(exist_ok=True)
    fps, width, height = 10, 1600, 1200
    count = math.ceil((end - start) * fps) + 1
    protocol = read_json(out / "protocol.json")
    result = read_json(out / "evaluation.json") if (out / "evaluation.json").exists() else {}
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-pix_fmt", "yuv420p", str(dest / "synchronized.mp4")]
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    cached = {}

    def picture(path):
        key = str(path)
        if key not in cached:
            with Image.open(path) as im:
                cached[key] = im.convert("RGB").resize((768, 576))
        if len(cached) > 8:
            cached.pop(next(iter(cached)))
        return cached[key]

    def block(draw, text, xy, max_lines, color="#e4edf6"):
        lines = []
        for paragraph in str(text).splitlines():
            lines += textwrap.wrap(paragraph, width=72, break_long_words=True) or [""]
        if len(lines) > max_lines:
            lines = lines[:max_lines-1] + ["[excerpt; full request/response saved with this run]"]
        for index, line in enumerate(lines):
            draw.text((xy[0], xy[1]+index*24), line, font=font, fill=color)

    # Selected stills retain the same causal timing as the video.
    markers = {0: "start"}
    for kind in ("report", "tool_call"):
        candidates = [e for e in events if e["kind"] == kind and
                      (kind == "tool_call" or any(v in e.get("text", "") for v in ("target_moved", "goal_moved", "grasp_failed", "goal_complete", '"status": "blocked"')))]
        if candidates:
            markers[min(count-1, max(0, math.ceil((candidates[0]["wall_time"]-start)*fps)))] = kind
    markers[count-1] = "final"
    try:
        for index in range(count):
            now = min(end, start + index / fps)
            ci, si, ei = latest_index(times, now), latest_index(sent_times, now), latest_index(event_times, now)
            visible = events[:ei+1]
            prompt = next((e for e in reversed(visible) if e["kind"] == "prompt"), None)
            output = next((e for e in reversed(visible) if e["kind"] in ("report", "tool_call", "tool_result")), None)
            phase = visible[-1]["phase"] if visible else "CONNECTING"
            canvas = Image.new("RGB", (width, height), "#111e30")
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 12), f"ER 2 Live | {protocol['scenario']} | Wall +{now-start:.1f}s | {phase}", font=title_font, fill="white")
            draw.text((20, 53), "ISAAC SIM / physical rollout (side camera)", font=font, fill="#82d9ff")
            draw.text((820, 53), "ACTUAL API INPUT / " + protocol.get("api_camera_label", "last JPEG sent (overhead camera)"), font=font, fill="#82d9ff")
            canvas.paste(picture(out / "side_frames" / f"{clocks[ci]['frame']:06d}.jpg"), (16, 82))
            if si >= 0:
                frame = sent[si]
                canvas.paste(picture(out / "sent_frames" / f"{frame['frame']:06d}.jpg"), (816, 82))
                draw.text((820, 665), f"Input #{frame['frame']} | sent +{sent_times[si]-start:.2f}s | age {now-sent_times[si]:.2f}s", font=small, fill="#b8cee3")
            else:
                draw.text((850, 300), "Waiting for first transmitted frame", font=font, fill="white")
            phase_note = " | control: "+control[clocks[ci]["frame"]] if control.get(clocks[ci]["frame"]) else ""
            draw.text((20, 665), f"Sim {clocks[ci]['simulation_time']:.2f}s | captured frame #{clocks[ci]['frame']}"+phase_note, font=small, fill="#b8cee3")
            draw.line((800, 710, 800, 1158), fill="#48617c", width=2)
            draw.text((20, 706), "API REQUEST / latest heartbeat", font=title_font, fill="#82d9ff")
            draw.text((820, 706), "API OUTPUT / visible response or tool call", font=title_font, fill="#85e6b0")
            block(draw, prompt["text"] if prompt else "Live connection is warming up. Task and tool definitions are saved in live_config.json.", (20, 750), 16)
            if output:
                label = f"{output['kind']} received +{output['wall_time']-start:.2f}s"
                block(draw, label + "\n" + output["text"], (820, 750), 16)
            else:
                block(draw, "Waiting for API response...", (820, 750), 16)
            draw.text((20, 1170), "Timestamp-synchronized replay | wall-clock 1x | JPEG <=1 FPS | displayed text = API response, not hidden reasoning", font=small, fill="#b8cee3")
            if index in markers:
                canvas.save(dest / (markers[index] + ".png"))
            pipe.stdin.write(canvas.tobytes())
        pipe.stdin.close()
        if pipe.wait() != 0:
            raise RuntimeError("ffmpeg presentation encoding failed")
    finally:
        if pipe.poll() is None:
            pipe.kill()
            pipe.wait()
    write_json(dest / "manifest.json", {"timebase": "wall_clock", "fps": fps, "frames": count,
        "seconds": count/fps, "source_start_wall_time": start, "source_end_wall_time": end,
        "causal_alignment": "latest source timestamp <= playback time; no future frames/responses",
        "rendering": "post-recording synchronized replay; source videos remain untouched",
        "scenario": protocol["scenario"], "evaluation": result})
    print("Presentation: " + str(dest / "synchronized.mp4"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
