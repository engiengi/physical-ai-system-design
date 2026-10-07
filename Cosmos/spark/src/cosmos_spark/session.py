"""Run a visible Isaac Sim experiment and record only its X11 window."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from .artifacts import manifest, write_json

ROOT = Path(__file__).resolve().parents[2]


def simulator_windows():
    tree = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True)
    windows = set()
    for line in tree.splitlines():
        match = re.search(r'(0x[0-9a-f]+) "([^"]+)".*? (\d+)x(\d+)[+-]', line)
        if match and ("Isaac" in match[2] or "isaac" in match[2]) and int(match[3]) > 500:
            details = subprocess.run(["xwininfo", "-id", match[1]], capture_output=True, text=True)
            if "Map State: IsViewable" in details.stdout:
                windows.add(match[1])
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--entrypoint", choices=["recorded", "official"], default="recorded")
    args, runner_args = parser.parse_known_args()
    if "--headless" in runner_args:
        parser.error("This command requires a visible GUI; omit --headless")
    if not os.environ.get("DISPLAY"):
        parser.error("DISPLAY is required for GUI recording")
    run_id = args.run_id or (datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z") + "_gui_" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9_+.-]+", run_id) or run_id in (".", ".."):
        parser.error("Invalid run ID")
    if (ROOT / "runs" / run_id).exists():
        parser.error("Run ID already exists")
    output = ROOT / "captures" / run_id
    output.mkdir(parents=True, exist_ok=False)
    prior = simulator_windows()
    if prior:
        parser.error("Close the existing Isaac Sim window before starting a recorded session")
    command = [sys.executable, "-u", "-m", "cosmos_spark.runner", "--run-id", run_id, *runner_args]
    if args.entrypoint == "official":
        command = [sys.executable, "-u", str(ROOT / "scripts/official_reference_client.py"), "--run-id", run_id, *runner_args]
    record = {"run_id": run_id, "command": command, "started_at": datetime.now().astimezone().isoformat(),
              "recording_kind": "X11 Isaac Sim window, wall-clock time, no audio",
              "simulation_videos": str(ROOT / "runs" / run_id), "display": os.environ["DISPLAY"]}
    deployment = os.environ.get("COSMOS_DEPLOYMENT_MANIFEST")
    if deployment:
        record["deployment"] = json.loads(Path(deployment).read_text())
    write_json(output / "session.json", record)
    recorder = None
    process = None
    try:
        with (output / "simulator.log").open("w") as sim_log, (output / "recording.log").open("w") as rec_log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=sim_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            deadline = time.monotonic() + 120
            while process.poll() is None and time.monotonic() < deadline:
                found = simulator_windows() - prior
                if found:
                    window = sorted(found)[0]
                    record.update(window_id=window, recording_started_at=datetime.now().astimezone().isoformat())
                    # Matroska remains readable if the simulator window closes during capture.
                    rec_command = ["ffmpeg", "-hide_banner", "-nostdin", "-f", "x11grab",
                                   "-window_id", str(int(window, 16)), "-framerate", "15", "-draw_mouse", "0",
                                   "-i", os.environ["DISPLAY"], "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                                   "-threads", "4", "-pix_fmt", "yuv420p", str(output / "gui_wallclock.mkv")]
                    record["recording_command"] = rec_command
                    recorder = subprocess.Popen(rec_command, stdout=rec_log, stderr=subprocess.STDOUT)
                    write_json(output / "session.json", record)
                    print(f"GUI recording started: {output}", flush=True)
                    break
                time.sleep(0.5)
            if recorder is None:
                raise RuntimeError("Isaac Sim window was not detected; see simulator.log")
            while process.poll() is None:
                if recorder.poll() is not None:
                    record["recording_ended_before_simulator"] = True
                    if not (ROOT / "runs" / run_id / "run_status.json").exists():
                        record["recording_stopped_before_run_status"] = True
                time.sleep(0.5)
            record["simulator_exit_code"] = process.returncode
    except BaseException as exc:
        record["session_error"] = str(exc)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        if recorder is not None and recorder.poll() is None:
            recorder.send_signal(signal.SIGINT)
            try:
                recorder.wait(timeout=15)
            except subprocess.TimeoutExpired:
                recorder.kill()
                recorder.wait()
        if recorder is not None:
            record["recorder_exit_code"] = recorder.returncode
        source = output / "gui_wallclock.mkv"
        if source.exists() and source.stat().st_size > 1024:
            converted = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                                        "-i", str(source), "-c", "copy", "-movflags", "+faststart",
                                        str(output / "gui_wallclock.mp4")], capture_output=True, text=True)
            record["mp4_remux_exit_code"] = converted.returncode
            if converted.returncode:
                record["mp4_remux_error"] = converted.stderr[-2000:]
            else:
                probed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                         "-show_entries", "stream=width,height,nb_frames:format=duration",
                                         "-of", "json", str(output / "gui_wallclock.mp4")],
                                        capture_output=True, text=True)
                if probed.returncode == 0:
                    record["recording_probe"] = json.loads(probed.stdout)
                    record["recording_valid"] = bool(record["recording_probe"].get("streams"))
        record["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(output / "session.json", record)
        manifest(output)
        print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(record.get("simulator_exit_code", 1) or
                     int(not record.get("recording_valid") or bool(record.get("recording_stopped_before_run_status"))))


if __name__ == "__main__":
    main()
