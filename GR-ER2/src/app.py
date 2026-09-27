"""Local-only GUI, Gemini comparison, and bounded simulator tool loop."""
import argparse
import base64
import hashlib
import getpass
import importlib.metadata
import platform
import io
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image, ImageDraw
from common import ROOT, parse_json, pixel, point, read_json, safe_child, uid, validate_action, write_json

CONFIG = read_json(ROOT / "configs/default.json")
TASKS = {
    "point": 'Find the objects requested below. Return only a JSON array of {"label": "name", "point": [y,x]}.',
    "box": 'Find the requested objects. Return only a JSON array of {"label": "name", "box": [ymin,xmin,ymax,xmax]}.',
    "relation": 'Resolve the spatial instruction below. Return only a JSON array of {"label": "selected object", "point": [y,x]}.',
    "plan": 'Describe an ordered robot task plan for the instruction below in Korean. State uncertainties. Do not claim to have executed actions.',
    "trajectory": 'Return a conceptual IMAGE-SPACE path for the instruction below as JSON: {"label":"path", "points":[[y,x],...]}. This is a 2D illustration, not an executable robot path.'
}
COORDS = " All image coordinates are normalized to 0..1000 in [y,x] order. Use the supplied image only."
LOCK = threading.Lock()
JOBS = {}


def load_env():
    # Deliberately no shell evaluation. Secrets are never returned to the browser.
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def redact(message):
    text = str(message)
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.getenv(key)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def bridge(command, **kwargs):
    status_path = ROOT / "runtime/status.json"
    if not status_path.exists() or time.time() - status_path.stat().st_mtime > 30:
        raise RuntimeError("시뮬레이터가 준비되지 않았습니다. scripts/sim.sh start와 logs를 확인하세요.")
    request_id = uid()
    req = {"id": request_id, "command": command, "created": time.time(), **kwargs}
    write_json(ROOT / "runtime/queue" / (request_id + ".json"), req)
    result_path = ROOT / "runtime/replies" / (request_id + ".json")
    deadline = time.monotonic() + CONFIG["simulation_timeout_seconds"]
    while time.monotonic() < deadline:
        if result_path.exists():
            result = read_json(result_path)
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result
        time.sleep(0.2)
    raise TimeoutError("시뮬레이터 응답 시간 초과. 자동 재실행하지 않습니다. 화면과 로그를 확인하세요.")


def run_dir(group):
    path = ROOT / "outputs" / group / uid()
    path.mkdir(parents=True)
    path.chmod(0o2775)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unavailable"
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                      text=True, timeout=5).strip()
    except Exception:
        gpu = "unavailable"
    write_json(path / "metadata.json", {"created": time.time(), "config": CONFIG, "git_commit": commit,
               "environment": {"python": platform.python_version(), "architecture": platform.machine(), "gpu": gpu,
                               "google-genai": importlib.metadata.version("google-genai")},
               "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "src").glob("*.py")}})
    return path


def input_image(relative):
    path = safe_child(ROOT, relative)
    if not any(path.is_relative_to(ROOT / p) for p in ("data", "outputs")):
        raise ValueError("data/ 또는 outputs/의 이미지를 선택하세요.")
    with Image.open(path) as im:
        im.verify()
    return path


def response_text(response):
    text = getattr(response, "output_text", None)
    if text:
        return text
    items = getattr(response, "outputs", None) or getattr(response, "steps", None) or []
    return "\n".join(getattr(item, "text", "") or "" for item in items if getattr(item, "type", "") == "text")


def client():
    load_env()
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError("GEMINI_API_KEY가 없습니다. 실행방법.md 2절을 따라 실습 호스트의 GR-ER2/.env에 등록하세요.")
    from google import genai
    return genai.Client(http_options={"timeout": CONFIG["api_timeout_ms"]})


def image_part(path):
    # Inline input avoids creating persistent Files API objects.
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
    return {"type": "image", "data": base64.b64encode(buf.getvalue()).decode(), "mime_type": "image/png"}


def infer(api, model, image, prompt, directory, tools=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    started = time.monotonic()
    kwargs = {"model": model, "input": [image_part(image), {"type": "text", "text": prompt}],
              "generation_config": {"thinking_level": CONFIG["thinking_level"],
                                    "max_output_tokens": CONFIG["max_output_tokens"]}}
    if tools:
        kwargs["tools"] = tools
    try:
        response = api.interactions.create(**kwargs)
        write_json(directory / "response.json", response.model_dump(mode="json"))
        text = response_text(response)
        (directory / "response.txt").write_text(text, encoding="utf-8")
        write_json(directory / "timing.json", {"model": model, "elapsed_seconds": time.monotonic() - started})
        return response, text
    except Exception as exc:
        write_json(directory / "error.json", {"model": model, "error": redact(exc), "elapsed_seconds": time.monotonic() - started})
        raise


def overlay(source, parsed, target):
    im = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(im)
    entries = parsed if isinstance(parsed, list) else [parsed]
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("공간 출력 항목은 JSON object여야 합니다.")
        color = "#ffdc00"
        if "point" in entry:
            x, y = pixel(entry["point"], *im.size)
            draw.ellipse((x-7, y-7, x+7, y+7), outline=color, width=3)
            draw.text((x+9, y), entry.get("label", "point"), fill=color)
        if "box" in entry:
            box = entry["box"]
            if len(box) != 4:
                raise ValueError("box는 [ymin,xmin,ymax,xmax]입니다.")
            p1, p2 = pixel(box[:2], *im.size), pixel(box[2:], *im.size)
            if p1[0] > p2[0] or p1[1] > p2[1]:
                raise ValueError("Bounding Box 순서가 잘못되었습니다.")
            draw.rectangle((*p1, *p2), outline=color, width=3)
            draw.text(p1, entry.get("label", "box"), fill=color)
        if "points" in entry:
            points = [pixel(p, *im.size) for p in entry["points"]]
            if len(points) < 2:
                raise ValueError("경로에는 두 점 이상이 필요합니다.")
            draw.line(points, fill=color, width=4)
            for i, (x, y) in enumerate(points):
                draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
                draw.text((x+5, y+5), str(i), fill=color)
    im.save(target)


def compare(payload):
    task = payload.get("task", "point")
    if task not in TASKS:
        raise ValueError("알 수 없는 task입니다.")
    models = payload.get("models", CONFIG["models"])
    if not models or any(m not in CONFIG["models"] for m in models):
        raise ValueError("configs/default.json에 등록한 모델을 선택하세요.")
    source = input_image(payload["image"])
    out = run_dir("01_model_comparison")
    (out / "input_images").mkdir()
    snapshot = out / "input_images/input.png"
    Image.open(source).convert("RGB").save(snapshot)
    prompt = TASKS[task] + COORDS + "\nInstruction: " + payload.get("instruction", "Find all colored blocks and the tray.")
    write_json(out / "request.json", {**payload, "prompt": prompt, "image_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()})
    api = client()
    results = []
    # Same exact immutable image/prompt. Run sequentially to avoid surprise quota bursts.
    for index, model in enumerate(models):
        directory = out / f"model_{index+1}"
        result = {"model": model}
        try:
            _, text = infer(api, model, snapshot, prompt, directory)
            result["text"] = text
            if task != "plan":
                try:
                    parsed = parse_json(text)
                    write_json(directory / "parsed.json", parsed)
                    overlay(snapshot, parsed, directory / "overlay.png")
                    result["overlay"] = str((directory / "overlay.png").relative_to(ROOT))
                except Exception as exc:
                    result["parse_error"] = str(exc)
        except Exception as exc:
            result["error"] = redact(exc)
        results.append(result)
    write_json(out / "summary.json", results)
    return {"output": str(out.relative_to(ROOT)), "results": results}


def encode_videos(out):
    for frames, dest in (("input_frames", "input_video/camera.mp4"), ("side_frames", "rollout_video/side.mp4")):
        src = out / frames
        if not src.exists() or not list(src.glob("*.jpg")):
            continue
        target = out / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10",
                        "-i", str(src / "%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)], check=True)


def sim_action(payload):
    command = payload["command"]
    if command == "pick_place":
        validate_action(payload["action"])
        out = run_dir("02_isaac_sim")
        write_json(out / "request.json", payload)
        try:
            extra = {}
            if payload.get("perturb_first_action"):
                extra["perturb_after_ticks"] = 60
                extra["scenario"] = payload.get("scenario", "target_shift")
            result = bridge(command, action=payload["action"], output=str(out.relative_to(ROOT)), **extra)
            encode_videos(out)
        except Exception as exc:
            write_json(out / "error.json", {"error": redact(exc)})
            raise
        return {**result, "output": str(out.relative_to(ROOT))}
    if command not in ("snapshot", "reset", "shift", "stop", "sample"):
        raise ValueError("허용되지 않은 시뮬레이터 명령입니다.")
    kwargs = {}
    if command in ("reset", "sample"):
        kwargs["scene"] = int(payload.get("scene", 0))
        if kwargs["scene"] not in (0, 1, 2):
            raise ValueError("scene은 0~2입니다.")
    return bridge(command, **kwargs)


TOOL = {"type": "function", "name": "pick_place", "description": "Physically pick a block at pick and place it at place. Coordinates normalized [y,x]. One call per observation. Both pixels refer to the current camera image.",
        "parameters": {"type": "object", "properties": {
            "pick": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "place": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "observation_id": {"type": "string"}}, "required": ["pick", "place", "observation_id"], "additionalProperties": False}}


def agent(payload, output=None):
    model = payload.get("model", CONFIG["models"][0])
    if model not in CONFIG["models"]:
        raise ValueError("설정에 등록한 모델을 사용하세요.")
    steps = int(payload.get("steps", 1))
    if not 1 <= steps <= CONFIG["max_agent_steps"]:
        raise ValueError("허용된 최대 실행 횟수를 확인하세요.")
    out = output or run_dir("02_isaac_sim_agent")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "request.json", payload)
    api, history = client(), []
    for index in range(steps):
        if (ROOT / "runtime/STOP").exists():
            history.append({"stopped_by_user": True})
            break
        obs = bridge("snapshot")
        directory = out / f"step_{index+1:02d}"
        directory.mkdir()
        src = ROOT / obs["image"]
        shutil.copy2(src, directory / "input.png")
        prompt = ("You control a simulated Franka via pick_place. Select one block and a clear placement point inside the green tray. "
                  "Only visible RGB and controller status are available. Never claim success without checking the new image. "
                  "Issue at most ONE function call for this observation, or explain in Korean why finished/unable. "
                  "Do not use world coordinates. " + COORDS + f"\nObservation ID: {obs['observation_id']}\n"
                  + "Goal: " + payload.get("instruction", "Put the red block into the green tray.")
                  + "\nPrevious tool outcomes (controller completion is not task success): " + json.dumps(history, ensure_ascii=False))
        response, text = infer(api, model, directory / "input.png", prompt, directory, [TOOL])
        items = getattr(response, "outputs", None) or getattr(response, "steps", None) or []
        calls = [c for c in items if getattr(c, "type", "") == "function_call"]
        if not calls:
            history.append({"model_message": text, "no_action": True})
            break
        if len(calls) != 1 or calls[0].name != "pick_place":
            raise ValueError("현재 관측에서는 pick_place 한 번만 허용합니다.")
        args = calls[0].arguments
        if isinstance(args, str):
            args = parse_json(args)
        validate_action(args)
        if args["observation_id"] != obs["observation_id"]:
            raise ValueError("모델이 다른 observation_id를 반환했습니다.")
        if (ROOT / "runtime/STOP").exists():
            break
        write_json(directory / "action.json", args)
        overlay(directory / "input.png", [{"label": "pick", "point": args["pick"]},
                                           {"label": "place", "point": args["place"]}], directory / "overlay.png")
        result = sim_action({"command": "pick_place", "action": args,
                             "perturb_first_action": bool(payload.get("perturb_first_action")) and index == 0})
        # Never expose privileged evaluator labels or positions to the model.
        history.append({"action": args, "controller_status": result["controller_status"], "output": result["output"]})
        write_json(directory / "tool_result.json", result)
        write_json(out / "history.json", history)
    # Preserve final observation even when step budget is exhausted.
    final = bridge("snapshot")
    shutil.copy2(ROOT / final["image"], out / "final.png")
    write_json(out / "history.json", history)
    return {"output": str(out.relative_to(ROOT)), "history": history,
            "note": "동작 횟수 제한에 도달해도 성공을 뜻하지 않습니다. 실제 영상과 metrics.json을 확인하세요."}


def dispatch(action, payload):
    status_file = ROOT / "runtime/status.json"
    scene = read_json(status_file).get("world_scenario", "blocks") if status_file.exists() else "blocks"
    if scene in ("drawer", "patrol") and (action in ("agent", "recovery", "stream_recovery") or
            action == "sim" and payload.get("command") in ("pick_place", "shift")):
        raise ValueError("현재 환경에서는 '다른 환경'의 실행 버튼을 사용하세요. 블록 전용 명령이 차단되었습니다.")
    if action == "compare":
        return compare(payload)
    if action == "sim":
        return sim_action(payload)
    if action == "agent":
        if (ROOT / "runtime/STOP").exists():
            raise ValueError("정지 상태입니다. 먼저 장면을 초기화하세요.")
        return agent(payload)
    if action == "recovery":
        from recovery import run_recovery
        return run_recovery()
    if action == "stream_recovery":
        import asyncio
        from live_recovery import experiment
        return asyncio.run(experiment(scenario=payload.get("scenario", "target_shift")))
    if action == "environment":
        import asyncio
        from scenario_run import experiment
        scene = payload.get("scene")
        if scene not in ("drawer", "patrol"):
            raise ValueError("Unknown environment")
        return asyncio.run(experiment(scene, perturb=bool(payload.get("perturb", True))))
    raise ValueError("알 수 없는 실행입니다.")


def start_job(action, payload):
    exclusive = not (action == "sim" and payload.get("command") == "shift")
    if exclusive and not LOCK.acquire(blocking=False):
        raise ValueError("현재 실행이 진행 중입니다. 완료 후 다시 실행하세요.")
    job = uid()
    JOBS[job] = {"status": "running"}
    def work():
        try:
            JOBS[job] = {"status": "done", "result": dispatch(action, payload)}
        except Exception as exc:
            JOBS[job] = {"status": "error", "error": redact(exc)}
        finally:
            if exclusive:
                LOCK.release()
    threading.Thread(target=work, daemon=True).start()
    return job


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send(self, data, kind="application/json", status=200):
        if kind == "application/json":
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/":
                return self.send((ROOT / "src/index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                state = read_json(ROOT / "runtime/status.json") if (ROOT / "runtime/status.json").exists() else {"ready": False}
                if state.get("updated", 0) < time.time() - 30:
                    state = {"ready": False, "note": "시뮬레이터 시작/로그를 확인하세요."}
                load_env()
                return self.send({"sim": state, "models": CONFIG["models"], "api_key_present": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
                                  "samples": [str(p.relative_to(ROOT)) for p in sorted((ROOT / "data/samples").glob("*.png"))]})
            if path == "/api/runs":
                records = []
                for group in ("01_model_comparison", "02_isaac_sim", "02_isaac_sim_agent", "03_perturbation_recovery", "04_streaming_recovery", "05_drawer", "06_patrol"):
                    parent = ROOT / "outputs" / group
                    if not parent.exists():
                        continue
                    for directory in sorted((p for p in parent.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)[:8]:
                        files = [str(p.relative_to(ROOT)) for p in directory.rglob("*") if p.is_file() and
                                 p.suffix in (".mp4", ".png", ".json", ".txt") and p.parent.name not in ("input_frames", "side_frames")]
                        records.append({"run": str(directory.relative_to(ROOT)), "files": files})
                return self.send(records)
            if path.startswith("/api/jobs/"):
                return self.send(JOBS.get(path.rsplit("/", 1)[-1], {"status": "unknown"}))
            if path == "/camera.jpg":
                return self.send((ROOT / "runtime/camera.jpg").read_bytes(), "image/jpeg")
            if path.startswith("/files/"):
                file = safe_child(ROOT, path[7:])
                if not any(file.is_relative_to(ROOT / p) for p in ("data", "outputs")) or file.suffix not in (".png", ".jpg", ".mp4", ".json", ".txt"):
                    raise ValueError("허용되지 않은 파일입니다.")
                return self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            return self.send({"error": "not found"}, status=404)
        except Exception as exc:
            self.send({"error": redact(exc)}, status=400)

    def do_POST(self):
        try:
            # Local-only service: reject browser cross-origin writes (DNS/CSRF).
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                raise ValueError("다른 사이트에서의 요청을 허용하지 않습니다.")
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError("요청 크기는 16 MB 이하여야 합니다.")
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/stop":
                (ROOT / "runtime").mkdir(exist_ok=True)
                (ROOT / "runtime/STOP").touch()
                return self.send({"stopping": True})
            if self.path == "/api/upload":
                raw = base64.b64decode(payload["data"], validate=True)
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width * im.height > 20_000_000:
                        raise ValueError("이미지는 20 MP 이하로 준비하세요.")
                    dst = ROOT / "data/uploads" / (uid() + ".png")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    im.convert("RGB").save(dst)
                return self.send({"image": str(dst.relative_to(ROOT))})
            action = self.path.removeprefix("/api/")
            return self.send({"job": start_job(action, payload)})
        except Exception as exc:
            self.send({"error": redact(exc)}, status=400)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    comp = sub.add_parser("compare")
    comp.add_argument("--image", required=True)
    comp.add_argument("--task", choices=TASKS, default="point")
    comp.add_argument("--instruction", default="Find all colored blocks and the green tray.")
    comp.add_argument("--model", action="append")
    sim = sub.add_parser("sim")
    sim.add_argument("action", choices=["snapshot", "reset", "sample", "shift", "stop"])
    sim.add_argument("--scene", type=int, default=0)
    sub.add_parser("check-models")
    key = sub.add_parser("set-key")
    key.add_argument("--stdin", action="store_true", help="Read a key from a secure pipe; never pass it as an argument")
    key.add_argument("--replace", action="store_true", help="Explicitly replace an existing project .env")
    args = parser.parse_args()
    load_env()
    if args.command == "serve":
        print(f"GUI: http://127.0.0.1:{args.port}", flush=True)
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    elif args.command == "compare":
        payload = {"image": args.image, "task": args.task, "instruction": args.instruction}
        if args.model:
            payload["models"] = args.model
        print(json.dumps(compare(payload), ensure_ascii=False, indent=2))
    elif args.command == "sim":
        print(json.dumps(sim_action({"command": args.action, "scene": args.scene}), ensure_ascii=False, indent=2))
    elif args.command == "set-key":
        import sys
        secret = sys.stdin.read().strip() if args.stdin else getpass.getpass("Gemini API key (hidden): ")
        if not secret or "\n" in secret or "\r" in secret:
            raise ValueError("유효한 한 줄 API 키를 입력하세요.")
        path = ROOT / ".env"
        flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if args.replace else os.O_EXCL)
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "w") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write("GEMINI_API_KEY=" + secret + "\n")
        print("프로젝트 .env에 권한 600으로 저장했습니다. 키는 출력하지 않습니다.")
    else:
        api = client()
        available = {m.name.removeprefix("models/") for m in api.models.list()}
        print(json.dumps({m: m in available for m in CONFIG["models"]}, indent=2))


if __name__ == "__main__":
    main()
