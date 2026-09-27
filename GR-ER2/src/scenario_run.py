"""Bounded real Live API experiments in two distinct Isaac Sim environments."""
import argparse
import asyncio
import json
import shutil
import time
from common import ROOT, read_json, write_json, parse_json
from scenario_rules import evaluate, skill_request

TOOLS = {
    "drawer": [{"function_declarations": [{"name": "open_drawer", "behavior": "BLOCKING",
        "description": "Execute a calibrated physical top-drawer skill: safe approach, finger alignment, contact-checked grasp, slow pull, release, retreat and settling. It uses RMPFlow, not a learned action policy. It may report a contact/tracking failure. After it returns, inspect fresh video. No image coordinates are needed.",
        "parameters": {"type": "OBJECT", "properties": {}}}]}],
    "patrol": [{"function_declarations": [{"name": "navigate", "behavior": "BLOCKING",
        "description": "Walk to the inspection station using a calibrated route graph. center = shortest blue-marked corridor; right = longer yellow-marked corridor to the robot's initial right. The local collision guard can return blocked. After completion inspect the camera and report the indicator color.",
        "parameters": {"type": "OBJECT", "properties": {"route": {"type": "STRING", "enum": ["center", "right"]}}, "required": ["route"]}}]}],
}
TASKS = {
    "drawer": "Open the TOP drawer and keep it open. Observe whether it stays open; if it closes again, reopen it. Only call a skill when visually needed.",
    "patrol": "Walk to the inspection station and report its indicator color. Prefer the short center route if it is clear. If the center corridor becomes blocked, inspect the scene and use a clear alternative. Goal completion requires navigation to report arrived AND a fresh visual inspection of the indicator. Do not declare arrival or a color you cannot see.",
}


def ensure_scene(scene):
    state = read_json(ROOT/"runtime/status.json")
    if not state.get("ready") or state.get("world_scenario") != scene or time.time()-state.get("updated", 0) > 30:
        raise RuntimeError(f"Start the {scene} scene using GR_ER2_SCENE={scene} bash scripts/sim.sh start")


async def experiment(scene, perturb=True, smoke=False):
    from app import bridge, client, run_dir, encode_videos, redact
    from live_recovery import MODEL, jpeg
    from presentation import render, jsonl
    from google.genai import types
    ensure_scene(scene)
    out = run_dir("05_drawer" if scene == "drawer" else "06_patrol")
    state = {"phase": "OBSERVE", "latest": None, "sent": 0, "reports": [], "actions": [], "injected": False}
    write_json(out/"protocol.json", {"scenario": scene, "streaming_model": None if smoke else MODEL,
        "mode": "controller_smoke" if smoke else "Live_API", "perturb": perturb,
        "task": TASKS[scene], "max_live_seconds": 360, "max_turns": 18, "max_skill_calls": 4,
        "api_camera_label": "fixed oblique cabinet camera" if scene == "drawer" else "robot-following forward RGB camera",
        "physical_tools": "blocking; video continues streaming, next reasoning heartbeat follows tool completion",
        "limitations": "Drawer uses calibrated RMPFlow skill; Spot uses official walking policy. Low-level control uses privileged state. Patrol uses a known route graph and simulator localization, not SLAM."})
    print("Scenario output: " + str(out), flush=True)

    def timeline(kind, **data):
        with (out/"api_timeline.jsonl").open("a") as handle:
            handle.write(json.dumps({"wall_time": time.time(), "kind": kind, "phase": state["phase"], **data})+"\n")

    await asyncio.to_thread(bridge, "reset")
    await asyncio.sleep(2)
    feeder = None
    failure = None
    await asyncio.to_thread(bridge, "record_start", output=str(out.relative_to(ROOT)), perturb=perturb)
    try:
        if smoke:
            params = {"skill": "open_drawer"} if scene == "drawer" else {"skill": "navigate", "route": "center"}
            timeline("tool_call", text=json.dumps(params))
            result = await asyncio.to_thread(bridge, "skill", **params)
            timeline("tool_result", text=json.dumps(result))
            write_json(out/"manual_result.json", result)
            if perturb and scene == "drawer":
                await asyncio.to_thread(bridge, "perturb")
                await asyncio.sleep(2)
                second = await asyncio.to_thread(bridge, "skill", **params)
                write_json(out/"manual_recovery.json", second)
        else:
            schema = '{"status":"working|goal_complete|blocked|uncertain","reason":"visible evidence"}' if scene == "drawer" else '{"status":"working|goal_complete|blocked|uncertain","reason":"visible evidence","indicator_color":"red|green|unknown"}'
            config = {"response_modalities": ["TEXT"], "tools": TOOLS[scene], "system_instruction":
                "You are a robot task supervisor. Use actual camera video for perception. "+TASKS[scene]+
                ' Tools control a real physics simulation. Call at most ONE physical tool per heartbeat. After its result, end your turn with a brief acknowledgment and wait for a fresh heartbeat. When no tool is needed return plain JSON '+schema+'. Never invent simulator state. Videos alone do not require replies; respond to heartbeats.'}
            write_json(out/"live_config.json", config)
            async with client().aio.live.connect(model=MODEL, config=config) as session:
                async def feed():
                    while True:
                        obs = await asyncio.to_thread(bridge, "snapshot")
                        index = state["sent"]
                        directory = out/"sent_frames"
                        directory.mkdir(exist_ok=True)
                        data = jpeg(ROOT/obs["image"])
                        (directory/f"{index:06d}.jpg").write_bytes(data)
                        await session.send_realtime_input(video=types.Blob(data=data, mime_type="image/jpeg"))
                        meta = {**obs, "frame": index, "sent_wall_time": time.time(), "phase": state["phase"]}
                        write_json(directory/f"{index:06d}.json", meta)
                        state["latest"], state["sent"] = meta, index+1
                        await asyncio.sleep(1.1)
                feeder = asyncio.create_task(feed())
                deadline = time.monotonic()+360
                while state["sent"] < 3:
                    if feeder.done():
                        feeder.result()
                    if time.monotonic() > deadline:
                        raise TimeoutError("No input frames")
                    await asyncio.sleep(.2)
                for turn in range(18):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Live experiment budget exhausted")
                    if feeder.done():
                        feeder.result()
                    if (ROOT/"runtime/STOP").exists():
                        raise RuntimeError("User stopped")
                    directory = out/f"turn_{turn+1:02d}"
                    directory.mkdir()
                    latest = dict(state["latest"])
                    shutil.copy2(out/"sent_frames"/f"{latest['frame']:06d}.jpg", directory/"input.jpg")
                    prompt = f"[HEARTBEAT {turn+1}] {TASKS[scene]} Inspect the most recent video. Decide whether another tool is needed or report the current visible status. At most one tool this turn; after tool result end turn."
                    (directory/"prompt.txt").write_text(prompt)
                    await session.send_realtime_input(text=prompt)
                    timeline("prompt", text=prompt, turn=turn+1)
                    messages, calls = [], 0
                    async def receive():
                        nonlocal calls
                        async for msg in session.receive():
                            raw = msg.model_dump(mode="json", exclude_none=True)
                            raw.pop("session_resumption_update", None)
                            messages.append({"wall_time": time.time(), "message": raw})
                            write_json(directory/"messages.json", messages)
                            if msg.tool_call:
                                state["phase"] = "EXECUTING"
                                timeline("tool_call", text=json.dumps(raw["tool_call"]))
                                replies = []
                                for call in msg.tool_call.function_calls:
                                    result = {"error": "One tool per heartbeat; await fresh observation"}
                                    if calls == 0 and len(state["actions"]) < 4:
                                        calls += 1
                                        try:
                                            payload = {**dict(call.args), "skill": call.name}
                                            skill_request(scene, payload)
                                            result = await asyncio.to_thread(bridge, "skill", **payload)
                                            state["actions"].append({"request": payload, "result": result, "wall_time": time.time()})
                                            write_json(directory/"tool_result.json", state["actions"][-1])
                                        except Exception as exc:
                                            result = {"error": redact(exc)}
                                    replies.append(types.FunctionResponse(id=call.id, name=call.name,
                                        response={**result, "next": "End this turn. Inspect fresh video on next heartbeat."}))
                                await session.send_tool_response(function_responses=replies)
                                state["phase"] = "VERIFY"
                                timeline("tool_result", text=json.dumps([r.model_dump(mode="json") for r in replies]))
                            if msg.server_content and msg.server_content.turn_complete:
                                break
                    await asyncio.wait_for(receive(), min(150, max(1, deadline-time.monotonic())))
                    parts = []
                    for item in messages:
                        content = item["message"].get("server_content", {})
                        if content.get("output_transcription", {}).get("text"):
                            parts.append(content["output_transcription"]["text"])
                        else:
                            parts.extend(p.get("text", "") for p in content.get("model_turn", {}).get("parts", []))
                    response = "".join(parts)
                    (directory/"response.txt").write_text(response)
                    if not calls:
                        try:
                            report = parse_json(response)
                            if not isinstance(report, dict):
                                raise ValueError("Expected object")
                        except ValueError:
                            report = {"status": "uncertain", "reason": response}
                        report["wall_time"] = time.time()
                        state["reports"].append(report)
                        timeline("report", text=json.dumps(report))
                        print(f"Turn {turn+1}: {report}", flush=True)
                        if report.get("status") == "goal_complete":
                            if scene == "drawer" and perturb and not state["injected"]:
                                await asyncio.sleep(1.5)
                                await asyncio.to_thread(bridge, "perturb")
                                state["injected"] = True
                                state["phase"] = "CONTINUED_OBSERVATION"
                            elif state["actions"]:
                                break
                    else:
                        print(f"Turn {turn+1}: {state['actions'][-1:]}", flush=True)
                    if not calls and len(state["actions"]) >= 4:
                        state["termination"] = "physical_tool_budget_exhausted"
                        break
                    await asyncio.sleep(2.5)
    except Exception as exc:
        failure = {"error": redact(exc)}
        write_json(out/"error.json", failure)
    finally:
        if feeder:
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)
        await asyncio.to_thread(bridge, "hold")
        final = await asyncio.to_thread(bridge, "snapshot")
        shutil.copy2(ROOT/final["image"], out/"final.png")
        await asyncio.to_thread(bridge, "record_stop")
        write_json(out/"state.json", state)
        await asyncio.to_thread(encode_videos, out)
    events = [read_json(p) for p in sorted((out/"events").glob("*.json"))]
    state["injected"] = any(e.get("type") in ("drawer_closed", "barrier_inserted") for e in events)
    write_json(out/"state.json", state)
    result = evaluate(scene, jsonl(out/"evaluation_trace.jsonl"), events, state["reports"])
    result.update(run_error=failure, mode="smoke" if smoke else "Live_API", frames_sent=state["sent"])
    if smoke:
        result["task_success"] = None
        result["recovery_success"] = None
        result["note"] = "Controller-only test: task-level visual judgment not assessed. Inspect arrival/opening metrics and manual_result.json."
    write_json(out/"evaluation.json", result)
    if not smoke:
        await asyncio.to_thread(render, out)
        from scenario_materials import build
        build(out)
    composite = "" if smoke else "- [통합 영상](presentation/synchronized.mp4)\n"
    mode_note = "API 없이 하위 제어기만 실행한 사전 점검입니다." if smoke else "하위 제어 도구와 ER2의 시각 판단을 연결한 단일 사례입니다."
    (out/"README.md").write_text(f"# {scene}\n\n{composite}- [입력 카메라](input_video/camera.mp4)\n- [실제 동작](rollout_video/side.mp4)\n- [평가](evaluation.json)\n\n{mode_note} 상세 해석은 프로젝트 실행방법을 참고하세요.\n")
    print(json.dumps(result, indent=2), flush=True)
    return {"output": str(out.relative_to(ROOT)), "evaluation": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=["drawer", "patrol"])
    parser.add_argument("--smoke", action="store_true", help="Low-level controller only; no API calls")
    parser.add_argument("--no-perturb", action="store_true")
    args = parser.parse_args()
    asyncio.run(experiment(args.scene, perturb=not args.no_perturb, smoke=args.smoke))
