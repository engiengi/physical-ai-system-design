"""ER 2 Live API streaming supervisor; explicit bounded experimental workflow."""
import argparse
import asyncio
import io
import json
import os
import shutil
import time

from PIL import Image
from common import ROOT, read_json, write_json, validate_action, parse_json

MODEL = read_json(ROOT / "configs/default.json").get("streaming_model", "gemini-robotics-er-2-streaming-preview")
SCENARIOS = {
    "target_shift": "집기 대상 이동: 동작 60 tick에 빨간 블록 위치 변경",
    "goal_shift": "목적지 이동: 동작 60 tick에 빈 트레이를 X -18 cm 이동",
    "grasp_drop": "운반 중 낙하: 빨간 블록이 12 cm 이상 들리면 그리퍼 개방",
}

TOOLS = [{"function_declarations": [
    {"name": "recover_pick_place", "behavior": "BLOCKING",
     "description": "Only in RECOVER phase: physically pick red block and place in green tray, waiting until movement ends. Use normalized [y,x] image coordinates and the supplied Observation ID.",
     "parameters": {"type": "OBJECT", "properties": {
         "pick": {"type": "ARRAY", "items": {"type": "NUMBER"}, "minItems": 2, "maxItems": 2},
         "place": {"type": "ARRAY", "items": {"type": "NUMBER"}, "minItems": 2, "maxItems": 2},
         "observation_id": {"type": "STRING"}}, "required": ["pick", "place", "observation_id"]}}
]}]


def evaluate_stream(event, detection, interrupted, first_metrics, recovery_metrics, model_complete):
    latency = None if not (event and detection) else detection["wall_time"] - event["wall_time"]
    red = recovery_metrics.get("after", {}).get("red") if recovery_metrics else None
    center = recovery_metrics.get("tray_center", [0.49, 0.235]) if recovery_metrics else [0.49, 0.235]
    inside = bool(red and abs(red[0] - center[0]) <= 0.115 and abs(red[1] - center[1]) <= 0.075 and 0.02 <= red[2] <= 0.08)
    success = bool(recovery_metrics and recovery_metrics.get("picked_candidate") == "red"
                   and recovery_metrics.get("placement_within_tolerance") and inside)
    before = recovery_metrics.get("before", {}).get("red") if recovery_metrics else None
    already_inside = bool(before and abs(before[0]-center[0]) <= 0.115 and abs(before[1]-center[1]) <= 0.075
                          and 0.02 <= before[2] <= 0.08)
    return {"evaluation_only": True, "perturbation_present": bool(event),
            "model_detected_change": bool(detection), "detection_latency_wall_seconds": latency,
            "interrupted_while_action_active": bool(interrupted),
            "first_controller_status": first_metrics.get("controller_status"),
            "recovery_placement_succeeded": success, "model_reported_complete": bool(model_complete),
            "red_already_inside_before_recovery_action": already_inside,
            "streaming_recovery_observed": bool(event and detection and latency is not None and latency >= 0
                                                 and interrupted and success),
            "note": "One fixed-camera trial, hybrid initial standard ER2 actor + Live supervisor/recovery. Not a controlled model accuracy comparison."}


def drop_confirmed(trace, event):
    """Physical validation only: opening request is not proof that the body fell."""
    if not event or event.get("type") != "grasp_drop" or event.get("red_at_release", [0, 0, 0])[2] <= 0.12:
        return False
    for row in trace:
        if row["wall_time"] < event["wall_time"]:
            continue
        red, tray = row["red"], row["tray_center"]
        outside = abs(red[0]-tray[0]) > 0.115 or abs(red[1]-tray[1]) > 0.075
        if red[2] < 0.06 and outside and min(row["fingers"]) > 0.025:
            return True
    return False


def write_report(out, result, state):
    lines = ["# Live 스트리밍 감지·중단·복구", "",
             "초기 행동은 일반 ER 2가 계획하고, 별도의 ER 2 Streaming 세션이 카메라를 계속 관찰합니다.",
             "Live 모델의 변화 보고 후 행동을 중단하고 관찰 자세로 복귀해 같은 Live 세션에서 재계획합니다. 물체를 초기화하지 않습니다.", "",
             "| 항목 | 값 |", "|---|---|"]
    for key, value in result.items():
        if key not in ("evaluation_only", "note"):
            lines.append(f"| {key} | {value} |")
    lines += ["", "## 영상·입력·출력", "",
              "- [전체 상부 영상](input_video/camera.mp4)", "- [전체 측면 영상](rollout_video/side.mp4)",
              "- [통합 해설 영상](presentation/synchronized.mp4)",
              "- [최종 장면](final.png)", "- [종합 평가](evaluation.json)",
              "- 실제 Live 입력은 `sent_frames/`의 JPEG와 전송 시각 JSON입니다.",
              "- `turn_XX/`에는 heartbeat 직전 기준 이미지, 원문 응답과 도구 호출이 있습니다.", ""]
    for index, action in enumerate(state["actions"], 1):
        relative = os.path.relpath(ROOT / action["output"], out)
        lines.append(f"- 복구 행동 {index}: [측면 영상]({relative}/rollout_video/side.mp4) · [평가]({relative}/metrics.json)")
    lines += ["", "## 해석 범위", "",
              "조건당 한 회의 사례입니다. 초기 모델·관찰 방식이 기본 실험과 달라 성공률 비교를 주장하지 않습니다.",
              "감지 시각은 모델의 보고가 호스트에 도착한 시각입니다. JPEG 최대 1 FPS 입력과 API 지연을 포함합니다.",
              "동작 중 중단 성공과 최종 배치 성공을 구분하고, 부수적인 물체 접촉도 영상으로 확인합니다."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def jpeg(path):
    buffer = io.BytesIO()
    Image.open(path).convert("RGB").save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


async def probe(transport="realtime"):
    from app import client, redact, run_dir
    from google.genai import types
    out = run_dir("04_streaming_probe")
    print("Live probe: " + str(out), flush=True)
    try:
        async with client().aio.live.connect(model=MODEL, config={"response_modalities": ["TEXT"]}) as session:
            for _ in range(3):
                await session.send_realtime_input(video=types.Blob(data=jpeg(ROOT / "data/samples/scene_01.png"), mime_type="image/jpeg"))
                await asyncio.sleep(1.1)
            prompt = "Briefly name the colored blocks you can see in this camera image."
            if transport == "realtime":
                await session.send_realtime_input(text=prompt)
            else:
                await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(
                    inline_data=types.Blob(data=jpeg(ROOT / "data/samples/scene_01.png"), mime_type="image/jpeg")),
                    types.Part(text=prompt)]), turn_complete=True)
            messages = []
            async def receive():
                async for msg in session.receive():
                    messages.append(msg.model_dump(mode="json", exclude_none=True))
                    if msg.server_content and msg.server_content.turn_complete:
                        break
            await asyncio.wait_for(receive(), 60)
            write_json(out / "messages.json", messages)
            if not messages:
                raise RuntimeError("No Live response")
            print(json.dumps(messages, ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(out / "error.json", {"error": redact(exc)})
        raise


async def experiment(max_seconds=240, scenario="target_shift"):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown scenario")
    from app import client, infer, bridge, sim_action, encode_videos, overlay, run_dir, CONFIG, TOOL, COORDS, redact
    from google.genai import types
    out = run_dir("04_streaming_recovery")
    protocol = {"streaming_model": MODEL, "initial_actor_model": CONFIG["models"][0],
                "camera": "/World/TopCamera (fixed overhead)", "transport": "persistent Live API WebSocket",
                "video_input": "JPEG frames, >=1.1 seconds between sends; saved in sent_frames/",
                "heartbeat": "only after previous turn completes; no blind prompt timer",
                "physical_tools": "blocking; independent Live supervisor observes while initial actor motion runs",
                "scenario": scenario, "perturbation": SCENARIOS[scenario],
                "controller": "hold on model target_moved report, return to observation joints (no object reset), then Live recover_pick_place",
                "max_live_seconds": max_seconds, "max_turns": 24, "max_recovery_actions": 2,
                "repetitions": 1, "model_has_perturbation_oracle": False}
    write_json(out / "protocol.json", protocol)
    print("Streaming output: " + str(out), flush=True)
    await asyncio.to_thread(bridge, "reset", scene=0)
    seed = await asyncio.to_thread(bridge, "snapshot")
    initial_dir = out / "initial_actor"
    initial_dir.mkdir()
    shutil.copy2(ROOT / seed["image"], initial_dir / "input.png")
    prompt = ("Put the red block into the green tray. Call pick_place exactly once. " + COORDS
              + f" Observation ID: {seed['observation_id']}")
    response, _ = await asyncio.to_thread(infer, client(), CONFIG["models"][0], initial_dir / "input.png", prompt, initial_dir, [TOOL])
    items = getattr(response, "outputs", None) or getattr(response, "steps", None) or []
    calls = [c for c in items if getattr(c, "type", "") == "function_call"]
    if len(calls) != 1 or calls[0].name != "pick_place":
        raise ValueError("Initial actor must propose exactly one pick_place")
    first_action = calls[0].arguments
    if isinstance(first_action, str):
        first_action = parse_json(first_action)
    validate_action(first_action)
    if first_action["observation_id"] != seed["observation_id"]:
        raise ValueError("Invalid initial observation ID")
    write_json(initial_dir / "action.json", first_action)

    state = {"latest": None, "sent": 0, "phase": "REFERENCE", "reports": [], "actions": [],
             "interrupted": False, "detection": None, "model_complete": False}
    def timeline(kind, **data):
        with (out / "api_timeline.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"wall_time": time.time(), "kind": kind, "phase": state["phase"], **data}, ensure_ascii=False) + "\n")
    motion = None
    first_result = None
    failure = None
    feeder = None
    recording = False
    try:
        await asyncio.to_thread(bridge, "record_start", output=str(out.relative_to(ROOT)))
        recording = True
        config = {"response_modalities": ["TEXT"], "tools": TOOLS,
                  "system_instruction": 'You supervise a Franka using a live overhead camera. The goal is the red block inside the green tray at its CURRENT location. Never assume controller completion means success. For monitoring return plain JSON {"status":"unchanged|target_moved|goal_moved|grasp_failed|goal_complete|uncertain","reason":"what you see"}. Report target_moved only if red moves away before being grasped; goal_moved if the green tray moved from its original location; grasp_failed if a lifted red block falls out of the gripper outside the tray. Normal carried motion is unchanged. If unsure report uncertain. Use recover_pick_place only when explicitly in RECOVER phase, ONCE per heartbeat. After its response, END YOUR TURN with a brief text acknowledgement; do not call any further tools until a new user heartbeat. All coordinates are [y,x] in 0..1000. Do not invent world coordinates.'}
        write_json(out / "live_config.json", config)
        async with client().aio.live.connect(model=MODEL, config=config) as session:
            async def feed():
                while True:
                    obs = await asyncio.to_thread(bridge, "snapshot")
                    data = jpeg(ROOT / obs["image"])
                    index = state["sent"]
                    directory = out / "sent_frames"
                    directory.mkdir(exist_ok=True)
                    (directory / f"{index:06d}.jpg").write_bytes(data)
                    await session.send_realtime_input(video=types.Blob(data=data, mime_type="image/jpeg"))
                    metadata = {**obs, "frame": index, "sent_wall_time": time.time(), "phase": state["phase"]}
                    write_json(directory / f"{index:06d}.json", metadata)
                    state["latest"] = metadata
                    state["sent"] += 1
                    await asyncio.sleep(1.1)

            feeder = asyncio.create_task(feed())
            deadline = time.monotonic() + max_seconds
            while state["sent"] < 3:
                if feeder.done():
                    feeder.result()
                if time.monotonic() > deadline:
                    raise TimeoutError("No streaming frames")
                await asyncio.sleep(0.2)

            for turn in range(24):
                if time.monotonic() > deadline:
                    raise TimeoutError("Bounded Live experiment time budget reached")
                if feeder.done():
                    feeder.result()
                if (ROOT / "runtime/STOP").exists():
                    raise RuntimeError("User stopped experiment")
                directory = out / f"turn_{turn+1:02d}"
                directory.mkdir()
                latest = dict(state["latest"])
                write_json(directory / "latest_sent_frame.json", latest)
                shutil.copy2(out / "sent_frames" / f"{latest['frame']:06d}.jpg", directory / "input.jpg")
                instruction = (f"[HEARTBEAT] Phase={state['phase']}. Original red pick [y,x]={first_action['pick']}. "
                               f"Original planned tray place [y,x]={first_action['place']}. "
                               f"Latest Observation ID: {latest['observation_id']}. "
                               'REFERENCE: remember original red and green tray positions and return JSON status. MONITOR: explicitly compare BOTH current red and tray locations with the reference; inspect for displaced ungrasped red, relocated green tray, or red dropped outside tray after lifting. Include current tray center in your visual reason. Return JSON status. '
                               'RECOVER: choose current red block and empty tray floor, call recover_pick_place ONCE, then end turn after tool response. VERIFY: return JSON status whether red is inside tray. '
                               'For a JSON report use {"status":"unchanged|target_moved|goal_moved|grasp_failed|goal_complete|uncertain","reason":"visual evidence"}. Only follow the CURRENT phase.')
                (directory / "prompt.txt").write_text(instruction, encoding="utf-8")
                await session.send_realtime_input(text=instruction)
                timeline("prompt", turn=turn+1, text=instruction, frame=latest["frame"])
                messages, reports = [], []
                physical_calls = 0

                async def receive_turn():
                    nonlocal physical_calls
                    async for msg in session.receive():
                        raw = msg.model_dump(mode="json", exclude_none=True)
                        raw.pop("session_resumption_update", None)
                        messages.append({"wall_time": time.time(), "message": raw})
                        write_json(directory / "messages.json", messages)
                        if msg.tool_call:
                            timeline("tool_call", text=json.dumps(raw["tool_call"], ensure_ascii=False))
                            replies = []
                            for call in msg.tool_call.function_calls:
                                params = dict(call.args)
                                result = {"accepted": False}
                                if call.name == "recover_pick_place" and state["phase"] == "RECOVER" and physical_calls == 0 and len(state["actions"]) < 2:
                                    physical_calls += 1
                                    try:
                                        validate_action(params)
                                        if params["observation_id"] != latest["observation_id"]:
                                            raise ValueError("Use latest heartbeat Observation ID")
                                        write_json(directory / "action.json", params)
                                        overlay(directory / "input.jpg", [{"label": "pick", "point": params["pick"]},
                                                                          {"label": "place", "point": params["place"]}], directory / "overlay.png")
                                        result = await asyncio.to_thread(sim_action, {"command": "pick_place", "action": params})
                                        state["actions"].append({"action": params, **result})
                                        write_json(directory / "tool_result.json", result)
                                        # Privileged metrics stay on disk; model receives controller status only.
                                        result = {"controller_status": result["controller_status"], "note": "Inspect fresh video before declaring success."}
                                    except Exception as exc:
                                        result = {"error": redact(exc)}
                                        write_json(directory / "action_error.json", result)
                                replies.append(types.FunctionResponse(name=call.name, id=call.id, response=result))
                            for reply in replies:
                                reply.response["next"] = "End this turn now with a short text acknowledgement. Wait for a new user heartbeat. Do not call another tool."
                            await session.send_tool_response(function_responses=replies)
                            timeline("tool_result", text=json.dumps([r.model_dump(mode="json") for r in replies], ensure_ascii=False))
                        if msg.server_content and msg.server_content.turn_complete:
                            break
                await asyncio.wait_for(receive_turn(), min(60, max(1, deadline - time.monotonic())))
                if state["phase"] != "RECOVER":
                    parts = []
                    for entry in messages:
                        content = entry["message"].get("server_content", {})
                        if content.get("output_transcription", {}).get("text"):
                            parts.append(content["output_transcription"]["text"])
                        elif content.get("model_turn"):
                            parts.extend(p.get("text", "") for p in content["model_turn"].get("parts", []))
                    text = "".join(parts)
                    (directory / "response.txt").write_text(text, encoding="utf-8")
                    try:
                        report = parse_json(text)
                        if report.get("status") not in ("unchanged", "target_moved", "goal_moved", "grasp_failed", "goal_complete", "uncertain"):
                            raise ValueError("Unknown visual status")
                    except (ValueError, AttributeError):
                        report = {"status": "uncertain", "reason": "Could not parse model response", "raw": text}
                    report.update(wall_time=time.time(), phase=state["phase"])
                    reports.append(report)
                    state["reports"].append(report)
                    timeline("report", text=json.dumps(report, ensure_ascii=False))
                    if report["status"] in ("target_moved", "goal_moved", "grasp_failed") and state["phase"] == "MONITOR" and not state["detection"]:
                        state["detection"] = report
                        held = await asyncio.to_thread(bridge, "hold")
                        state["interrupted"] = held["interrupted"]
                        write_json(directory / "hold.json", held)
                write_json(directory / "messages.json", messages)
                write_json(directory / "reports.json", reports)
                print(f"Live turn {turn+1} {state['phase']}: {reports}", flush=True)

                if state["phase"] == "REFERENCE":
                    state["phase"] = "MONITOR"
                    timeline("action_start", text=json.dumps(first_action))
                    motion = asyncio.create_task(asyncio.to_thread(sim_action, {"command": "pick_place", "action": first_action, "perturb_first_action": True, "scenario": scenario}))
                elif state["phase"] == "MONITOR" and state["detection"]:
                    first_result = await motion
                    write_json(initial_dir / "tool_result.json", first_result)
                    state["phase"] = "RETURN_TO_OBSERVE"
                    home = await asyncio.to_thread(bridge, "return_home")
                    write_json(out / "return_home.json", home)
                    if home["controller_status"] != "completed":
                        raise RuntimeError("Observation-pose return was stopped")
                    state["phase"] = "RECOVER"
                    # Let a fresh home-pose observation reach the stream before planning.
                    await asyncio.sleep(2.5)
                elif state["phase"] == "MONITOR" and motion and motion.done():
                    # A missed disturbance remains a recorded negative case. No blind retries.
                    state["model_complete"] = any(r.get("status") == "goal_complete" for r in reports)
                    break
                elif state["phase"] == "RECOVER" and state["actions"]:
                    state["phase"] = "VERIFY"
                elif state["phase"] == "VERIFY":
                    if any(r.get("status") == "goal_complete" for r in reports):
                        state["model_complete"] = True
                        break
                    if len(state["actions"]) < 2:
                        state["phase"] = "RECOVER"
                await asyncio.sleep(1.2)
            if motion:
                if not motion.done():
                    await asyncio.to_thread(bridge, "hold")
                first_result = await motion
                write_json(initial_dir / "tool_result.json", first_result)
            final = await asyncio.to_thread(bridge, "snapshot")
            shutil.copy2(ROOT / final["image"], out / "final.png")
    except Exception as exc:
        failure = {"error": redact(exc), "phase": state["phase"]}
        write_json(out / "error.json", failure)
    finally:
        if feeder:
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)
        if motion and not motion.done():
            await asyncio.to_thread(bridge, "hold")
            first_result = await motion
        elif motion and first_result is None:
            first_result = await motion
        if recording:
            # Also stop a recovery tool whose awaiting task timed out/cancelled.
            await asyncio.to_thread(bridge, "hold")
            final = await asyncio.to_thread(bridge, "snapshot")
            shutil.copy2(ROOT / final["image"], out / "final.png")
            await asyncio.to_thread(bridge, "record_stop")
            await asyncio.to_thread(encode_videos, out)
        write_json(out / "state.json", state)
    events = [read_json(p) for p in sorted((out / "events").glob("*.json"))]
    first_metrics = read_json(ROOT / first_result["output"] / "metrics.json") if first_result else {}
    recovery_metrics = read_json(ROOT / state["actions"][-1]["output"] / "metrics.json") if state["actions"] else {}
    result = evaluate_stream(events[0] if events else None, state["detection"], state["interrupted"], first_metrics, recovery_metrics, state["model_complete"])
    result["frames_sent"] = state["sent"]
    result["scenario"] = scenario
    final_metrics = recovery_metrics or first_metrics
    result["final_task_succeeded"] = evaluate_stream(None, None, False, {}, final_metrics, False)["recovery_placement_succeeded"]
    result["run_error"] = failure
    result["detection_type_matches_scenario"] = bool(state["detection"] and state["detection"]["status"] ==
        {"target_shift": "target_moved", "goal_shift": "goal_moved", "grasp_drop": "grasp_failed"}[scenario])
    if scenario == "grasp_drop":
        from presentation import jsonl
        result["physical_drop_confirmed"] = drop_confirmed(jsonl(out / "evaluation_trace.jsonl"), events[0] if events else None)
        result["streaming_recovery_observed"] &= result["physical_drop_confirmed"]
    write_json(out / "evaluation.json", result)
    write_json(out / "initial_metrics.json", first_metrics)
    write_json(out / "recovery_metrics.json", recovery_metrics)
    write_report(out, result, state)
    from presentation import render
    await asyncio.to_thread(render, out)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return {"output": str(out.relative_to(ROOT)), "evaluation": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--transport", choices=["realtime", "turn"], default="realtime")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="target_shift")
    args = parser.parse_args()
    asyncio.run(probe(args.transport) if args.probe else experiment(scenario=args.scenario))
