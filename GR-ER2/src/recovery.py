"""One baseline + one disturbed task-level visual feedback trial (not a benchmark)."""
import argparse
import json
import shutil
from pathlib import Path

from common import ROOT, read_json, write_json

GOAL = "Put the red block into the green tray. Check the current image before acting. If it is already inside the tray, report completion without another action."


def summarize(actions, events):
    first_failed = bool(actions) and not actions[0]["metrics"]["placement_within_tolerance"]
    later_red_success = any(a["metrics"]["picked_candidate"] == "red" and
                            a["metrics"]["placement_within_tolerance"] for a in actions[1:])
    latest = actions[-1]["metrics"] if actions else {}
    red = latest.get("after", {}).get("red")
    red_in_tray = bool(red and 0.375 <= red[0] <= 0.605 and 0.16 <= red[1] <= 0.31
                       and 0.02 <= red[2] <= 0.08)
    picks = [a["action"]["pick"] for a in actions]
    return {"evaluation_only": True, "actions": len(actions), "perturbations": len(events),
            "first_action_failed": first_failed, "later_red_placement_succeeded": later_red_success,
            "final_red_inside_tray": red_in_tray,
            "recovery_observed": bool(events and first_failed and later_red_success and red_in_tray),
            "pick_coordinates_yx": picks,
            "changed_pick_after_first_action": len(picks) > 1 and picks[0] != picks[1],
            "note": "Geometric checks plus image review; one trial, not a success-rate estimate."}


def collect_actions(trial):
    records = []
    for step in sorted(trial.glob("step_*")):
        result_file = step / "tool_result.json"
        if not result_file.exists():
            continue
        source = ROOT / read_json(result_file)["output"]
        dst = trial / "actions" / step.name
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "final.png", "input_video/camera.mp4", "rollout_video/side.mp4"):
            (dst / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, dst / name)
        records.append({"action": read_json(step / "action.json"),
                        "metrics": read_json(source / "metrics.json"),
                        "original_output": str(source.relative_to(ROOT))})
    return records


def write_report(out, results):
    lines = ["# 방해 후 재관찰·복구 결과", "",
             "고정 상부 카메라의 단계별 RGB 관찰을 사용합니다. 전체 영상은 기록용이며 모델에 연속 전송하지 않습니다.", "",
             "| 조건 | 실제 행동 수 | 방해 수 | 첫 행동 배치 실패 | 마지막 빨간 블록 트레이 내부 | 복구 관찰 |",
             "|---|---:|---:|---|---|---|"]
    for name, result in results.items():
        lines.append(f"| {name} | {result['actions']} | {result['perturbations']} | {result['first_action_failed']} | {result['final_red_inside_tray']} | {result['recovery_observed']} |")
    lines += ["", "baseline의 recovery_observed=false는 방해를 적용하지 않았기 때문입니다.", "",
              "## 자료 보기", ""]
    for name in results:
        lines += [f"### {name}", "",
                  f"- [전체 상부 관찰 영상]({name}/input_video/camera.mp4)",
                  f"- [전체 측면 동작 영상]({name}/rollout_video/side.mp4)",
                  f"- [최종 이미지]({name}/final.png)",
                  f"- [평가 JSON]({name}/evaluation.json)", ""]
        for step in sorted((out / name).glob("step_*")):
            prefix = f"{name}/{step.name}"
            lines.append(f"- {step.name}: [모델 입력]({prefix}/input.png) · [모델 응답]({prefix}/response.json)")
            if (step / "overlay.png").exists():
                lines.append(f"  · [행동 좌표 오버레이]({prefix}/overlay.png) · [선택 좌표]({prefix}/action.json)")
        lines.append("")
    lines += ["## 해석 범위", "",
              "- 물체 이동은 첫 행동 시작 60 tick 후 실험 코드가 적용한 위치 변경입니다.",
              "- 진행 중인 저수준 행동은 원래 목표를 사용하고, 다음 행동에서 새 관찰을 반영합니다.",
              "- 기하 판정과 실제 영상을 함께 확인합니다. 부수적인 물체 접촉·이동도 검토해야 합니다.",
              "- 조건당 한 번 실행한 사례입니다. 복구 성공률·실시간 시각 서보·손목 카메라 실험으로 해석하지 않습니다."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_recovery():
    # Lazy import allows app.py to dispatch this workflow without an import cycle.
    from app import agent, bridge, encode_videos, redact, run_dir
    out = run_dir("03_perturbation_recovery")
    protocol = {"instruction": GOAL, "camera": "fixed overhead /World/TopCamera",
                "model_input": "one fresh RGB image per decision; no continuous video input",
                "control": "Franka pick/place; RGB-D coordinates, no evaluator poses sent to model",
                "conditions": {"baseline": {"max_decisions": 1},
                               "perturbed": {"max_decisions": 3, "shift_action_tick": 60}},
                "perturbation": "red block teleported after first action starts; at 60 Hz tick 60 is 1 simulation second",
                "feedback": "model sees fresh image and prior action/controller status; no perturbation oracle",
                "reset": "scene 0 before each condition", "repetitions": 1}
    write_json(out / "protocol.json", protocol)
    print("Recovery output: " + str(out), flush=True)
    results = {}
    try:
        for condition, budget in (("baseline", 1), ("perturbed", 3)):
            trial = out / condition
            trial.mkdir()
            bridge("reset", scene=0)
            recording = False
            try:
                bridge("record_start", output=str(trial.relative_to(ROOT)))
                recording = True
                agent({"instruction": GOAL, "steps": budget,
                       "perturb_first_action": condition == "perturbed"}, output=trial)
            finally:
                if recording:
                    bridge("record_stop")
                    encode_videos(trial)
            actions = collect_actions(trial)
            events = [read_json(p) for p in sorted((trial / "events").glob("*.json"))]
            result = summarize(actions, events)
            write_json(trial / "evaluation.json", result)
            results[condition] = result
            write_json(out / "summary.json", results)
            write_report(out, results)
            print(condition + ": " + json.dumps(result, ensure_ascii=False), flush=True)
        return {"output": str(out.relative_to(ROOT)), "conditions": results}
    except Exception as exc:
        write_json(out / "error.json", {"error": redact(exc), "completed_conditions": results,
                                       "note": "Incomplete trial; do not report as recovery success."})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run_recovery(), ensure_ascii=False, indent=2))
