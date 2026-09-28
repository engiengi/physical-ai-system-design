"""Build experiment stills and motion-quality plots from recorded frames; no API or simulator calls."""
import argparse
import bisect
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from common import read_json, write_json
from presentation import jsonl
from scenario_rules import evaluate


def build(out):
    out = Path(out)
    scene = read_json(out/"protocol.json")["scenario"]
    rows = jsonl(out/"evaluation_trace.jsonl")
    if not rows:
        raise ValueError("Recorded evaluation trace required")
    events = sorted([read_json(p) for p in (out/"events").glob("*.json")], key=lambda x: x["wall_time"])
    reports = read_json(out/"state.json")["reports"]
    dest = out/"materials"
    dest.mkdir(exist_ok=True)
    audit = evaluate(scene, rows, events, reports)
    write_json(dest/"evaluation_audit.json", audit)
    chosen = [("01_initial", rows[0])]
    times = [r["wall_time"] for r in rows]

    def at(timestamp):
        return rows[max(0, min(len(rows)-1, bisect.bisect_right(times, timestamp)-1))]

    if scene == "drawer":
        closure = next((e for e in events if e["type"] == "drawer_closed"), None)
        before = [r for r in rows if not closure or r["wall_time"] < closure["wall_time"]]
        chosen.append(("02_peak_open_before_closure", max(before, key=lambda r: r["drawer_open_m"])))
        if closure:
            chosen.append(("03_after_external_closure", at(closure["wall_time"]+.5)))
            after = [r for r in rows if r["wall_time"] > closure["wall_time"]]
            if after:
                chosen.append(("04_peak_reopening", max(after, key=lambda r: r["drawer_open_m"])))
    else:
        for label, kind in (("02_barrier_inserted", "barrier_inserted"), ("03_safety_stop", "obstacle_stop")):
            event = next((e for e in events if e["type"] == kind), None)
            if event:
                chosen.append((label, at(event["wall_time"]+.3)))
        reroute = next((e for e in events if e["type"] == "skill_start" and e.get("route") == "right"), None)
        if reroute:
            detour = [r for r in rows if r["wall_time"] > reroute["wall_time"]]
            chosen.append(("04_detour_in_progress", min(detour, key=lambda r: (r["position"][0]-2.5)**2+(r["position"][1]+1.8)**2)))
    chosen.append(("05_final", rows[-1]))
    if scene == "drawer" and any("controller_phase" in r for r in rows):
        for label, phase in (("06_grasp_contact", "grasp"), ("07_slow_pull", "pull"), ("08_release", "release"), ("09_retreat", "retreat")):
            candidates = [r for r in rows if r.get("controller_phase") == phase]
            if candidates:
                # Last frame of this phase in the first opening, before injection.
                before = [r for r in candidates if not closure or r["wall_time"] < closure["wall_time"]]
                chosen.append((label, (before or candidates)[-1]))
    manifest = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    for label, row in chosen:
        frame = row["frame"]
        canvas = Image.new("RGB", (1280, 540), "#111e30")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 6), label+f" | recorded frame {frame} | sim {row['simulation_time']:.2f}s", font=font, fill="white")
        draw.text((12, 33), "Recorded camera (10 FPS); exact API JPEGs: sent_frames/", font=font, fill="#82d9ff")
        draw.text((652, 33), "Physical rollout / overview", font=font, fill="#82d9ff")
        for folder, x in (("input_frames", 0), ("side_frames", 640)):
            with Image.open(out/folder/f"{frame:06d}.jpg") as im:
                canvas.paste(im.convert("RGB"), (x, 60))
        canvas.save(dest/(label+".png"))
        manifest.append({"image": label+".png", "source_frame": frame, "trace": row})
    write_json(dest/"stills_manifest.json", manifest)
    if scene == "drawer" and audit["motion_quality_available"]:
        canvas = Image.new("RGB", (1280, 900), "#111e30")
        draw = ImageDraw.Draw(canvas)
        draw.text((25, 12), "Drawer motion quality | actual recorded measurements", font=font, fill="white")
        duration = max(.1, rows[-1]["simulation_time"]-rows[0]["simulation_time"])
        panels = [("Drawer opening (m): TOP blue / BOTTOM orange", .30,
                   [("drawer_open_m", "#82d9ff"), ("drawer_bottom_open_m", "#ffbb62")]),
                  ("Arm speed (rad/s): sampled blue / physics-step peak orange", 1.05,
                   [("arm_joint_speed_max_rad_s", "#82d9ff"), ("physics_peak_arm_speed_rad_s", "#ffbb62")]),
                  ("Finger-to-handle forces (N): left blue / right orange", 40.,
                   [("left_contact", "#82d9ff"), ("right_contact", "#ffbb62")])]
        for index, (label, upper, series) in enumerate(panels):
            y = 75+index*260
            draw.text((25, y-27), label, font=font, fill="white")
            draw.rectangle((75, y, 1230, y+195), outline="#5b7389")
            draw.text((8, y), str(upper), font=font, fill="white")
            draw.text((42, y+176), "0", font=font, fill="white")
            for key, color in series:
                points = []
                for row in rows:
                    value = row.get(key, 0.)
                    if key in ("left_contact", "right_contact"):
                        value = row.get("finger_handle_contact_n", [0., 0.])[key == "right_contact"]
                    x = 75+(row["simulation_time"]-rows[0]["simulation_time"])/duration*1155
                    points.append((x, y+195-min(upper, max(0., value))/upper*195))
                draw.line(points, fill=color, width=2)
            for event in events:
                if event["type"] == "drawer_closed":
                    x = 75+(event["simulation_time"]-rows[0]["simulation_time"])/duration*1155
                    draw.line((x, y, x, y+195), fill="#e886ff", width=2)
            draw.text((75, y+199), "0", font=font, fill="white")
            draw.text((1060, y+199), f"{duration:.1f} sim seconds", font=font, fill="white")
        draw.text((25, 862), "Purple = external closure | contact chart clipped at 40N; full values in evaluation_trace.jsonl", font=font, fill="#b8cee3")
        canvas.save(dest/"motion_quality.png")
        with (dest/"motion_quality.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerows((key, value) for key, value in audit.items() if not isinstance(value, (dict, list)))
    with (dest/"events.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wall_seconds_from_recording", "event", "detail"])
        for event in events:
            writer.writerow([round(event["wall_time"]-times[0], 3), event["type"], str(event)])
    links = "\n".join(f"- [{m['image']}]({m['image']}) — 원본 프레임 {m['source_frame']}" for m in manifest)
    if (dest/"motion_quality.png").exists():
        links += "\n- [동작 품질 그래프](motion_quality.png)\n- [품질 수치표](motion_quality.csv)"
    (dest/"README.md").write_text("# 실험 장면과 동작 품질\n\n실행 중 기록한 주요 장면과 평가 자료입니다.\n\n"
        +links+"\n\n- [입력·판단·동작 통합 영상](../presentation/synchronized.mp4)\n"
        "- [동작 평가](evaluation_audit.json)\n- [물리 이벤트 시점](events.csv)\n"
        "- [프레임 출처](stills_manifest.json)\n\n이미지 왼쪽은 전체 카메라 기록입니다. 모델에 실제 전송한 저속 JPEG는 `../sent_frames/`에 별도로 보관합니다.\n")
    print(str(dest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    build(parser.parse_args().run)
