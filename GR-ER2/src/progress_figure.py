"""Render the recorded progress evaluations as a manuscript table; no API calls."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("두 Franka · 정상", "dual_franka/normal/20260923_115838_f575a075"),
    ("두 Franka · 전달 실패", "dual_franka/exception/20260923_120319_39fb72d8"),
    ("자재 운송 · 정상", "transport/normal/20260923_135551_c283363f"),
    ("자재 운송 · 적재 실패", "transport/load_failure/20260923_140000_72da572d"),
    ("자재 운송 · 통로 장애", "transport/obstacle/20260923_140558_80ceac85"),
]


def seconds(value):
    return "미완료" if value is None else f"{value:.1f}초"


def load_rows():
    rows = []
    for label, run in RUNS:
        for phase, phase_label in [("midpoint", "중간"), ("final", "최종")]:
            source = ROOT / "outputs/07_cooperative" / run / "video_progress" / phase / "evaluation.json"
            data = json.loads(source.read_text())
            model = data["model"]
            if data["bracket_matches"] is False:
                verdict = "진행률·완료 오판" if data["completion_presence_matches"] is False else "진행률 과대 판단"
            elif data["completion_absolute_error_seconds"] is not None:
                error = data["completion_absolute_error_seconds"]
                verdict = "시각 표기 오류¹" if run.startswith("dual_franka/") else f"완료 시각 차이 {error:.1f}초"
            else:
                verdict = "구간·미완료 일치"
            if data["response_status"] != "completed":
                verdict += "²"
            rows.append({
                "experiment": label, "phase": phase_label,
                "reference_bracket": data["reference_bracket"],
                "model_bracket": model["progress_level"],
                "reference_completion_seconds": data["reference_completion_time_seconds"],
                "model_completion_seconds": model["completion_time_seconds"],
                "verdict": verdict, "response_status": data["response_status"],
                "source": source.relative_to(ROOT).as_posix(),
                "evaluation": data,
            })
    return rows


def render(rows, output, font_path):
    width, height = 2560, 1710
    image = Image.new("RGB", (width, height), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    fonts = {size: ImageFont.truetype(str(font_path), size, index=1) for size in (25, 28, 30, 32, 36, 62)}

    def text(x, y, value, size=30, color="#1d2d45"):
        draw.text((x, y), value, font=fonts[size], fill=color)

    text(90, 54, "영상 Progress 평가 | 실제 상태와 모델 판단", 62)
    text(94, 149, "5개 실행 × 중간·최종 영상  ·  저장된 API 응답과 사후 물리 평가 비교  ·  Gemini Robotics ER 2", 30, "#566579")
    xs = [90, 550, 695, 990, 1285, 1610, 1925, 2470]
    top, header, row_height = 225, 96, 83
    draw.rounded_rectangle((90, top, 2470, top + header), radius=12, fill="#19334f")
    headers = ["실험 조건", "시점", "실제 구간", "모델 구간", "실제 완료 시각", "모델 완료 시각", "확인할 점"]
    for x, label in zip(xs, headers):
        text(x + 20, top + 24, label, 32, "#ffffff")
    for index, row in enumerate(rows):
        y = top + header + index * row_height
        mismatch = not row["evaluation"]["bracket_matches"]
        time_error = row["verdict"].startswith("시각")
        bg = "#fff0ee" if mismatch else ("#fff7e7" if time_error else ("#ffffff" if index // 2 % 2 == 0 else "#ecf1f7"))
        draw.rectangle((90, y, 2470, y + row_height), fill=bg)
        if mismatch or time_error:
            draw.rectangle((90, y, 99, y + row_height), fill="#c33f35" if mismatch else "#ba7923")
        values = [row["experiment"], row["phase"], row["reference_bracket"].replace("-", "–") + "%",
                  row["model_bracket"].replace("-", "–") + "%", seconds(row["reference_completion_seconds"]),
                  seconds(row["model_completion_seconds"]), row["verdict"]]
        for col, (x, value) in enumerate(zip(xs, values)):
            color = "#ad3028" if mismatch and col in (3, 5, 6) else "#1d2d45"
            text(x + 20, y + 21, value, 30, color)
        if index % 2 == 1:
            draw.line((90, y + row_height, 2470, y + row_height), fill="#d1dbe8", width=2)
    y = 1190
    draw.rounded_rectangle((90, y, 2470, y + 163), radius=14, fill="#19334f")
    text(120, y + 18, "통로 장애 · 중간: 도킹 완료 전, 모델은 B 도착까지 마쳤다고 판단", 36, "#ffffff")
    text(120, y + 85, "두 Franka 예외 · 최종: 실제 요구사항 3/4 충족, 모델은 205초에 전체 완료했다고 보고", 36, "#ffffff")
    notes = [
        "¹ 원문은 ‘03:00 (3.0 seconds)’로 단위가 혼재. JSON의 3.0초를 그대로 표시했으며 180초로 바꾸지 않음.",
        "² 두 Franka 예외의 중간 응답은 API 상태 incomplete. 파싱된 값은 보존했으며 정상 완료 응답과 구분함.",
        "실제 구간: 네 배치 요구사항 / 네 운송 단계의 물리 평가. 80–100% 구간과 작업 완료 여부는 별개로 확인.",
        "시각: 입력 영상 시작 기준. 영상 입력은 1 FPS. 운송 완료는 자재 해제·안정 배치 기준이며 팔 복귀 시각과 구분.",
        "출처: outputs/07_cooperative/<장면>/<조건>/<실행 ID>/video_progress/{midpoint,final}/evaluation.json",
        "각 조건 1회 사례 · 작업 제어와 별도의 영상 평가 · 가림이 오류의 원인인지는 별도 검증하지 않음",
    ]
    for index, note in enumerate(notes):
        text(96, 1380 + index * 46, note, 25, "#536075")
    image.save(output / "video_progress_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures/video_progress")
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    render(rows, args.output, args.font)
    (args.output / "sources.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    fields = [key for key in rows[0] if key != "evaluation"]
    with (args.output / "video_progress_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 원고 삽입용 영상 Progress 평가 표", "", "![Progress 평가](video_progress_comparison.png)", "",
             "원본 evaluation.json 10개를 직접 읽어 생성했다. 재추론·원본 결과 수정은 하지 않았다.", "",
             "- PNG: 원고 삽입용 2560×1710 이미지", "- CSV: 편집 가능한 수치·출처 표",
             "- sources.json: 사용한 원본 평가 값과 상대 경로", "",
             "## 원고 문장 해석", "",
             "두 Franka 예외에서 모델의 완료 주장과 실제 배치가 어긋났고, 통로 장애 중간 영상에서 진행률을 과대 판단했다.",
             "카메라 가림이 오류를 일으켰다는 인과관계는 별도 검증하지 않았다. 다음 표현을 권장한다:", "",
             "> 이번 사례에서는 모델의 완료 보고와 실제 최종 배치가 일치하지 않았습니다. 작업 성공을 결정할 때는 모델 보고와 별도로 최종 물체 위치와 배치 조건을 확인해야 합니다.", "",
             "두 Franka 예외의 중간 응답은 response_status=incomplete였다. 표의 각주로 표시했으며 원본은 유지했다.", "",
             "## 재생성", "", "GR-ER2 폴더에서 (결과 원본·Pillow·Noto CJK 폰트 필요, API/GPU 호출 없음):", "",
             "```bash", "python src/progress_figure.py", "```", "", "## 원본 출처", ""]
    lines.extend(f"- {row['experiment']} / {row['phase']}: `{row['source']}`" for row in rows)
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(args.output / "video_progress_comparison.png")


if __name__ == "__main__":
    main()
