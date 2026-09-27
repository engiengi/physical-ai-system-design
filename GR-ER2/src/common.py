"""Pure helpers shared by the host application and Isaac's Python environment."""
import json
import math
import os
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def uid():
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("좌표는 [y, x] 두 값이어야 합니다.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1000 for v in value):
        raise ValueError("좌표는 0~1000의 유한한 숫자여야 합니다.")
    return [float(v) for v in value]


def pixel(value, width, height):
    y, x = point(value)
    return x * (width - 1) / 1000, y * (height - 1) / 1000


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def safe_child(base, relative):
    base = Path(base).resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError("허용되지 않은 경로입니다.")
    return target


def validate_action(action):
    if set(action) != {"pick", "place", "observation_id"}:
        raise ValueError("pick, place, observation_id만 전달해야 합니다.")
    point(action["pick"])
    point(action["place"])
    obs = action["observation_id"]
    if not isinstance(obs, str) or not obs or any(c not in "0123456789_abcdef" for c in obs):
        raise ValueError("잘못된 observation_id입니다.")
    return action
