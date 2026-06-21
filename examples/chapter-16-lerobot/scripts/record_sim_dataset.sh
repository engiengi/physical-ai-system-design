#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/../configs/gym_hil_record.json}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "오류: 설정 파일을 찾을 수 없습니다: ${CONFIG_PATH}" >&2
  exit 1
fi

python -m lerobot.rl.gym_manipulator \
  --config_path "${CONFIG_PATH}"
