#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-lerobot}"
WORK_DIR="${WORK_DIR:-$HOME/lerobot_test}"

if ! command -v conda >/dev/null 2>&1; then
  echo "오류: conda 명령을 찾을 수 없습니다. Miniconda 또는 Anaconda를 먼저 설치하세요." >&2
  exit 1
fi

if [[ -n "${PYTHONPATH:-}" ]] && echo "${PYTHONPATH}" | grep -q "/opt/ros/"; then
  echo "경고: PYTHONPATH에 ROS 경로가 감지되었습니다." >&2
  echo "  현재 PYTHONPATH: ${PYTHONPATH}" >&2
  echo "" >&2
  echo "  ROS 경로가 Python 3.12 conda 환경과 충돌해 PyTorch/NumPy import 오류가" >&2
  echo "  발생할 수 있습니다. 새 터미널을 열거나 다음을 먼저 실행하세요." >&2
  echo "" >&2
  echo "    export PYTHONPATH=" >&2
  echo "" >&2
  read -r -p "PYTHONPATH를 초기화하고 계속 진행하시겠습니까? [y/N] " _reply
  if [[ "${_reply}" =~ ^[Yy]$ ]]; then
    export PYTHONPATH=
    echo "PYTHONPATH를 초기화했습니다." >&2
  else
    echo "설치를 중단합니다." >&2
    exit 1
  fi
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12
fi

conda activate "${ENV_NAME}"
conda install -y "ffmpeg>=6" -c conda-forge

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

if [[ ! -d lerobot/.git ]]; then
  git clone https://github.com/huggingface/lerobot.git
fi

cd lerobot
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ".[smolvla]"
python -m pip install -e ".[libero]"
python -m pip install -e ".[hilserl]"

python - <<'PY'
import torch
import lerobot
print("LeRobot import OK")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo
echo "설치가 완료되었습니다."
echo "다음부터는 'conda activate ${ENV_NAME}'를 실행해 환경을 활성화하세요."
