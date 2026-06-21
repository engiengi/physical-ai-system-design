#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-lerobot}"
WORK_DIR="${WORK_DIR:-$HOME/lerobot_test}"

if ! command -v conda >/dev/null 2>&1; then
  echo "오류: conda 명령을 찾을 수 없습니다. Miniconda 또는 Anaconda를 먼저 설치하세요." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12
fi

conda activate "${ENV_NAME}"
conda install -y ffmpeg=7.1.1 -c conda-forge

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
