#!/usr/bin/env bash
# First-time setup on a NEW Ubuntu Brev GPU VM; never replaces the GPU driver.
set -euo pipefail
unset PYTHONPATH PYTHONHOME
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" =~ ^(22.04|24.04)$ ]] || {
  echo 'Ubuntu 22.04/24.04 VM이 필요합니다.' >&2; exit 1;
}
[[ "$(uname -m)" == x86_64 ]] || { echo '이 Brev 경로는 x86_64용입니다.' >&2; exit 1; }
nvidia-smi
[[ "${GR_ACCEPT_ISAAC_EULA:-}" == YES ]] || {
  echo '실행방법의 NVIDIA 약관을 확인한 뒤 GR_ACCEPT_ISAAC_EULA=YES로 다시 실행하세요.' >&2
  exit 1
}
sudo apt-get update
sudo apt-get install -y git curl ca-certificates gnupg rsync python3-venv \
  ffmpeg tmux fonts-dejavu-core xauth xvfb x11vnc novnc websockify openbox x11-utils
command -v docker >/dev/null || {
  echo '먼저 실행방법 1.5절의 Docker Engine을 설치하세요.' >&2; exit 1;
}
docker version
mkdir -p runtime
python3 -m venv runtime/bootstrap-venv
runtime/bootstrap-venv/bin/python -m pip install 'uv==0.8.22'
runtime/bootstrap-venv/bin/uv python install 3.12.11
GR_PYTHON="$(runtime/bootstrap-venv/bin/uv python find 3.12.11)"
PYTHON="$GR_PYTHON" bash scripts/setup.sh
.venv/bin/python -I -m pip check
docker pull nvcr.io/nvidia/isaac-sim:6.0.0-dev2
docker run --rm --gpus all --entrypoint nvidia-smi nvcr.io/nvidia/isaac-sim:6.0.0-dev2
{
  cat /etc/os-release
  uname -m
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  docker version
  dpkg-query -W nvidia-container-toolkit libnvidia-container1 2>/dev/null || true
  .venv/bin/python --version
  dpkg-query -W xvfb x11vnc novnc websockify openbox
  docker image inspect nvcr.io/nvidia/isaac-sim:6.0.0-dev2 --format '{{json .RepoDigests}}'
} > runtime/brev_environment.txt
echo '설치 완료. 실행방법 Brev GUI 절을 진행하세요.'
