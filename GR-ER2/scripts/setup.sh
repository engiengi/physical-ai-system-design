#!/usr/bin/env bash
set -euo pipefail
# Keep system ROS/Conda Python paths out of this project's environment.
unset PYTHONPATH PYTHONHOME
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-python3.12}"
command -v "$PYTHON" >/dev/null || {
  echo 'Python 3.12가 필요합니다. 실행방법.md 1.1절을 확인하세요.' >&2; exit 1;
}
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 required")'
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "기존 .venv의 Python 버전이 다릅니다. 별도 이름으로 보존한 뒤 재설치하세요.")'
fi
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade 'pip==25.2'
.venv/bin/python -m pip install -r requirements.txt -c constraints.txt
mkdir -p data/samples data/uploads outputs runtime
.venv/bin/python -m pip freeze > runtime/host_requirements.lock.txt
.venv/bin/python -m unittest discover -s tests -v
echo '설치 완료. bash scripts/web.sh 로 GUI를 실행하세요.'
