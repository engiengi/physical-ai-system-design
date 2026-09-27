#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$PROJECT_ROOT"
CONTAINER=gr-er2-sim
GUI_CONTAINER=gr-er2-sim-gui
IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0-dev2
ACTION="${1:-status}"
case "$ACTION" in
  start|stream|gui)
    SIM_ARGS=()
    SCENARIO="${GR_ER2_SCENE:-blocks}"
    ENTRY=src/simulation.py
    case "$SCENARIO" in
      blocks) ;;
      drawer|patrol) ENTRY=src/scenario_sim.py; SIM_ARGS=(--scenario "$SCENARIO") ;;
      dual_franka|transport) ENTRY=src/coop_sim.py; SIM_ARGS=(--scenario "$SCENARIO") ;;
      *) echo 'GR_ER2_SCENE은 blocks, drawer, patrol, dual_franka, transport 중 하나입니다.'; exit 1 ;;
    esac
    DISPLAY_ARGS=()
    OTHER_CONTAINER="$GUI_CONTAINER"
    if [[ "$ACTION" == stream ]]; then
      [[ -n "${2:-}" ]] || { echo '사용법: bash scripts/sim.sh stream SERVER_IP'; exit 1; }
      python3 -c 'import ipaddress,sys; ipaddress.IPv4Address(sys.argv[1])' "$2"
      SIM_ARGS+=(--stream --stream-ip "$2")
    elif [[ "$ACTION" == gui ]]; then
      [[ "${DISPLAY:-}" =~ ^:[0-9]+(\.[0-9]+)?$ ]] || {
        echo 'GPU 호스트 데스크톱의 터미널에서 실행하세요. 로컬 DISPLAY(:숫자)가 필요합니다.'; exit 1;
      }
      command -v xauth >/dev/null || { echo 'sudo apt-get install xauth 로 설치하세요.'; exit 1; }
      OTHER_CONTAINER="$CONTAINER"
      CONTAINER="$GUI_CONTAINER"
      SIM_ARGS+=(--gui)
    fi
    if [[ "$(docker inspect --format '{{.State.Running}}' "$OTHER_CONTAINER" 2>/dev/null || true)" == true ]]; then
      echo '다른 화면 모드가 실행 중입니다. 먼저 bash scripts/sim.sh stop 을 실행하세요.'; exit 1
    fi
    # A copied checkout must never silently execute another checkout's code.
    if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
      BOUND_PROJECT="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/GR-ER2"}}{{.Source}}{{end}}{{end}}' "$CONTAINER")"
      if [[ "$BOUND_PROJECT" != "$PROJECT_ROOT" ]]; then
        echo "컨테이너 프로젝트 경로 불일치: $BOUND_PROJECT" >&2
        echo "현재 경로: $PROJECT_ROOT. 실행방법.md 10절의 폴더 복사 후 복구 절차를 확인하세요." >&2
        exit 1
      fi
    fi
    if [[ "$ACTION" == gui ]]; then
      # Copy only this display's cookie. Never disable X access control with xhost +.
      mkdir -p "$PROJECT_ROOT/runtime/x11"
      chmod 750 "$PROJECT_ROOT/runtime/x11"
      AUTH_SOURCE="${XAUTHORITY:-}"
      if [[ -z "$AUTH_SOURCE" && -r "$HOME/.Xauthority" ]]; then
        AUTH_SOURCE="$HOME/.Xauthority"
      fi
      if [[ -z "$AUTH_SOURCE" && -r "/run/user/$(id -u)/gdm/Xauthority" ]]; then
        AUTH_SOURCE="/run/user/$(id -u)/gdm/Xauthority"
      fi
      [[ -r "$AUTH_SOURCE" ]] || { echo '데스크톱의 XAUTHORITY 파일을 확인하세요.'; exit 1; }
      AUTH_FILE="$PROJECT_ROOT/runtime/x11/Xauthority"
      AUTH_TEMP=$(mktemp "$PROJECT_ROOT/runtime/x11/auth.XXXXXX")
      xauth -f "$AUTH_SOURCE" nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$AUTH_TEMP" nmerge -
      if [[ ! -s "$AUTH_TEMP" ]]; then
        rm -f "$AUTH_TEMP"
        echo '현재 DISPLAY의 인증 정보를 찾지 못했습니다.'; exit 1
      fi
      chmod 640 "$AUTH_TEMP"
      mv "$AUTH_TEMP" "$AUTH_FILE"
      DISPLAY_ARGS=(-v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    fi
    mkdir -p "$PROJECT_ROOT"/{runtime,data/samples,outputs}
    if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
      docker run -d --name "$CONTAINER" --gpus all --network host --ipc host \
        -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y --entrypoint bash \
        -v "$PROJECT_ROOT:/workspace/GR-ER2" "${DISPLAY_ARGS[@]}" "$IMAGE" -lc 'sleep infinity'
    fi
    docker start "$CONTAINER" >/dev/null
    if docker exec "$CONTAINER" bash -lc 'pgrep -f "[s]rc/(simulation|scenario_sim|coop_sim).py" >/dev/null'; then
      echo '시뮬레이터가 이미 실행 중입니다. 전환하려면 먼저 stop 하세요.'
      exit 0
    fi
    # Isaac's installation is private to uid 1234. Share only project output
    # directories through the host group, retaining the image's original user.
    chmod 2775 "$PROJECT_ROOT" "$PROJECT_ROOT/runtime" "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/data/samples"
    # Container uid 1234 creates this file with the host group and umask 0002.
    # A host group member can write it, but cannot chmod a different owner's file.
    if [[ -f "$PROJECT_ROOT/runtime/isaac.log" && ! -w "$PROJECT_ROOT/runtime/isaac.log" ]]; then
      chmod g+w "$PROJECT_ROOT/runtime/isaac.log" || {
        echo 'runtime/isaac.log 쓰기 권한을 확인하세요. 기존 로그를 보존한 뒤 소유자/그룹을 확인하세요.' >&2
        exit 1
      }
    fi
    EXEC_ENV=()
    python3 -c 'import json,time,pathlib,sys; pathlib.Path("runtime/status.json").write_text(json.dumps({"ready":False,"updated":time.time(),"world_scenario":sys.argv[1],"note":"starting simulator"}))' "$SCENARIO"
    if [[ "$ACTION" == gui ]]; then
      EXEC_ENV=(-e "DISPLAY=$DISPLAY" -e XAUTHORITY=/workspace/GR-ER2/runtime/x11/Xauthority)
    fi
    docker exec -d -u "1234:$(id -g)" "${EXEC_ENV[@]}" "$CONTAINER" bash -lc \
      'umask 0002; cd /workspace/GR-ER2; /isaac-sim/python.sh "$@" > runtime/isaac.log 2>&1' bash "$ENTRY" "${SIM_ARGS[@]}"
    echo '시작 요청 완료. bash scripts/sim.sh logs 에서 GR_ER2_READY를 기다리세요.'
    ;;
  logs) tail -n 80 -f "$PROJECT_ROOT/runtime/isaac.log" ;;
  status)
    docker ps -a --filter 'name=^/gr-er2-sim(-gui)?$'
    if [[ -f "$PROJECT_ROOT/runtime/status.json" ]]; then
      python3 -m json.tool "$PROJECT_ROOT/runtime/status.json"
    fi
    ;;
  stop)
    python3 -c 'import json,time,pathlib; p=pathlib.Path("runtime/status.json"); p.parent.mkdir(exist_ok=True); p.write_text(json.dumps({"ready":False,"updated":time.time(),"note":"stopping simulator"}))'
    # Exact project container only; existing lecture containers are untouched.
    for target in "$CONTAINER" "$GUI_CONTAINER"; do
      if docker container inspect "$target" >/dev/null 2>&1; then
        docker stop -t 20 "$target"
      fi
    done
    python3 -c 'import json,time,pathlib; pathlib.Path("runtime/status.json").write_text(json.dumps({"ready":False,"updated":time.time(),"note":"simulator stopped"}))'
    if [[ -f "$PROJECT_ROOT/runtime/x11/Xauthority" ]]; then
      rm -f "$PROJECT_ROOT/runtime/x11/Xauthority"
    fi
    ;;
  *) echo '사용법: bash scripts/sim.sh start|gui|stream SERVER_IP|status|logs|stop'; exit 1 ;;
esac
