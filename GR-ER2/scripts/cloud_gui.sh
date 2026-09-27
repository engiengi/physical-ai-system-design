#!/usr/bin/env bash
# Full Isaac Sim GUI over a localhost-only noVNC desktop and an SSH tunnel.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
STATE="$ROOT/runtime/cloud-desktop"
export DISPLAY=:99
export XAUTHORITY="$STATE/Xauthority"
ACTION="${1:-status}"
sessions=(gr-er2-cloud-vnc gr-er2-cloud-rfb gr-er2-cloud-wm gr-er2-cloud-x)
start_session() {
  local name="$1"; shift
  tmux new-session -d -s "$name" -c "$ROOT" \
    bash -c 'log="$1"; shift; exec "$@" >"$log" 2>&1' bash "$STATE/$name.log" "$@"
  tmux set-option -t "$name" @gr_er2_root "$ROOT"
}
case "$ACTION" in
  start)
    for command in Xvfb xauth x11vnc websockify openbox xdpyinfo tmux; do
      command -v "$command" >/dev/null || { echo "Missing $command: setup_brev.sh를 실행하세요." >&2; exit 1; }
    done
    if tmux has-session -t gr-er2-cloud-x 2>/dev/null; then
      [[ "$(tmux show-option -v -t gr-er2-cloud-x @gr_er2_root)" == "$ROOT" ]] || {
        echo '다른 프로젝트의 원격 데스크톱입니다. 기존 작업을 확인하세요.' >&2; exit 1;
      }
      for session in "${sessions[@]}"; do
        tmux has-session -t "$session" 2>/dev/null || {
          echo '원격 데스크톱 일부가 종료됐습니다. status와 로그를 확인한 뒤 이 프로젝트의 stop → start를 실행하세요.' >&2
          exit 1
        }
        [[ "$(tmux show-option -v -t "$session" @gr_er2_root)" == "$ROOT" ]] || {
          echo "다른 프로젝트의 세션입니다: $session" >&2; exit 1;
        }
      done
      xdpyinfo >/dev/null
      bash scripts/sim.sh gui
      exit 0
    fi
    [[ ! -e /tmp/.X99-lock && ! -S /tmp/.X11-unix/X99 ]] || {
      echo 'DISPLAY :99가 다른 작업에서 사용 중입니다. 기존 작업을 보존하고 확인하세요.' >&2; exit 1;
    }
    if ss -ltnH '( sport = :6080 or sport = :5901 )' | grep -q .; then
      echo '6080 또는 5901 포트가 사용 중입니다. 기존 작업을 확인하세요.' >&2; exit 1
    fi
    mkdir -p "$STATE"
    touch "$XAUTHORITY"
    chmod 600 "$XAUTHORITY"
    xauth -f "$XAUTHORITY" add "$DISPLAY" . "$(mcookie)"
    start_session gr-er2-cloud-x Xvfb "$DISPLAY" -screen 0 1280x960x24 -nolisten tcp -auth "$XAUTHORITY"
    for attempt in {1..30}; do
      if xdpyinfo >/dev/null 2>&1; then break; fi
      sleep 1
    done
    xdpyinfo >/dev/null
    start_session gr-er2-cloud-wm openbox
    start_session gr-er2-cloud-rfb x11vnc -display "$DISPLAY" -auth "$XAUTHORITY" \
      -rfbport 5901 -localhost -nopw -forever -shared -noxdamage
    start_session gr-er2-cloud-vnc websockify --web=/usr/share/novnc 127.0.0.1:6080 127.0.0.1:5901
    bash scripts/sim.sh gui
    echo '개인 PC에서 SSH -L 6080:127.0.0.1:6080 터널을 열고 http://127.0.0.1:6080/vnc.html 에 접속하세요.'
    ;;
  status)
    for session in "${sessions[@]}"; do
      tmux has-session -t "$session" 2>/dev/null && echo "$session: running" || echo "$session: stopped"
    done
    ss -ltn '( sport = :6080 or sport = :5901 )'
    bash scripts/sim.sh status
    ;;
  stop)
    for session in "${sessions[@]}"; do
      if tmux has-session -t "$session" 2>/dev/null; then
        [[ "$(tmux show-option -v -t "$session" @gr_er2_root 2>/dev/null || true)" == "$ROOT" ]] || {
          echo '소유권을 확인할 수 없는 원격 데스크톱입니다. 다른 작업을 종료하지 않습니다.' >&2
          exit 1
        }
      fi
    done
    bash scripts/sim.sh stop
    for session in "${sessions[@]}"; do
      if tmux has-session -t "$session" 2>/dev/null; then
        # Xvfb can survive tmux's SIGHUP. Terminate the verified pane process
        # first so it releases its display socket and lock before the next start.
        pid="$(tmux display-message -p -t "$session" '#{pane_pid}')"
        if [[ "$pid" =~ ^[0-9]+$ && -d "/proc/$pid" ]]; then
          # openbox and websockify legitimately change their working directory.
          # Ownership is established by this pane's project marker and uid.
          process_uid="$(ps -o uid= -p "$pid" | tr -d ' ')"
          [[ "$process_uid" == "$(id -u)" ]] || {
            echo "프로세스 소유자가 다릅니다: $session. 자동 종료를 중단합니다." >&2; exit 1;
          }
          kill -TERM "$pid"
        fi
        if tmux has-session -t "$session" 2>/dev/null; then
          tmux kill-session -t "$session" 2>/dev/null || true
        fi
      fi
    done
    for attempt in {1..30}; do
      [[ ! -e /tmp/.X99-lock && ! -S /tmp/.X11-unix/X99 ]] && break
      sleep 1
    done
    [[ ! -e /tmp/.X99-lock && ! -S /tmp/.X11-unix/X99 ]] || {
      echo 'DISPLAY :99 프로세스 종료를 확인하세요. 잠금 파일은 자동 삭제하지 않습니다.' >&2; exit 1;
    }
    ;;
  *) echo '사용법: bash scripts/cloud_gui.sh start|status|stop' >&2; exit 2 ;;
esac
