#!/usr/bin/env bash
# Private X11 desktop, reachable through an authenticated SSH tunnel only.
set -euo pipefail
[[ "$(uname -m)" == x86_64 ]] || { echo 'Run this on the x86 Brev VM.' >&2; exit 1; }
sudo apt-get install -y xserver-xorg-core x11vnc novnc websockify openbox xterm x11-xserver-utils
if ! tmux has-session -t cosmos-xorg 2>/dev/null; then
    nvidia-xconfig --allow-empty-initial-configuration --no-use-display-device \
        --virtual=1600x900 --output-xconfig=/tmp/cosmos-xorg.conf
    tmux new-session -d -s cosmos-xorg \
        'sudo -n Xorg :99 -config /tmp/cosmos-xorg.conf -dpi 96 -noreset -nolisten tcp -ac -logfile /tmp/cosmos-xorg.log'
fi
for trial in $(seq 1 30); do
    if DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then break; fi
    sleep 1
done
DISPLAY=:99 xdpyinfo >/dev/null
COSMOS_DISPLAY_OUTPUT=$(DISPLAY=:99 xrandr --query | awk '/ connected/{print $1; exit}')
[[ -n "$COSMOS_DISPLAY_OUTPUT" ]] || { echo 'No connected virtual display found.' >&2; exit 1; }
DISPLAY=:99 xrandr --output "$COSMOS_DISPLAY_OUTPUT" --mode 1600x900 --dpi 96
if ! tmux has-session -t cosmos-openbox 2>/dev/null; then
    tmux new-session -d -s cosmos-openbox 'DISPLAY=:99 openbox-session > /tmp/cosmos-openbox.log 2>&1'
fi
if ! tmux has-session -t cosmos-vnc 2>/dev/null; then
    tmux new-session -d -s cosmos-vnc \
        'x11vnc -display :99 -localhost -rfbport 5900 -nopw -forever -shared -xkb > /tmp/cosmos-vnc.log 2>&1'
fi
if ! tmux has-session -t cosmos-novnc 2>/dev/null; then
    tmux new-session -d -s cosmos-novnc \
        'websockify --web=/usr/share/novnc/ 127.0.0.1:6080 127.0.0.1:5900 > /tmp/cosmos-novnc.log 2>&1'
fi
for trial in $(seq 1 15); do
    if curl -fsS http://127.0.0.1:6080/vnc.html >/dev/null; then
        printf 'Desktop ready: display :99, SSH-forward localhost:6080\n'
        exit 0
    fi
    sleep 1
done
echo 'Desktop web endpoint unavailable. Inspect /tmp/cosmos-{xorg,vnc,novnc}.log.' >&2
exit 1
