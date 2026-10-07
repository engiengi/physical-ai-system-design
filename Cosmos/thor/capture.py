"""Record a visible Thor terminal showing actual policy request artifacts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

TITLE = 'Thor - Cosmos Live Inference'

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['start', 'stop', 'monitor'])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--policy-run', required=True)
    p.add_argument('--workspace', required=True, type=Path)
    a = p.parse_args()
    root = a.workspace.resolve()
    title = TITLE + ' | ' + a.policy_run
    run_dir = root / 'outputs/thor_project' / a.policy_run
    requests = root / 'outputs/thor_project' / a.policy_run / 'requests'
    env = dict(os.environ)
    display = env.get('DISPLAY')
    if not display: raise RuntimeError('Visible X11 session required; set DISPLAY and XAUTHORITY')
    meta = a.output / 'thor_recording.json'
    if a.mode == 'monitor':
        while True:
            rows = []
            for d in sorted(requests.glob('*')):
                try:
                    r = json.loads((d / 'result.json').read_text())
                    rows.append(f"  {r['request_id']:06d}     {r['status']:8s}   {r.get('inference_seconds', 0):7.3f} s    {str(r.get('action_shape', '-')):10s}")
                except (OSError, ValueError):
                    rows.append(f'  {d.name}     RECEIVED / PROCESSING (result not written yet)')
            for f in sorted(run_dir.glob('*/result.json')):
                try:
                    r=json.loads(f.read_text())
                    rows.append(f"  {r['id']} | {r['status']} | {r.get('seconds',0):.2f}s")
                    if r.get('prediction'): rows.append('  '+json.dumps(r['prediction'],ensure_ascii=False))
                    if r.get('error'): rows.append('  '+r['error'])
                    if r.get('video'): rows.append('  VIDEO SAVED: '+r['video'])
                except (OSError,ValueError): pass
            for name in ['progress.json','failure.json']:
                f=run_dir/name
                if f.exists():
                    try: rows.append('  '+json.dumps(json.loads(f.read_text()),ensure_ascii=False))
                    except (OSError,ValueError): pass
            done = (a.output / 'finished.flag').exists()
            print('\033[2J\033[H\033[1;36mNVIDIA JETSON THOR | COSMOS LIVE INFERENCE\033[0m\n', end='')
            print('  Run: '+a.policy_run+'\n  Source: actual model progress and saved results\n')
            print('  Mode: '+('policy requests from Spark' if requests.exists() else 'stored video analysis / generation')+'\n')
            print('  ACTUAL REQUEST / CASE RESULTS (seconds)\n  ' + '-' * 63)
            print('\n'.join(rows[-17:]) or '  Waiting for Spark observations ...')
            print('\n  ' + ('CAPTURE COMPLETE - results saved separately' if done else 'LIVE | ' + now()), flush=True)
            if (a.output / 'capture_closed.flag').exists(): break
            time.sleep(0.5)
    elif a.mode == 'start':
        a.output.mkdir(parents=True, exist_ok=False)
        subprocess.run(['gnome-terminal', '--title='+title, '--geometry=112x32+20+40', '--zoom=1.2', '--',
                        sys.executable, str(Path(__file__).resolve()), 'monitor', '--output', str(a.output),
                        '--policy-run', a.policy_run, '--workspace', str(root)], env=env, check=True)
        window = None
        for _ in range(40):
            tree = subprocess.check_output(['xwininfo', '-root', '-tree'], env=env, text=True)
            for line in tree.splitlines():
                if title in line:
                    match = re.search(r'(0x[0-9a-f]+)', line)
                    candidate = match[1] if match else None
                    if candidate and 'Map State: IsViewable' in subprocess.check_output(['xwininfo', '-id', candidate], env=env, text=True):
                        window = candidate
                        break
            if window:
                break
            time.sleep(.25)
        if not window:
            raise RuntimeError('Visible Thor monitor window not found')
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-f', 'x11grab', '-window_id', str(int(window, 16)),
                   '-framerate', '15', '-draw_mouse', '0', '-i', display, '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20', '-threads', '4', '-pix_fmt', 'yuv420p',
                   str(a.output / 'thor_screen.mkv')]
        with (a.output / 'recording.log').open('w') as log:
            proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        record = dict(started_at=now(), recorder_pid=proc.pid, window_id=window, command=command,
                      policy_run=a.policy_run, capture='Actual Thor X11 terminal window, wall-clock, no audio',
                      display=display, source=str(run_dir))
        meta.write_text(json.dumps(record, indent=2))
        time.sleep(2)
        if proc.poll() is not None:
            raise RuntimeError('Thor recorder exited; inspect recording.log')
        print(json.dumps(record))
    else:
        r = json.loads(meta.read_text())
        (a.output / 'finished.flag').write_text(now())
        time.sleep(2)
        pid = r['recorder_pid']
        cmdline = Path(f'/proc/{pid}/cmdline')
        if cmdline.exists() and str(a.output / 'thor_screen.mkv').encode() in cmdline.read_bytes():
            os.kill(pid, signal.SIGINT)
        elif not cmdline.exists() or not cmdline.read_bytes():
            r['recorder_exited_before_stop'] = True
        else:
            raise RuntimeError('Recorder PID changed; refusing to signal another process')
        for _ in range(80):
            status = Path(f'/proc/{pid}/status')
            if not status.exists() or re.search(r'State:\s+Z', status.read_text()):
                break
            time.sleep(.25)
        else:
            raise RuntimeError('Recorder did not finish')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(a.output / 'thor_screen.mkv'),
                        '-c', 'copy', '-movflags', '+faststart', str(a.output / 'thor_screen.mp4')], check=True)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                         'stream=width,height,nb_frames:format=duration', '-of', 'json', str(a.output / 'thor_screen.mp4')], text=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(a.output / 'thor_screen.mp4'), '-f', 'null', '-'], check=True)
        r.update(finished_at=now(), probe=json.loads(probe), full_decode_valid=True)
        (a.output / 'capture_closed.flag').write_text(now())
        meta.write_text(json.dumps(r, indent=2))
        print(json.dumps(r))

if __name__ == '__main__':
    main()
