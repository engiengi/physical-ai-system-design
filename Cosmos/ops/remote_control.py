"""Thor-side lifecycle for one explicitly managed policy process."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def alive(state):
    if not state:
        return False
    try:
        proc = Path('/proc') / str(state['pid'])
        args = (proc / 'cmdline').read_bytes().split(b'\0')
        return (state['entrypoint'].encode() in args and
                str(proc.stat().st_ctime_ns) == state['process_identity'])
    except (FileNotFoundError, ProcessLookupError):
        return False


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('status', 'stop', 'check'):
        sub.add_parser(name)
    start = sub.add_parser('start')
    start.add_argument('--run', required=True)
    start.add_argument('--port', type=int, default=8000)
    start.add_argument('--seed', type=int, default=0)
    start.add_argument('--decode-video', action='store_true')
    logs = sub.add_parser('logs')
    logs.add_argument('--follow', action='store_true')
    logs.add_argument('--lines', type=int, default=60)
    ready = sub.add_parser('ready'); ready.add_argument('--snapshot', required=True)
    a = p.parse_args(); root = a.workspace.resolve()
    folder = root / 'deploy'
    folder.mkdir(exist_ok=True)
    # Serialize state changes from simultaneous Spark commands.
    lock = (folder / 'control.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX)
    state_path = folder / 'policy.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if a.command == 'check':
        code = (folder / 'current').resolve(strict=True)
        required = [root / '.venv/bin/python', root / 'external/cosmos-framework',
                    root / 'models/Cosmos3-Edge-Policy-DROID', code / 'thor/serve.py']
        missing = [str(x) for x in required if not x.exists()]
        if missing: raise FileNotFoundError(missing)
        print(json.dumps({'status': 'ok', 'release': str(code)}, indent=2)); return
    if a.command == 'status':
        active = alive(state)
        print(json.dumps({'active': active, 'listening': active and listening(state['port']),
                          'policy': state}, indent=2)); return
    if a.command == 'ready':
        if not alive(state) or not listening(state['port']):
            raise RuntimeError('Managed Thor policy is not ready; inspect lab logs')
        if state['snapshot_sha256'] != a.snapshot:
            raise RuntimeError('Running policy uses a different snapshot; stop and start it before running Spark')
        print(json.dumps({'ready': True, 'run': state['run']})); return
    if a.command == 'logs':
        if not state: raise RuntimeError('No managed policy run')
        fcntl.flock(lock, fcntl.LOCK_UN)
        subprocess.run(['tail', '-n', str(a.lines), *(['-f'] if a.follow else []), state['log']], check=True)
        return
    if a.command == 'stop':
        if alive(state):
            os.kill(state['pid'], signal.SIGINT)
            deadline = time.monotonic() + 30
            while alive(state) and time.monotonic() < deadline:
                time.sleep(.2)
            if alive(state): raise RuntimeError('Policy did not stop after SIGINT; inspect its log')
        print(json.dumps({'active': False})); return
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', a.run) or not 1 <= a.port <= 65535:
        raise ValueError('Invalid run ID or port')
    if alive(state): raise RuntimeError('A managed policy is active; stop it before starting another')
    gpu_lock = (root / '.thor-experiment.lock').open('a+')
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another experiment owns the GPU; leave it running or stop it explicitly')
    if listening(a.port): raise RuntimeError('Policy port is already occupied')
    release = (folder / 'current').resolve(strict=True)
    deployment = json.loads((release / 'deployment.json').read_text())
    if (root / 'outputs/thor_project' / a.run).exists(): raise FileExistsError(a.run)
    control = folder / 'runs' / a.run
    control.mkdir(parents=True, exist_ok=False)
    pidfile = control / 'pid'
    logfile = control / 'server.log'
    entrypoint = release / 'thor/serve.py'
    cmd = [str(root / '.venv/bin/python'), '-u', str(entrypoint), '--workspace', str(root),
           '--run', a.run, '--host', '0.0.0.0', '--port', str(a.port), '--seed', str(a.seed)]
    if a.decode_video: cmd.append('--decode-video')
    command = 'echo $$ > ' + shlex.quote(str(pidfile)) + '; exec ' + shlex.join(cmd)
    command += ' > ' + shlex.quote(str(logfile)) + ' 2>&1'
    gpu_lock.close()  # serve.py acquires and retains the actual GPU lock.
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'cosmos-lab-' + a.run,
                    'sh -c ' + shlex.quote(command)], check=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if pidfile.exists():
            pid = int(pidfile.read_text())
            proc = Path('/proc') / str(pid)
            try:
                if str(entrypoint).encode() in (proc / 'cmdline').read_bytes().split(b'\0'):
                    state = {'pid': pid, 'process_identity': str(proc.stat().st_ctime_ns),
                             'run': a.run, 'entrypoint': str(entrypoint), 'port': a.port,
                             'release': str(release), 'log': str(logfile),
                             'snapshot_sha256': deployment['snapshot_sha256'],
                             'git_commit': deployment['git_commit']}
                    save(state_path, state)
                    print(json.dumps({'status': 'starting', 'policy': state}, indent=2)); return
            except FileNotFoundError:
                pass
        time.sleep(.1)
    raise RuntimeError('Policy failed to launch; inspect ' + str(logfile))


if __name__ == '__main__':
    main()
