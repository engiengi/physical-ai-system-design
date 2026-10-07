"""Record fixed-setting, fresh-process trials varying only the action execution horizon."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

from cosmos_spark.artifacts import write_json
from cosmos_spark.remote import ssh
from cosmos_spark.settings import HOST, WORKSPACE, URI

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--horizons', type=int, nargs='+', default=[32, 16, 8], choices=[8, 16, 32])
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.batch):
        parser.error('Use an alphanumeric batch ID')
    out = ROOT / 'reports' / args.batch
    out.mkdir(parents=True, exist_ok=False)
    remote_python = WORKSPACE + '/.venv/bin/python'
    helper = WORKSPACE + '/deploy/current/spark/scripts/thor/official_reference_server.py'
    write_json(out / 'plan.json', {
        'task': 'ToolOrganizationTask', 'horizons': args.horizons,
        'environment_seed': 0, 'server_rng_seed': 1200,
        'guidance_interval': [960, 1001], 'fresh_server_each_case': True,
        'scope': 'One trajectory per horizon; not a success-rate benchmark. Initial rendered observations are audited separately.',
    })
    for horizon in args.horizons:
        run_id = f'{args.batch}_h{horizon}'
        server = 'policy_' + run_id
        remote_out = WORKSPACE + '/outputs/thor_project/' + server
        command = ['env', 'COSMOS_THOR_CODE=' + WORKSPACE + '/deploy/current/thor',
                   remote_python, '-u', helper, '--run', server, '--interval', 'on']
        current = False
        process = None
        try:
            write_json(out / 'status.json', {'phase': 'loading', 'horizon': horizon, 'run': run_id})
            ssh(['tmux', 'new-session', '-d', '-s', server,
                 shlex.join(command) + ' > ' + shlex.quote(WORKSPACE + '/' + server + '.log') + ' 2>&1'])
            current = True
            endpoint = URI.replace('ws://', 'http://').rstrip('/') + '/healthz'
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(endpoint, timeout=3) as response:
                        if response.status == 200:
                            runtime = json.loads(ssh([remote_python, '-c',
                                'import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())', remote_out + '/runtime.json']))
                            write_json(out / f'h{horizon}_runtime.json', runtime)
                            break
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError('Server did not become ready: ' + server)
            write_json(out / 'status.json', {'phase': 'running', 'horizon': horizon, 'run': run_id})
            with (out / f'h{horizon}.log').open('w') as log:
                process = subprocess.Popen([sys.executable, '-m', 'cosmos_spark.session',
                    '--entrypoint', 'official', '--run-id', run_id, '--task', 'ToolOrganizationTask',
                    '--horizon', str(horizon)], cwd=ROOT, env=dict(os.environ, DISPLAY=':1'),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = process.wait(timeout=5400)
            episode = ROOT / 'runs' / run_id / 'episode_0000/episode.json'
            if code or not episode.exists():
                raise RuntimeError(f'Trial failed, exit {code}: {run_id}')
            write_json(out / f'h{horizon}_result.json', json.loads(episode.read_text()))
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=45)
            if current:
                ssh([remote_python, '-c',
                    "import pathlib,os,signal,sys,time; p=pathlib.Path(sys.argv[1]); pid=int((p/'pid').read_text()); c=pathlib.Path(f'/proc/{pid}/cmdline'); b=c.read_bytes() if c.exists() else b''; assert not b or (b'official_reference_server.py' in b and sys.argv[2].encode() in b); os.kill(pid,signal.SIGINT) if b else None",
                    remote_out, server])
                # Confirm release before the next process tries to acquire the GPU lock.
                for _ in range(60):
                    alive = ssh([remote_python, '-c',
                        "import pathlib,sys;p=pathlib.Path(sys.argv[1]); pid=int((p/'pid').read_text());c=pathlib.Path(f'/proc/{pid}/cmdline');print(int(c.exists() and sys.argv[2].encode() in c.read_bytes()))", remote_out, server]).strip()
                    if alive == '0':
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('Server did not stop')
                subprocess.run(['rsync', '-a', HOST + ':' + remote_out + '/', str(out / server) + '/'], check=True)
    write_json(out / 'status.json', {'phase': 'complete', 'horizons': args.horizons})


if __name__ == '__main__':
    main()
