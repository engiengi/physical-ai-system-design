"""Deploy immutable source snapshots; keep models and environments on each device."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']


def run(argv, **kwargs):
    return subprocess.run([str(a) for a in argv], check=True, text=True, **kwargs)


def config(path):
    with path.open('rb') as f:
        cfg = tomllib.load(f)
    host = cfg['thor']['host']
    workspace = cfg['thor']['workspace']
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', host) or host.startswith('-'):
        raise ValueError('Invalid SSH host')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', workspace) or '..' in Path(workspace).parts or workspace == '/':
        raise ValueError('Use an absolute Thor workspace path without traversal')
    return cfg


def ssh(cfg, argv, **kwargs):
    return run(['ssh', *SSH_OPTIONS, cfg['thor']['host'], shlex.join([str(a) for a in argv])], **kwargs)


def remote_python(cfg, code, *args):
    result = ssh(cfg, ['python3', '-c', code, *args], capture_output=True)
    return result.stdout.strip()


def source_files(root=ROOT):
    files = []
    for directory in ('spark/src', 'spark/scripts', 'spark/tests', 'spark/configs', 'thor', 'ops'):
        for p in sorted((root / directory).rglob('*')):
            if p.is_symlink() or not p.is_file() or '__pycache__' in p.parts:
                continue
            if p.suffix in ('.py', '.sh', '.json', '.toml', '.lock') or p.name == 'spark-python':
                files.append(p)
    files.extend(root / name for name in ('lab', 'config.example.toml', 'spark/pyproject.toml', 'spark/uv.lock', '.gitignore'))
    return sorted(set(files))


def snapshot(root=ROOT):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files(root)}


def deploy(cfg):
    # One local snapshot records BOTH devices, even though only Thor payloads are sent.
    with tempfile.TemporaryDirectory(prefix='cosmos-source-') as tmp:
        local = Path(tmp)
        for p in source_files():
            dest = local / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        hashes = snapshot(local)
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        release_id = f'{stamp}_{digest[:12]}_{time.time_ns() % 1000000:06d}'
        commit = run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True).stdout.strip()
        dirty = bool(run(['git', '-C', ROOT, 'status', '--porcelain', '--', '.'], capture_output=True).stdout)
        manifest = {'id': release_id, 'git_commit': commit, 'dirty': dirty,
                    'source_sha256': hashes, 'snapshot_sha256': digest}
        (local / 'deployment.json').write_text(json.dumps(manifest, indent=2) + '\n')
        archive = ROOT / '.lab' / 'snapshots' / release_id
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local, archive)
        workspace = cfg['thor']['workspace']
        releases = workspace + '/deploy/releases'
        remote = releases + '/' + release_id
        ssh(cfg, ['mkdir', '-p', releases])
        ssh(cfg, ['mkdir', remote])
        # Transfer the whole small code snapshot, never environments, weights or results.
        run(['rsync', '-a', '-e', shlex.join(['ssh', *SSH_OPTIONS]), str(local) + '/',
             cfg['thor']['host'] + ':' + remote + '/'])
        verify = '''import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); m=json.loads((r/'deployment.json').read_text())
for name,digest in m['source_sha256'].items():
 p=r/name
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise RuntimeError('Hash mismatch: '+name)
print(m['id'])'''
        remote_python(cfg, verify, remote)
        ssh(cfg, [workspace + '/.venv/bin/python', '-m', 'compileall', '-q', remote + '/thor'])
        for name in ('serve.py', 'media.py', 'session.py', 'scenarios.py'):
            ssh(cfg, [workspace + '/.venv/bin/python', remote + '/thor/' + name, '--help'], stdout=subprocess.DEVNULL)
        switch = '''import os,pathlib,sys
root=pathlib.Path(sys.argv[1]); target=sys.argv[2]; temp=root/('current-'+pathlib.Path(target).name)
temp.symlink_to(target); os.replace(temp,root/'current')'''
        remote_python(cfg, switch, workspace + '/deploy', remote)
        (ROOT / '.lab' / 'deployed.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'release': remote, 'snapshot_sha256': digest, 'git_commit': commit, 'dirty': dirty}, indent=2))


def environment(cfg):
    env = os.environ.copy()
    workspace = cfg['thor']['workspace']
    env.update(COSMOS_THOR_HOST=cfg['thor']['host'], COSMOS_THOR_WORKSPACE=workspace,
               COSMOS_THOR_CODE=workspace + '/deploy/current/thor',
               COSMOS_THOR_PYTHON=workspace + '/.venv/bin/python',
               COSMOS_THOR_URI=cfg['thor']['uri'],
               COSMOS_SPARK_WORKSPACE=cfg['spark']['workspace'])
    return env


def spark(cfg, args):
    return run([ROOT / 'spark/scripts/spark-python', *args], env=environment(cfg))


def remote_control(cfg, command, *args):
    workspace = cfg['thor']['workspace']
    return ssh(cfg, ['python3', workspace + '/deploy/current/ops/remote_control.py',
                     '--workspace', workspace, command, *args])


def collect(cfg, run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
        raise ValueError('Invalid run ID')
    # Refuse active jobs, since copying live videos yields inconsistent artifacts.
    status = remote_python(cfg, '''import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'deploy/policy.json'
print(p.read_text() if p.exists() else '{}')''', cfg['thor']['workspace'])
    state = json.loads(status)
    if state.get('run') == run_id:
        alive = remote_python(cfg, '''import pathlib,sys
p=pathlib.Path('/proc')/sys.argv[1]/'cmdline'
print(int(p.exists() and sys.argv[2].encode() in p.read_bytes()))''', str(state['pid']), state['entrypoint'])
        if alive == '1':
            raise RuntimeError('Stop this run before collecting its final output')
    source = cfg['thor']['workspace'] + '/outputs/thor_project/' + run_id
    local = Path(cfg['spark']['workspace']) / 'thor_results' / run_id
    if local.exists():
        raise FileExistsError(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local.parent, prefix='.collect-') as tmp:
        run(['rsync', '-a', '-e', shlex.join(['ssh', *SSH_OPTIONS]),
             cfg['thor']['host'] + ':' + source + '/', tmp + '/'])
        remote_hashes = json.loads(remote_python(cfg, '''import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); result={}
for p in r.rglob('*'):
 if p.is_file() and not p.is_symlink():
  with p.open('rb') as f: result[str(p.relative_to(r))]=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(result))''', source))
        local_hashes = {}
        for p in Path(tmp).rglob('*'):
            if p.is_symlink(): raise RuntimeError('Unexpected result symlink')
            if p.is_file():
                with p.open('rb') as f: local_hashes[str(p.relative_to(tmp))] = hashlib.file_digest(f, 'sha256').hexdigest()
        if local_hashes != remote_hashes:
            raise RuntimeError('Collected files differ from remote files')
        shutil.copytree(tmp, local)
    print(local)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'config.local.toml')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('deploy', 'status', 'stop'):
        sub.add_parser(name)
    start = sub.add_parser('start')
    start.add_argument('--run', default=None)
    start.add_argument('--decode-video', action='store_true')
    start.add_argument('--seed', type=int, default=0)
    logs = sub.add_parser('logs')
    logs.add_argument('--follow', action='store_true')
    logs.add_argument('--lines', type=int, default=60)
    c = sub.add_parser('collect'); c.add_argument('run')
    sim = sub.add_parser('run'); sim.add_argument('args', nargs=argparse.REMAINDER)
    py = sub.add_parser('spark'); py.add_argument('args', nargs=argparse.REMAINDER)
    sub.add_parser('check')
    argv = sys.argv[1:]
    for name in ('spark', 'run'):
        if name in argv:
            i = argv.index(name) + 1
            if i < len(argv) and argv[i] != '--': argv.insert(i, '--')
            break
    a = p.parse_args(argv); cfg = config(a.config)
    if a.command == 'deploy': deploy(cfg)
    elif a.command == 'check':
        remote_control(cfg, 'check')
        spark(cfg, ['-c', 'import cosmos_spark; print(cosmos_spark.__file__)'])
    elif a.command == 'start':
        run_id = a.run or 'policy_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        args = ['--run', run_id, '--port', str(cfg['thor'].get('port', 8000)), '--seed', str(a.seed)]
        if a.decode_video: args.append('--decode-video')
        remote_control(cfg, 'start', *args)
    elif a.command in ('status', 'stop'): remote_control(cfg, a.command)
    elif a.command == 'logs':
        remote_control(cfg, 'logs', '--lines', str(a.lines), *(['--follow'] if a.follow else []))
    elif a.command == 'collect': collect(cfg, a.run)
    elif a.command == 'spark': spark(cfg, a.args[1:] if a.args[:1] == ['--'] else a.args)
    elif a.command == 'run':
        saved = json.loads((ROOT / '.lab/deployed.json').read_text())
        if snapshot() != saved['source_sha256']:
            raise RuntimeError('Source changed since deployment; run lab deploy before this experiment')
        remote_control(cfg, 'ready', '--snapshot', saved['snapshot_sha256'])
        args = a.args[1:] if a.args[:1] == ['--'] else a.args
        os.environ['COSMOS_DEPLOYMENT_MANIFEST'] = str(ROOT / '.lab/deployed.json')
        spark(cfg, ['-m', 'cosmos_spark.session', '--uri', cfg['thor']['uri'], *args])


if __name__ == '__main__':
    main()
