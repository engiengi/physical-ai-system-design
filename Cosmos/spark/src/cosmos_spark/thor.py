"""Control the deployed Thor runtime through the shared lifecycle manager."""
import argparse
import re
import shlex
import subprocess

from .settings import HOST, WORKSPACE, CODE, PYTHON


def ssh(command, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', HOST,
                           shlex.join([str(x) for x in command])], check=True, text=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['status', 'start-policy', 'stop-policy', 'analyze', 'generate'])
    p.add_argument('--run')
    p.add_argument('--manifest')
    p.add_argument('--decode-video', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--prompt-version', choices=['v1', 'plain-json-v2'], default='plain-json-v2')
    a = p.parse_args()
    if a.command in ('start-policy', 'analyze', 'generate'):
        if not a.run or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', a.run):
            p.error('A unique safe --run name is required')
    if a.command in ('status', 'start-policy', 'stop-policy'):
        operation = {'status': 'status', 'start-policy': 'start', 'stop-policy': 'stop'}[a.command]
        command = ['python3', WORKSPACE + '/deploy/current/ops/remote_control.py', '--workspace', WORKSPACE, operation]
        if operation == 'start':
            command += ['--run', a.run, '--seed', str(a.seed)]
            if a.decode_video: command.append('--decode-video')
        ssh(command)
        return
    if not a.manifest or not a.manifest.startswith(WORKSPACE + '/'):
        p.error('--manifest must be an absolute path inside the Thor workspace')
    release_code = ssh(['readlink', '-f', CODE], capture_output=True).stdout.strip()
    command = [PYTHON, '-u', release_code + '/media.py', a.command, '--workspace', WORKSPACE,
               '--manifest', a.manifest, '--run', a.run]
    if a.command == 'analyze': command += ['--prompt-version', a.prompt_version]
    ssh(command)


if __name__ == '__main__':
    main()
