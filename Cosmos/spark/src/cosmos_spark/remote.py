"""Explicit, bounded transfers and recorder control in the authorized Thor workspace."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess

from .settings import HOST, WORKSPACE
ROOT = Path(__file__).resolve().parents[2]


def ssh(argv, *, timeout=60):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', HOST,
                           shlex.join([str(x) for x in argv])], capture_output=True, text=True,
                          check=True, timeout=timeout).stdout


def remote_path(value):
    # Also safe with legacy SCP implementations interpreting the remote name as shell text.
    if not isinstance(value, str) or not re.fullmatch(r'/[A-Za-z0-9_./+-]+', value):
        raise ValueError('Invalid Thor artifact path')
    p = PurePosixPath(value)
    if '..' in p.parts or not str(p).startswith(WORKSPACE + '/'):
        raise ValueError('Artifact must be within Thor cosmos_ws')
    return str(p)


def fetch_file(source, target, expected_sha256=None):
    source = remote_path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(target.name + '.partial')
    try:
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                        HOST + ':' + source, str(temporary)], check=True, timeout=180)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('Thor artifact checksum mismatch')
        temporary.replace(target)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


class ThorRecording:
    def __init__(self, run_id, policy_run, destination):
        if not all(re.fullmatch(r'[A-Za-z0-9_+-]+', x) for x in (run_id, policy_run)):
            raise ValueError('Unsafe recording or policy run ID')
        self.remote = WORKSPACE + '/captures/' + run_id
        self.policy_run = policy_run
        self.destination = Path(destination)
        self.helper = WORKSPACE + '/spark_capture_tools/record_thor_desktop.py'

    def start(self):
        # Separate helper location: never replaces Thor's server, media code or original helper.
        ssh(['mkdir', '-p', WORKSPACE + '/spark_capture_tools'])
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', str(ROOT/'scripts/record_thor_desktop.py'),
                        HOST + ':' + self.helper], check=True, timeout=30)
        return json.loads(ssh(['python3', self.helper, 'start', '--output', self.remote,
                              '--policy-run', self.policy_run]))

    def stop(self):
        record = json.loads(ssh(['python3', self.helper, 'stop', '--output', self.remote,
                                '--policy-run', self.policy_run], timeout=180))
        for name in ('thor_recording.json', 'thor_screen.mp4'):
            fetch_file(self.remote + '/' + name, self.destination / name)
        return record
