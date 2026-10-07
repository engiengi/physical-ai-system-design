"""Validated seed contract and checksummed on-disk artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess


def policy_seed(obs, base, request_id):
    if 'policy_seed' not in obs:
        return (base + request_id) % (2**31), 'server_seed_plus_global_request_id'
    value = obs['policy_seed']
    if type(value) is not int or not 0 <= value < 2**31:
        raise ValueError('policy_seed must be an integer in [0, 2147483647]; bool is invalid')
    return value, 'explicit_request_policy_seed'


def file_info(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def video_info(path):
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames:format=duration',
        '-of', 'json', str(path)], text=True))
    s = data['streams'][0]
    n,d = map(int, s['avg_frame_rate'].split('/'))
    return {**file_info(path), 'frames': int(s['nb_read_frames']), 'fps': n/d,
            'width': s['width'], 'height': s['height'], 'duration_seconds': float(data['format']['duration'])}
