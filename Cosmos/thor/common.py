"""Local runtime and OpenPI-compatible transport utilities."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import msgpack
import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def configure(workspace):
    root = Path(workspace).expanduser().resolve()
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ['PATH'] = str(root / '.venv/bin') + os.pathsep + os.environ.get('PATH', '')
    for key in ['TRITON_PTXAS_PATH', 'TRITON_PTXAS_BLACKWELL_PATH']:
        if Path('/usr/local/cuda-13.0/bin/ptxas').is_file():
            os.environ.setdefault(key, '/usr/local/cuda-13.0/bin/ptxas')
    framework = root / 'external/cosmos-framework'
    if not framework.is_dir():
        raise FileNotFoundError(framework)
    sys.path.insert(0, str(framework))
    return root


def gpu_lock(root):
    handle = (root / '.thor-experiment.lock').open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Another Thor experiment owns the GPU lock. Stop its server first.')
    return handle


def new_run(root, name):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', name):
        raise ValueError('run must contain only letters, digits, _ or -')
    path = root / 'outputs/thor_project' / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def provenance(root):
    import torch
    source = Path(__file__).parent
    deployment = source.parent / 'deployment.json'
    return {'deployment': json.loads(deployment.read_text()) if deployment.exists() else None,
            'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(),
            'framework_revision': revision(root / 'external/cosmos-framework'),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(source.glob('*.py'))},
            'recorded_at_unix': time.time()}


def _encode(x):
    if isinstance(x, np.ndarray):
        if x.dtype.kind not in 'buif':
            raise ValueError('Unsupported array dtype')
        return {b'__ndarray__': True, b'data': x.tobytes(), b'dtype': x.dtype.str, b'shape': x.shape}
    if isinstance(x, np.generic):
        return {b'__npgeneric__': True, b'data': x.item(), b'dtype': x.dtype.str}
    raise TypeError(type(x).__name__)


def _decode(x):
    if b'__ndarray__' in x or b'__npgeneric__' in x:
        dtype = np.dtype(x[b'dtype'])
        if dtype.kind not in 'buif':
            raise ValueError('Unsupported array dtype')
        if b'__ndarray__' in x:
            return np.frombuffer(x[b'data'], dtype=dtype).reshape(tuple(x[b'shape'])).copy()
        return dtype.type(x[b'data'])
    return x


def pack(x):
    return msgpack.packb(x, default=_encode)


def unpack(x):
    return msgpack.unpackb(x, object_hook=_decode, strict_map_key=False)


def validate_observation(obs):
    if not isinstance(obs, dict):
        raise ValueError('Observation must be a dictionary')
    if not isinstance(obs.get('prompt'), str) or not 0 < len(obs['prompt']) <= 8192:
        raise ValueError('A nonempty prompt is required')
    keys = ['observation/image'] if 'observation/image' in obs else [
        'observation/wrist_image_left', 'observation/exterior_image_1_left',
        'observation/exterior_image_2_left']
    for key in keys:
        a = np.asarray(obs.get(key))
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 2 or max(a.shape[:2]) > 2160:
            raise ValueError(f'{key} must be a uint8 RGB image, sides 2..2160')
    joint = np.asarray(obs.get('observation/joint_position'), dtype=np.float32)
    if joint.shape not in [(7,), (1, 7)] or not np.isfinite(joint).all():
        raise ValueError('joint_position must contain 7 finite absolute joint positions')
    grip = np.asarray(obs.get('observation/gripper_position'), dtype=np.float32)
    if grip.size != 1 or grip.ndim > 2 or not np.isfinite(grip).all() or not ((grip >= 0) & (grip <= 1)).all():
        raise ValueError('gripper_position must be one finite value in [0,1]')
    return obs
