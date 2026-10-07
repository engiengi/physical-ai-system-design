"""Export real DROID observations and replay them through the policy server."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from common import pack, unpack, validate_observation, write_json

VIEWS = ['wrist_image_left', 'exterior_image_1_left', 'exterior_image_2_left']


def frames(path, seconds, count=1):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(seconds), '-i', str(path),
        '-frames:v', str(count), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', 'pipe:1'])
    video = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 360, 640, 3)
    if len(video) != count:
        raise ValueError(f'Expected {count} frames, got {len(video)}')
    return video


def canvas(views):
    import cv2
    return np.concatenate([views[0], np.concatenate([cv2.resize(v, (320, 180)) for v in views[1:]], axis=1)], axis=0)


def export(a):
    import pyarrow.parquet as pq
    import imageio.v3 as iio
    root = a.dataset / 'success'
    meta_cols = ['episode_index', 'tasks', 'length', 'data/chunk_index', 'data/file_index']
    for v in VIEWS:
        meta_cols += [f'videos/observation.image.{v}/{s}' for s in ['chunk_index', 'file_index', 'from_timestamp']]
    episodes = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet', columns=meta_cols).to_pylist()
    ep = next(e for e in episodes if e['episode_index'] == a.episode)
    path = root / f"data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet"
    columns = ['observation.state.joint_positions', 'observation.state.gripper_position',
               'action.joint_position', 'action.gripper_position', 'frame_index', 'timestamp']
    rows = pq.read_table(path, columns=columns, filters=[('episode_index', '=', a.episode)]).to_pylist()
    rows = sorted(rows, key=lambda r: r['frame_index'])
    if any(f < 0 or f + 32 >= len(rows) for f in a.frames):
        raise ValueError('Each requested frame requires 32 following frames in the episode')
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = {'dataset': 'nvidia/Cosmos3-DROID', 'revision': json.loads((a.dataset / 'download.json').read_text())['revision'],
                'episode_index': a.episode, 'fps': 15, 'samples': [],
                'evaluation': 'recorded observations; actions are NOT executed; no closed-loop success claim'}
    for f in a.frames:
        row = rows[f]
        clips = []
        obs = {'prompt': ep['tasks'][0].split('|')[0].strip(),
               'session_id': f'droid-episode-{a.episode}-frame-{f}',
               'observation/joint_position': np.asarray(row['observation.state.joint_positions'], dtype=np.float32),
               'observation/gripper_position': np.float32(row['observation.state.gripper_position'])}
        name = f'episode-{a.episode:04d}-frame-{f:04d}'
        dest = a.output / name
        dest.mkdir()
        for view in VIEWS:
            prefix = f'videos/observation.image.{view}'
            path = root / f"{prefix}/chunk-{ep[prefix + '/chunk_index']:03d}/file-{ep[prefix + '/file_index']:03d}.mp4"
            clips.append(frames(path, ep[prefix + '/from_timestamp'] + row['timestamp'], 33))
            obs[f'observation/{view}'] = clips[-1][0].copy()
        obs['observation/image'] = canvas([c[0] for c in clips])
        validate_observation(obs)
        (dest / 'observation.msgpack').write_bytes(pack(obs))
        reference = np.asarray([r['action.joint_position'] + [r['action.gripper_position']] for r in rows[f:f + 32]], dtype=np.float32)
        np.save(dest / 'reference_action.npy', reference)
        video = np.stack([canvas([c[t] for c in clips]) for t in range(33)])
        iio.imwrite(dest / 'recorded.mp4', video, fps=15, macro_block_size=1)
        iio.imwrite(dest / 'observation.png', video[0])
        write_json(dest / 'sample.json', {'prompt': obs['prompt'], 'frame': f, 'timestamp': row['timestamp'],
            'episode': a.episode, 'reference_action_range': [f, f + 32],
            'reference_action_semantics': 'dataset action.joint_position radians + action.gripper_position; raw convention'})
        manifest['samples'].append(name)
    write_json(a.output / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def replay(a):
    from websockets.sync.client import connect
    manifest = json.loads((a.inputs / 'manifest.json').read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    headers = {'Authorization': 'Bearer ' + os.environ['COSMOS_POLICY_TOKEN']} if os.environ.get('COSMOS_POLICY_TOKEN') else {}
    with connect(a.uri, compression=None, max_size=64 * 1024**2, additional_headers=headers) as ws:
        metadata = unpack(ws.recv(timeout=a.timeout))
        write_json(a.output / 'server.json', metadata)
        results = []
        for name in manifest['samples']:
            source = a.inputs / name
            dest = a.output / name
            dest.mkdir()
            started = time.perf_counter()
            try:
                ws.send((source / 'observation.msgpack').read_bytes())
                result = unpack(ws.recv(timeout=a.timeout))
                elapsed = time.perf_counter() - started
                if result.get('type') == 'error':
                    raise RuntimeError(result['message'])
                action = np.asarray(result['action'])
                if action.shape != (32, 8) or not np.isfinite(action).all():
                    raise ValueError('Invalid action response')
                np.save(dest / 'action.npy', action)
                reference = np.load(source / 'reference_action.npy', allow_pickle=False)
                row = {'sample': name, 'status': 'success', 'round_trip_seconds': elapsed,
                       'joint_mae_rad': float(np.abs(action[:, :7] - reference[:, :7]).mean()),
                       'gripper_mae': float(np.abs(action[:, 7] - reference[:, 7]).mean()),
                       'server_request_id': result['request_id'], 'server_run_id': result['run_id'],
                       'server_timing': result['server_timing']}
            except Exception as e:
                write_json(dest / 'result.json', {'status': 'failed', 'error': str(e)})
                raise
            write_json(dest / 'result.json', row)
            results.append(row)
            print(json.dumps(row), flush=True)
    write_json(a.output / 'summary.json', {'evaluation': 'offline recorded DROID observations; no robot control or held-out generalization claim',
        'dataset_revision': manifest['revision'], 'samples': results,
        'median_round_trip_seconds': float(np.median([r['round_trip_seconds'] for r in results]))})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--dataset', type=Path, required=True)
    e.add_argument('--episode', type=int, default=0)
    e.add_argument('--frames', type=int, nargs='+', default=[30, 150, 300])
    e.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('replay')
    r.add_argument('--inputs', type=Path, required=True)
    r.add_argument('--uri', default='ws://127.0.0.1:8000')
    r.add_argument('--timeout', type=float, default=180)
    r.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    export(a) if a.command == 'export' else replay(a)

if __name__ == '__main__':
    main()
