"""Prepare DROID recorded-motion conditioning for the base Edge world model."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from common import configure, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--sample', type=Path, required=True, help='A droid.py export sample directory')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = configure(a.workspace)
    from cosmos_framework.data.generator.action.utils.pose_utils import build_abs_pose_from_components, pose_abs_to_rel
    metadata = json.loads((a.sample / 'sample.json').read_text())
    data = root / 'data/droid/success'
    meta = pq.read_table(data / 'meta/episodes/chunk-000/file-000.parquet',
        columns=['episode_index', 'data/chunk_index', 'data/file_index'],
        filters=[('episode_index', '=', metadata['episode'])]).to_pylist()[0]
    table = pq.read_table(data / f"data/chunk-{meta['data/chunk_index']:03d}/file-{meta['data/file_index']:03d}.parquet",
        columns=['frame_index', 'observation.state.cartesian_position', 'action.gripper_position'],
        filters=[('episode_index', '=', metadata['episode'])])
    rows = sorted(table.to_pylist(), key=lambda x: x['frame_index'])
    f = metadata['frame']
    states = np.asarray([r['observation.state.cartesian_position'] for r in rows[f:f+17]])
    if len(states) != 17:
        raise ValueError('Need 17 consecutive recorded states')
    poses = build_abs_pose_from_components(states[:, :3], states[:, 3:6], 'euler_xyz')
    # Match NVIDIA DROIDLeRobotDataset midtrain conversion exactly.
    camera_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    poses[:, :3, :3] = poses[:, :3, :3] @ camera_rotation
    relative = pose_abs_to_rel(poses, rotation_format='rot6d', pose_convention='backward_framewise')
    gripper = 1.0 - np.asarray([r['action.gripper_position'] for r in rows[f:f+16]])[:, None]
    actions = np.concatenate([relative, gripper], axis=-1)
    if actions.shape != (16, 10) or not np.isfinite(actions).all():
        raise ValueError(f'Invalid converted action: {actions.shape}')
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / 'recorded_motion.json', actions.tolist())
    write_json(a.output / 'manifest.json', {'cases': [{
        'id': 'droid_recorded_motion', 'mode': 'forward_dynamics',
        'source': str((a.sample / 'observation.png').resolve()),
        'prompt': metadata['prompt'], 'action_path': 'recorded_motion.json',
        'domain_name': 'droid_lerobot', 'action_chunk_size': 16, 'fps': 15,
        'view_point': 'concat_view', 'resolution': '480',
        'action_semantics': '10D: measured end-effector relative motion, backward_framewise, camera rotation, rot6d, flipped gripper; no normalization. This is recorded-motion conditioning, NOT 8D policy joint commands.',
        'reference_video': str((a.sample / 'recorded.mp4').resolve()),
        'reference_frames': 17, 'seed': 0}]})
    print(a.output / 'manifest.json')

if __name__ == '__main__':
    main()
