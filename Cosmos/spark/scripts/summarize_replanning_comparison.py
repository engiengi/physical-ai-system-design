"""Audit completed horizon trials and render a comparison from recorded cameras."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
CAMERAS = ['egocentric_mirrored_camera', 'over_shoulder_right_camera']


def read_json(path):
    return json.loads(path.read_text())


def audit(episodes, report):
    import h5py
    from robolab.eval.websocket_transport import MsgPackNumpy

    packer = MsgPackNumpy()
    rows = []
    with np.load(episodes[0] / 'initial_state.npz') as data:
        reference_state = dict(data)
    with np.load(episodes[0] / 'request_00000_observation.npz') as data:
        reference_observation = dict(data)
    reference_policy = None
    for ep in episodes:
        result = read_json(ep / 'episode.json')
        horizon = result['horizon']
        policy = read_json(report / f'h{horizon}_runtime.json')['policy']
        if reference_policy is None:
            reference_policy = policy
        requests = [json.loads(line) for line in (ep / 'requests.jsonl').read_text().splitlines()]
        applied = [json.loads(line) for line in (ep / 'applied_actions.jsonl').read_text().splitlines()]
        assert len(requests) == result['requests']
        assert len(applied) == result['steps']
        tail_only = 0
        chunk_lengths = []
        seeds = []
        for index, request in enumerate(requests):
            action = np.load(ep / f"request_{request['id']:05d}_actions.npy").reshape(-1, 8)
            assert action.shape == (32, 8) and np.isfinite(action).all()
            trace = report / ('policy_' + ep.parent.name) / 'requests' / f"{request['id']:06d}"
            remote_observation = packer.unpack((trace / 'observation.msgpack').read_bytes())
            with np.load(ep / f"request_{request['id']:05d}_observation.npz") as local:
                for key, remote_key in [('image', 'observation/image'), ('joint', 'observation/joint_position'), ('gripper', 'observation/gripper_position')]:
                    assert np.array_equal(local[key], remote_observation[remote_key])
            assert np.array_equal(action, np.load(trace / 'action.npy'))
            assert remote_observation['prompt'] == request['prompt'] == result['instruction']
            seeds.append(read_json(trace / 'result.json')['seed'])
            end = requests[index + 1]['step'] if index + 1 < len(requests) else len(applied)
            count = end - request['step']
            assert 0 < count <= horizon <= len(action)
            assert index == len(requests) - 1 or count == horizon
            expected = action[:count].copy()
            expected[:, -1] = expected[:, -1] > 0.5
            actual = np.asarray([row['action'] for row in applied[request['step']:end]]).reshape(-1, 8)
            assert np.allclose(actual, expected, atol=1e-6, rtol=0), 'Applied actions do not match the response prefix'
            close = action[:, -1] > 0.5
            tail_only += int(count == horizon and not close[:horizon].any() and close[horizon:].any())
            chunk_lengths.append(count)
        assert seeds == np.random.default_rng(1200).integers(0, 2**31, size=len(seeds)).tolist()
        native = [json.loads(line) for line in (ROOT / 'RoboLab/output' / ep.parent.name / 'episode_results.jsonl').read_text().splitlines()]
        assert len(native) == 1 and native[0]['episode_step'] == result['steps']
        assert native[0]['success'] == (result['status'] == 'success')
        with np.load(ep / 'initial_state.npz') as data:
            same_keys = set(data.files) == set(reference_state)
            exact_state = same_keys and all(np.array_equal(data[k], reference_state[k]) for k in data.files)
        with np.load(ep / 'request_00000_observation.npz') as data:
            delta = np.abs(data['image'].astype(float) - reference_observation['image'].astype(float))
            obs_audit = {'image_exact': bool(not delta.any()), 'image_mean_absolute_difference': float(delta.mean()),
                         'image_max_absolute_difference': float(delta.max()),
                         'joint_exact': bool(np.array_equal(data['joint'], reference_observation['joint'])),
                         'gripper_exact': bool(np.array_equal(data['gripper'], reference_observation['gripper']))}
        latency = np.asarray([r['seconds'] for r in requests])
        object_motion = {}
        trajectory = ROOT / 'RoboLab/output' / ep.parent.name / 'ToolOrganizationTask/run_0.hdf5'
        with h5py.File(trajectory, 'r') as data:
            for name in ['red_hammer', 'husky_hammer', 'cordless_drill', 'spring_clamp']:
                positions = data[f'data/demo_0/states/rigid_object/{name}/root_pose'][:, :3]
                assert len(positions) == result['steps']
                object_motion[name] = {
                    'final_xyz': positions[-1].tolist(),
                    'maximum_distance_from_first_recorded_position_m': float(np.linalg.norm(positions - positions[0], axis=1).max()),
                    'minimum_z_m': float(positions[:, 2].min()),
                    'maximum_z_m': float(positions[:, 2].max()),
                }
        rows.append({**result, 'policy_settings_equal': policy == reference_policy,
                     'initial_physics_exact': exact_state, 'initial_observation_vs_first': obs_audit,
                     'applied_prefix_verified': True, 'remote_traces_and_seed_sequence_verified': True,
                     'native_evaluator_verified': True, 'applied_chunk_lengths': sorted(set(chunk_lengths)),
                     'requests_close_only_in_unexecuted_tail': tail_only,
                     'request_mean_s': float(latency.mean()), 'request_median_s': float(np.median(latency)),
                     'request_total_s': float(latency.sum()),
                     'recorded_object_motion': object_motion,
                     'planned_sim_seconds_per_request': horizon * result['simulated_elapsed_s'] / result['steps']})
    (report / 'audit.json').write_text(json.dumps({
        'scope': 'One trajectory per horizon, not a success-rate or causal benchmark. Inspect final camera views separately.',
        'rows': rows,
    }, indent=2) + '\n')
    return rows


def final_montage(episodes, rows, report):
    width, height = 480, 268
    canvas = Image.new('RGB', (width * len(rows), height * 2 + 117), '#161a22')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 16)
    for col, (ep, row) in enumerate(zip(episodes, rows)):
        x = col * width
        draw.text((x + 12, 10), f"Execute {row['horizon']}/32 actions, then observe", font=font, fill='white')
        draw.text((x + 12, 40), f"Final state at {row['simulated_elapsed_s']:.2f}s simulation time", font=small, fill='#d0d4db')
        for panel, camera in enumerate(CAMERAS):
            frame = Image.open(ep / (camera + '_final.png')).convert('RGB').resize((width, height))
            canvas.paste(frame, (x, 65 + panel * height))
    draw.text((12, height * 2 + 71), 'One trial per condition. Separate trial in each column.', font=small, fill='white')
    draw.text((12, height * 2 + 94), 'Top: front camera. Bottom: side camera.', font=small, fill='white')
    canvas.save(report / 'final_comparison.png')


def execution_schedule(rows, report):
    canvas = Image.new('RGB', (1100, 160 + 110 * len(rows)), '#161a22')
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(FONT, 26)
    font = ImageFont.truetype(FONT, 20)
    draw.text((25, 20), 'When should the robot observe again?', font=title, fill='white')
    draw.text((25, 65), 'Each reply contains 32 actions. Cyan: execute. Gray: discard and request a new plan.', font=font, fill='#d0d4db')
    for index, row in enumerate(rows):
        y = 125 + index * 110
        h = row['horizon']
        draw.text((25, y), f"{h}/32 actions: observe after {row['planned_sim_seconds_per_request']:.2f}s of simulation", font=font, fill='white')
        for action in range(32):
            x = 25 + action * 32
            draw.rectangle((x, y + 35, x + 27, y + 70), fill='#38bdf8' if action < h else '#414957')
    draw.text((25, canvas.height - 35), 'Conceptual schedule. Inference waiting time is excluded from simulation time.', font=font, fill='#d0d4db')
    canvas.save(report / 'execution_schedule.png')


def comparison_video(episodes, rows, report):
    # Full recordings share simulation time, not real waiting time. Successful
    # early termination is explicitly labelled before padding the last frame.
    duration = max(row['simulated_elapsed_s'] * (1 + 1 / row['steps']) for row in rows)
    inputs, filters = [], []
    for col, (ep, row) in enumerate(zip(episodes, rows)):
        for camera in CAMERAS:
            inputs.extend(['-i', str(ep / (camera + '.mp4'))])
        for panel in range(2):
            i = col * 2 + panel
            filters.append(f'[{i}:v]setpts=PTS-STARTPTS,scale=480:268,tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration}[v{i}]')
        title = f"Execute {row['horizon']}/32 actions, then observe"
        end = row['simulated_elapsed_s']
        filters.append(f"[v{col * 2}][v{col * 2 + 1}]vstack=inputs=2,pad=480:596:0:60:color=0x161a22,drawtext=fontfile={FONT}:text='{title}':x=10:y=8:fontsize=21:fontcolor=white,drawtext=fontfile={FONT}:text='4x simulation time; NOT real-time speed':x=10:y=35:fontsize=16:fontcolor=white,drawtext=fontfile={FONT}:text='Trial ended - final frame held':x=10:y=76:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.8:enable='gte(t,{end})'[c{col}]")
    columns = ''.join(f'[c{i}]' for i in range(len(rows)))
    stack = f'hstack=inputs={len(rows)},' if len(rows) > 1 else ''
    filters.append(f'{columns}{stack}setpts=PTS/4,fps=15[out]')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-filter_complex_threads', '2',
                    *inputs, '-filter_complex', ';'.join(filters), '-map', '[out]',
                    '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '25', '-maxrate', '700k',
                    '-bufsize', '1400k', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(report / 'horizon_comparison_4x.mp4')], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    report = ROOT / 'reports' / args.batch
    status = read_json(report / 'status.json')
    if status['phase'] != 'complete':
        raise RuntimeError('All trials must complete before comparison')
    episodes = [ROOT / 'runs' / f'{args.batch}_h{h}' / 'episode_0000' for h in status['horizons']]
    rows = audit(episodes, report)
    final_montage(episodes, rows, report)
    execution_schedule(rows, report)
    if args.video:
        comparison_video(episodes, rows, report)


if __name__ == '__main__':
    main()
