"""Render a frozen recorded task state under explicit simulator light intensities."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from cosmos_spark.artifacts import write_json
from cosmos_spark.runner import load_snapshot, snapshot
from isaaclab.app import AppLauncher


def make_comparison(output):
    from PIL import Image, ImageDraw, ImageFont

    result = json.loads((output / 'result.json').read_text())
    cases = result['cases']
    canvas = Image.new('RGB', (480 * len(cases), 614), '#161a22')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    for index, case in enumerate(cases):
        image = Image.open(case['images']['egocentric_mirrored_camera']['path']).convert('RGB')
        x = index * 480
        draw.text((x + 12, 10), f"Light intensity: {case['intensity_scale']:.2f}x", font=font, fill='white')
        canvas.paste(image.resize((480, 267)), (x, 45))
        # Identical crop in each condition, showing the gripper and cube/bowl.
        canvas.paste(image.crop((430, 75, 610, 245)).resize((280, 264)), (x + 100, 318))
    draw.text((12, 590), 'Frozen simulator state; physics and camera poses verified unchanged. Not Cosmos-generated images.', font=small, fill='white')
    canvas.save(output / 'lighting_comparison.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0, 0.7, 0.25])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not all(np.isfinite(value) and value > 0 for value in args.scales):
        parser.error('Light intensity scales must be positive and finite')
    args.output.mkdir(parents=True, exist_ok=False)
    args.enable_cameras = True
    app = AppLauncher(args).app
    env = None
    try:
        import omni.usd
        from pxr import UsdLux
        from robolab.constants import set_output_dir
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.core.replay import apply_recorded_env_cfg
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        source = args.episode.resolve()
        run = json.loads((source.parent / 'run.json').read_text())
        task = run['arguments']['task']
        seed = run['arguments']['seed']
        auto_register_droid_envs(task=task, cameras=WRIST_LEFT_RIGHT_HEAD)
        names = get_envs(task=task)
        assert len(names) == 1
        scene = parse_env_cfg(names[0], device=args.device, seed=seed, num_envs=1, use_fabric=True)
        recorded = json.loads((source / 'env_cfg.json').read_text())
        # Keep registered callable predicates; their stringified addresses cannot
        # be deserialized. No task step or success evaluation is run here.
        recorded.pop('subtasks', None)
        skipped = apply_recorded_env_cfg(scene, recorded)
        assert not [key for key in skipped if not key.startswith('/_')], skipped
        set_output_dir(str(args.output))
        env, cfg = create_env(scene, device=args.device, seed=seed, num_envs=1, policy='diagnostic')
        env.reset_to(load_snapshot(source / 'final_state.npz', env.device), env_ids=None, seed=seed, is_relative=True)
        snapshot(env, args.output / 'restored_state.npz')
        with np.load(source / 'final_state.npz') as expected, np.load(args.output / 'restored_state.npz') as actual:
            restoration = {key: float(np.max(np.abs(expected[key] - actual[key]))) for key in expected.files if expected[key].size}
        assert all(value < 1e-5 for value in restoration.values()), restoration
        stage = omni.usd.get_context().get_stage()
        lights = [(UsdLux.LightAPI(prim).GetIntensityAttr(), prim.GetPath().pathString)
                  for prim in stage.Traverse() if prim.HasAPI(UsdLux.LightAPI)]
        lights = [(attr, path, float(attr.Get())) for attr, path in lights if attr and attr.Get() is not None]
        assert lights, 'No scene lights found'
        cameras = {key: sensor for key, sensor in env.scene.sensors.items() if hasattr(sensor.data, 'output') and 'rgb' in sensor.data.output}
        camera_poses = {key: (sensor.data.pos_w.clone(), sensor.data.quat_w_world.clone()) for key, sensor in cameras.items()}
        records = []
        for index, scale in enumerate(args.scales):
            for attr, _, value in lights:
                attr.Set(value * scale)
            # Render only: no physics integration or policy request is made.
            for _ in range(32):
                env.sim.render()
            paths = {}
            for name, camera in cameras.items():
                camera.update(0.0, force_recompute=True)
                frame = camera.data.output['rgb'][0].detach().cpu().numpy()[..., :3]
                path = args.output / f'light_{index}_{name}.png'
                cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                paths[name] = {'path': str(path), 'mean_rgb': float(frame.mean()),
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                assert (camera.data.pos_w == camera_poses[name][0]).all()
                assert (camera.data.quat_w_world == camera_poses[name][1]).all()
            state_path = args.output / f'light_{index}_state.npz'
            snapshot(env, state_path)
            with np.load(args.output / 'restored_state.npz') as before, np.load(state_path) as after:
                assert all(np.array_equal(before[key], after[key]) for key in before.files)
            records.append({'intensity_scale': scale, 'physics_unchanged': True, 'camera_poses_unchanged': True, 'images': paths})
        write_json(args.output / 'result.json', {
            'source_episode': str(source), 'source_state_sha256': hashlib.sha256((source / 'final_state.npz').read_bytes()).hexdigest(),
            'scope': 'Frozen simulator-state lighting comparison, not a Cosmos-generated video or task/analysis performance test.',
            'restoration_max_abs_errors': restoration,
            'lights': [{'path': path, 'original_intensity': value} for _, path, value in lights], 'cases': records,
        })
        make_comparison(args.output)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
