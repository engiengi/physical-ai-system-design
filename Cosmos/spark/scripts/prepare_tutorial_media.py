"""Create unlabeled terminal-analysis and lighting-generation inputs from one ended trial."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil


def prepare(episode, output):
    episode = Path(episode).resolve()
    output = Path(output).resolve()
    result = json.loads((episode / 'episode.json').read_text())
    if result.get('status') not in ('success', 'task_failure') or result.get('termination_reason') not in ('success_condition', 'task_time_limit'):
        raise ValueError('Use a naturally ended trial, not a capped diagnostic or execution error.')
    video = episode / 'egocentric_mirrored_camera.mp4'
    if not video.is_file():
        raise FileNotFoundError(video)
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-sseof', '-3', '-i', str(video),
                    '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(output / 'terminal.mp4')], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video), '-frames:v', '1',
                    '-update', '1', str(output / 'initial.png')], check=True)
    context = {'scenario_id': result.get('scenario_id') or 'RecordedTask',
               'experiment_track': result.get('experiment_track') or 'new_standalone',
               'source_type': 'simulation', 'source_provenance': 'User-recorded RoboLab trial',
               'task': result['instruction'],
               'evaluation_criteria': ['All instructed target objects are visibly placed in the instructed container and released from the gripper at trial end.']}
    # Object names refer to the robot scene, not left/right in the mirrored camera.
    if 'hammer' in result['instruction'].lower() and 'left bin' in result['instruction'].lower():
        context['evaluation_criteria'].append('In this egocentric mirrored camera, the instructed left_bin is the BLUE bin on the RIGHT of the image, not the grey bin on the left.')
    cases = []
    for name, source in [('terminal', output / 'terminal.mp4'),
                         ('settled', episode / 'post_terminal/egocentric_mirrored_camera.mp4')]:
        if not source.is_file():
            continue
        if name == 'settled':
            shutil.copyfile(source, output / 'settled.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(output / (name + '.mp4')),
                        '-vf', 'reverse', '-frames:v', '1', '-update', '1',
                        str(output / (name + '.png'))], check=True)
        for kind, suffix in [('video', '.mp4'), ('still', '.png')]:
            cases.append({**context, 'id': name + '_' + kind, 'source': name + suffix,
                          'fps': 4, 'seed': 0, 'pair_id': episode.parent.name,
                          'variant': name + '_' + kind,
                          'evaluation_phase': 'terminal_outcome', 'trial_ended': True})
    analysis = {'criteria_version': 'tutorial-terminal-v2-released-and-camera-target', 'cases': cases}
    generation = {'cases': [{**context, 'id': 'lighting', 'source': 'initial.png', 'mode': 'i2v',
                            'seed': 21, 'num_frames': 65,
                            'prompt': 'Keep the same camera, robot, objects, positions and geometry. Only make the room lighting slightly dimmer. No new objects. No object movement.'}]}
    for name, obj in [('analysis.json', analysis), ('generation.json', generation)]:
        (output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    # Reference outcomes are never part of the inference manifest.
    reference = {k: result.get(k) for k in ('status', 'termination_reason', 'steps',
                                           'post_terminal_observation')}
    (output / 'reference.json').write_text(json.dumps(reference, ensure_ascii=False, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.episode, args.output)
