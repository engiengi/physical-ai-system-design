"""Prepare static-scene controls for the official Cosmos-Transfer2.5 lighting recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--prompt', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.image.resolve()
    video = (args.output / 'static_control.mp4').resolve()
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-loop', '1', '-framerate', '16',
                    '-i', str(source), '-frames:v', '93', '-c:v', 'libx264', '-crf', '16',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
    spec = {'name': args.name, 'video_path': str(video), 'prompt': args.prompt,
            'guidance': 3, 'edge': {'control_weight': 1.0}, 'vis': {'control_weight': 0.2},
            'resolution': '720', 'num_video_frames_per_chunk': 93, 'max_frames': 93,
            'num_steps': 35, 'seed': 41, 'keep_input_resolution': True}
    (args.output / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    (args.output / 'provenance.json').write_text(json.dumps({
        'source_image': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'control_video': str(video), 'control_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
        'control_kind': '93 repeated frames from one image; not a physical simulation rollout',
        'purpose': 'Preserve a static scene while changing illumination',
        'recipe': 'https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer2_5/inference-real-augmentation/inference.html',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
