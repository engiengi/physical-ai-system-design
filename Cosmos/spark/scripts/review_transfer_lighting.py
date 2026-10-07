"""Compare a static control video with a generated lighting variation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames or fps <= 0 or (expected > 0 and len(frames) != expected):
        raise ValueError(f'Incomplete or invalid video: {path}')
    return frames, fps


def panel(frame, title, crop, width=512):
    h, w = frame.shape[:2]
    full_h = round(h * width / w)
    x0, y0, x1, y1 = crop
    region = frame[round(y0*h):round(y1*h), round(x0*w):round(x1*w)]
    crop_h = round(region.shape[0] * width / region.shape[1])
    out = np.full((full_h + crop_h + 76, width, 3), 24, np.uint8)
    cv2.putText(out, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 1, cv2.LINE_AA)
    out[40:40+full_h] = cv2.resize(frame, (width, full_h), interpolation=cv2.INTER_AREA)
    cv2.putText(out, 'Same fixed crop: gripper / cube / bowl', (12, 40+full_h+25),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (210, 210, 210), 1, cv2.LINE_AA)
    out[full_h+76:] = cv2.resize(region, (width, crop_h), interpolation=cv2.INTER_CUBIC)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='Transfer 2.5')
    parser.add_argument('--crop', type=float, nargs=4, default=(.50, .12, .72, .52))
    args = parser.parse_args()
    label = 'SELF-CHECK: input reused' if args.control.samefile(args.generated) else args.label
    x0, y0, x1, y1 = args.crop
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        parser.error('--crop must be normalized x0 y0 x1 y1 within [0, 1]')
    control, fps = read_video(args.control)
    generated, generated_fps = read_video(args.generated)
    if len(control) != len(generated) or abs(fps-generated_fps) > .001:
        raise ValueError('Control and generated video timing must match')
    if any(f.shape != control[0].shape for f in control + generated):
        raise ValueError('All frames must have the same dimensions')
    args.output.mkdir(parents=True, exist_ok=False)
    h, w = control[0].shape[:2]
    if round(x0*w) == round(x1*w) or round(y0*h) == round(y1*h):
        raise ValueError('Crop is smaller than one pixel')
    n = len(generated)
    samples = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    for index in samples:
        cv2.imwrite(str(args.output/f'generated_{index:03d}.png'), generated[index])
    cv2.imwrite(str(args.output/'control_first.png'), control[0])
    still = np.hstack([
        panel(control[0], 'Original static control', args.crop),
        panel(generated[n//2], f'{label}: {n//2/fps:.2f}s', args.crop),
        panel(generated[-1], f'{label}: {(n-1)/fps:.2f}s', args.crop),
    ])
    cv2.imwrite(str(args.output/'comparison.png'), still)
    contact = np.hstack([panel(generated[i], f'Generated: {i/fps:.2f}s', args.crop, 320) for i in samples])
    cv2.imwrite(str(args.output/'temporal_contact.png'), contact)
    first = np.hstack([panel(control[0], 'Original static control', args.crop),
                       panel(generated[0], label, args.crop)])
    vh, vw = first.shape[:2]
    video = args.output/'comparison.mp4'
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{vw}x{vh}', '-r', str(fps), '-i', '-', '-an', '-vf',
           'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for i, frame in enumerate(generated):
            pair = np.hstack([panel(control[i], 'Original static control', args.crop),
                              panel(frame, f'{label}: {i/fps:.2f}s', args.crop)])
            proc.stdin.write(pair.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('Comparison video encoding failed')
    review_frames, review_fps = read_video(video)
    assert len(review_frames) == n and abs(review_fps-fps) < .001
    means = [float(f.mean()) for f in generated]
    control_means = [float(f.mean()) for f in control]
    audit = {
        'control': str(args.control.resolve()), 'generated': str(args.generated.resolve()),
        'label': label,
        'control_sha256': hashlib.sha256(args.control.read_bytes()).hexdigest(),
        'generated_sha256': hashlib.sha256(args.generated.read_bytes()).hexdigest(),
        'frames': n, 'fps': fps, 'duration_seconds': n/fps, 'size': [w, h],
        'crop_normalized': args.crop, 'sampled_frame_indices': samples,
        'control_mean_rgb': control_means, 'generated_mean_rgb': means,
        'generated_mean_rgb_range': [min(means), max(means)],
        'generated_mean_rgb_over_video': float(np.mean(means)),
        'control_mean_rgb_over_video': float(np.mean(control_means)),
        'interpretation': 'Decoded RGB means are not illuminance or proof of shape preservation. '
                          'Inspect all frames and fixed crops. This is a static-scene test, not a motion benchmark.',
    }
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if not isinstance(v,list) or len(v)<6}, indent=2))


if __name__ == '__main__':
    main()
