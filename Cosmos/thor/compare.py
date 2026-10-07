"""Render a time-aligned recorded/predicted video comparison for review."""
import argparse
import json
from pathlib import Path
import subprocess
from common import write_json
from artifacts import file_info


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames:format=duration', '-of', 'json', str(path)]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, help='Optional prediction_audit JSON, preserved with comparison')
    p.add_argument('--recorded', type=Path, required=True)
    p.add_argument('--predicted', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, required=True)
    p.add_argument('--fps', type=float, default=15)
    p.add_argument('--kind', choices=['recorded-motion-conditioned', 'joint-policy-prediction'], required=True)
    p.add_argument('--alignment-note', required=True, help='Camera/time/action alignment and remaining differences')
    a = p.parse_args()
    if a.frames < 2 or a.fps <= 0:
        p.error('At least two frames and positive FPS required')
    for path in [a.recorded, a.predicted]:
        if not path.is_file(): raise FileNotFoundError(path)
    audit_data=json.loads(a.audit.read_text()) if a.audit else None
    if audit_data:
        if file_info(a.predicted)['sha256']!=audit_data['predicted']['sha256'] or file_info(a.recorded)['sha256']!=audit_data['actual']['sha256']:
            raise ValueError('Audit belongs to different video files')
    metadata = [probe(path) for path in [a.recorded, a.predicted]]
    if any(float(m['format']['duration']) + 1e-3 < a.frames / a.fps for m in metadata):
        raise ValueError('A video is shorter than the requested comparison interval')
    a.output.mkdir(parents=True, exist_ok=False)
    filters = []
    for i, label in enumerate(['RECORDED', 'PREDICTED']):
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,fps={a.fps},scale=640:540:force_original_aspect_ratio=decrease,pad=640:580:(ow-iw)/2:40,setsar=1,drawtext=text={label}:x=12:y=8:fontsize=24:fontcolor=white[v{i}]')
    filters.append('[v0][v1]hstack=inputs=2:shortest=1[v]')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(a.recorded), '-i', str(a.predicted),
        '-filter_complex', ';'.join(filters), '-map', '[v]', '-frames:v', str(a.frames),
        '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(a.output / 'comparison.mp4')], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', str((a.frames-1)/a.fps), '-i', str(a.output/'comparison.mp4'),
        '-frames:v', '1', str(a.output / 'last_frame.png')], check=True)
    write_json(a.output / 'comparison.json', {'recorded': str(a.recorded.resolve()),
        'predicted': str(a.predicted.resolve()), 'frames': a.frames, 'fps': a.fps,
        'audit':audit_data, 'kind': a.kind, 'alignment_note': a.alignment_note, 'media': metadata,
        'note': 'Visual comparison only; preserves aspect ratios. Prediction padding/cropping may differ. No pixel metric or closed-loop success claim.'})

if __name__ == '__main__':
    main()
