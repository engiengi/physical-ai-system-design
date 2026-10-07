"""Audit a saved prediction/action execution pair without claiming visual accuracy."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from common import write_json
from artifacts import file_info, video_info
from scenarios import context


def audit(spec, base):
    ctx=context(spec)
    def path(key):
        p=Path(spec[key]);return p if p.is_absolute() else base/p
    manifest=json.loads(path('prediction_manifest').read_text())
    raw=np.load(path('raw_action'),allow_pickle=False)
    applied=np.load(path('applied_action'),allow_pickle=False)
    if raw.ndim!=2 or raw.shape[1]!=8 or applied.ndim!=2 or applied.shape[1]!=8:
        raise ValueError('Expected 7 absolute joint positions plus gripper')
    if not np.isfinite(raw).all() or not np.isfinite(applied).all(): raise ValueError('Nonfinite action')
    count=len(applied)
    if not 0<count<=len(raw):raise ValueError('Applied length must be within the predicted chunk')
    expected=raw[:count].copy()
    g=spec['gripper_transform']
    if g['kind']=='threshold':
        if g['threshold']!=0.5:raise ValueError('DROID threshold must be 0.5')
        expected[:,-1]=np.where(expected[:,-1]>g['threshold'],g['above'],g['otherwise'])
    elif g['kind']!='identity':raise ValueError('Document gripper transform explicitly')
    pred=video_info(path('predicted_video'));actual=video_info(path('actual_video'))
    checks={
        'execution_log_present': bool(spec.get('execution_log')) and path('execution_log').is_file(),
        'initial_observation_checksum': bool(spec.get('initial_observation')) and file_info(path('initial_observation'))['sha256']==manifest['observation']['sha256'],
        'action_checksum':file_info(path('raw_action'))['sha256']==manifest['action']['sha256'],
        'prediction_checksum':pred['sha256']==manifest['prediction']['sha256'],
        'matching_applied_actions':bool(np.allclose(applied,expected,rtol=0,atol=1e-6)),
        'matching_fps':abs(pred['fps']-actual['fps'])<1e-6,
        'sufficient_frames':pred['frames']>=count+1 and actual['frames']>=count+1,
        'no_replanning':spec.get('replanned') is False,
        'initial_frame_included':spec.get('initial_frame_included') is True,
        'scene_identity':manifest.get('client_context',{}).get('scenario_id')==ctx['scenario_id'] if ctx['scenario_id']!='legacy_unspecified' else None,
        'camera_layout_checked':spec.get('camera_layout_verified') is True,
        'spatial_mapping_documented':pred['width']==actual['width'] and pred['height']==actual['height'] or bool(spec.get('spatial_mapping')),
    }
    events=spec.get('execution_events',{})
    # Both events must originate from the SAME Spark monotonic clock session.
    received=events.get('prediction_verified_monotonic'); first=events.get('first_action_monotonic')
    checks['saved_and_verified_before_execution']=bool(events.get('clock_session_id')) and type(received) in (int,float) and type(first) in (int,float) and math.isfinite(received) and math.isfinite(first) and 0<=received<first
    checks['joint_policy_path']=manifest.get('comparison_kind')=='jointly_generated_action_and_video'
    return {**ctx,'checks':checks,'status':'alignment_checks_passed' if all(v is True for v in checks.values()) else 'review_required',
        'applied_steps':count,'max_action_difference':float(np.max(np.abs(applied-expected))),
        'predicted':pred,'actual':actual,'execution_events':events,
        'evidence_source':file_info(path('execution_log')) if checks['execution_log_present'] else None, 'early_termination':spec.get('early_termination'),
        'visual_accuracy':'not_evaluated','human_review':'pending',
        'note':'Reported clock events and camera checks require inspection of Spark logs. Image dimensions alone do not prove camera alignment. Legacy scenario identity stays unverified.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    spec=json.loads(a.spec.read_text());r=audit(spec,a.spec.resolve().parent)
    r['spec']=file_info(a.spec);write_json(a.output,r)

if __name__=='__main__':main()
