"""Read-only recording checks plus a saved validation report; no API/GPU needed."""
import argparse
import json
import subprocess
import time
from common import ROOT,read_json,safe_child,write_json
from coop_rules import evaluate_arm


def validate(path):
    recording=read_json(path/'recording.json')
    frames=[json.loads(x) for x in (path/'frames.jsonl').read_text().splitlines()]
    truth=[json.loads(x) for x in (path/'truth.jsonl').read_text().splitlines()]
    assert len(frames)==len(truth)==recording['frames']>0,'Frame/trace count mismatch'
    assert [r['frame'] for r in frames]==list(range(len(frames))),'Noncontiguous frames'
    assert all(a['wall_time']<b['wall_time'] and a['simulation_time']<b['simulation_time'] for a,b in zip(frames,frames[1:])), 'Nonmonotonic clock'
    observations=0
    refinements=0
    if (path/'events.jsonl').exists():
        import numpy as np
        from transport_geometry import refine_block_top
        for event in [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]:
            if event['kind']!='depth_point_refinement':continue
            source=path/'control_inputs'/event['observation_id']
            depth=np.load(source/'depth.npy');read_json(source/'meta.json')
            x,y,_=refine_block_top(depth,*event['requested_pixel'])
            assert np.allclose([x,y],event['refined_pixel'],atol=1e-6),'Depth refinement differs from saved sensor input'
            refinements+=1
    for reqpath in (path/'api').glob('*/request.json'):
        request=read_json(reqpath)
        if read_json(path/'experiment.json')['scene']=='transport':
            assert 'overview' not in request['observations'],'Global viewer camera leaked to transport model'
        for obs in request['observations'].values():
            observations+=1
            assert obs['wall_time']<=request['wall_time'],'Future observation leaked'
            assert (path/obs['saved_image']).is_file(),'Missing input still'
            if obs.get('saved_clip'):
                assert (path/obs['saved_clip']).is_file(),'Missing input clip'
                assert obs['clip_simulation_range'][1]<=obs['simulation_time'],'Future clip leaked'
    summary=read_json(path/'summary.json')
    jobs=list(read_json(path/'jobs.json').values())
    assert len(jobs)==summary['robot_job_count']
    assert all(j['started_wall']<=j['ended_wall'] for j in jobs),'Missing/inverted job clock'
    if 'objects' in truth[-1]:
        assert evaluate_arm(truth[-1]['objects'])['physical_task_success']==summary['physical_task_success'],'Physical score mismatch'
    if 'payload_position' in truth[-1]:
        last=truth[-1];x,y,z=last['payload_position']
        goal=last['arrived_loaded'] and abs(x-4.4)<.065 and abs(y+.65)<.13 and .73<z<.82
        assert bool(goal)==summary['physical_task_success'],'Transport physical score mismatch'
    expected=[path/'input_video'/f'{c}.mp4' for c in ['overview','A','B']]+[path/'rollout_video/side.mp4',path/'presentation/synchronized.mp4']
    assert all(p.is_file() for p in expected),'Missing exported video'
    checked=[]
    for video in sorted(path.rglob('*.mp4')):
        metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,duration,nb_frames','-of','json',str(video)],text=True))['streams'][0]
        assert float(metadata['duration'])>0 and int(metadata['nb_frames'])>0,'Empty video'
        result=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(video),'-f','null','-'],capture_output=True,text=True)
        assert result.returncode==0 and not result.stderr.strip(),f'Decode failure: {video}: {result.stderr}'
        if video in expected[:4]:
            assert abs(float(metadata['duration'])-len(frames)/recording['fps'])<.03,'Raw video timing mismatch'
        checked.append({'path':str(video.relative_to(path)),**metadata})
    report={'passed':True,'checked_at':time.time(),'frames':len(frames),'request_observation_references':observations,
        'videos_decoded':len(checked),'videos':checked,'no_future_observations':True,
        'depth_refinements_verified':refinements,
        'physical_score_recomputed': 'objects' in truth[-1] or 'payload_position' in truth[-1]}
    write_json(path/'validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='videos'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    validate(safe_child(ROOT,args.run))
