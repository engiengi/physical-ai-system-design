"""Build reviewable video cases; frozen final-frame variants are explicitly labeled."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .artifacts import manifest, write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(episodes,synthetic_runs,output,hold_seconds=2,synthetic_task=None):
    if not 0<=hold_seconds<=30:raise ValueError('hold_seconds must be in [0,30]')
    output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for ep in episodes:
        result=json.loads((ep/'episode.json').read_text())
        if result['status'] not in ('success','task_failure'):continue
        source=ep/'egocentric_mirrored_camera.mp4'
        if not source.exists():raise FileNotFoundError(source)
        ident=ep.parent.name+'_'+ep.name
        common={'group_id':ident,'source_type':'simulation','task':result['instruction'],
                'simulator_status':result['status'],'episode':str(ep),'human_status':None}
        cases.append({**common,'id':ident+'_original','video':str(source),'sha256':sha(source),'variant':'original'})
        if hold_seconds:
            dest=output/(ident+'_final_hold.mp4')
            # Deliberate video-only counterfactual: no extra simulation or stability claim.
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(source)],text=True))
            duration=float(probe['format']['duration'])
            filt=f"tpad=stop_mode=clone:stop_duration={hold_seconds},drawtext=text='FROZEN FINAL FRAME':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7:enable='gte(t,{duration})'"
            subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(source),'-vf',filt,'-an',
                            '-c:v','libx264','-preset','fast','-threads','2','-crf','20','-movflags','+faststart',str(dest)],check=True)
            cases.append({**common,'id':ident+'_final_hold','video':str(dest),'sha256':sha(dest),'variant':'frozen_final_frame',
                          'hold_seconds':hold_seconds,'source_duration':duration,
                          'note':'Repeated final frame with visible label; not additional physical observation.'})
    for run in synthetic_runs:
        summary=json.loads((run/'summary.json').read_text())
        manifest_path=run/'manifest.json'
        manifest_data=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        metadata={c['id']:c for c in manifest_data.get('cases',[])} if isinstance(manifest_data,dict) else {}
        for r in summary['results']:
            if r['status']!='success':continue
            video=run/r['id']/'native/vision.mp4'
            if not video.exists():raise FileNotFoundError(video)
            inp=run/r['id']/'input.json'
            input_data=json.loads(inp.read_text()) if inp.exists() else {}
            if input_data.get('model_mode')=='forward_dynamics':continue
            source_case=metadata.get(r['id'],{})
            task=source_case.get('task') or synthetic_task
            if not isinstance(task,str) or not task.strip():
                raise ValueError('Generation case needs task metadata or explicit --synthetic-task; never assume banana')
            prefix=run.parent.name if run.name=='output' else run.name
            cases.append({'id':prefix+'_'+r['id'],'group_id':prefix+'_'+source_case.get('pair_id',r['id']),
                          'source_run':str(run),
                          'source_type':'synthetic','task':task,
                          **{k:source_case[k] for k in ('scenario_id','track','experiment_track','source_provenance','evaluation_criteria') if k in source_case},
                          'video':str(video),'sha256':sha(video),'variant':'generated',
                          'requested_event':input_data.get('prompt',''),'human_status':None,
                          'note':'Generation prompt is a request, never ground truth.'})
    if not cases:raise ValueError('No eligible cases')
    if len({c['id'] for c in cases})!=len(cases):raise ValueError('Duplicate case IDs')
    dataset={'schema_version':1,'cases':cases,'human_review':None}
    write_json(output/'dataset.json',dataset);manifest(output)
    return dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',nargs='*',type=Path,default=[])
    p.add_argument('--synthetic-runs',nargs='*',type=Path,default=[])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--hold-seconds',type=float,default=2)
    p.add_argument('--synthetic-task',help='Explicit task for legacy generation manifests without task metadata')
    a=p.parse_args();d=build([p.resolve() for p in a.episodes],[p.resolve() for p in a.synthetic_runs],a.output.resolve(),a.hold_seconds,a.synthetic_task)
    print(f"{len(d['cases'])} cases: {a.output/'dataset.json'}")
if __name__=='__main__':main()
