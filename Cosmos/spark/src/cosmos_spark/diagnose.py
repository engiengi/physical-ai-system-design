"""Package failure evidence and compare controlled reruns without rewriting originals."""
import argparse
import json
from pathlib import Path
import subprocess
import re
import numpy as np
from .artifacts import manifest, write_json


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []

def comparable_config(path):
    value=read(path)
    # Resolved instruction metadata and output location do not change scene physics.
    value.pop('_instruction_variants',None)
    value.get('recorders',{}).pop('dataset_export_dir_path',None)
    return re.sub(r'0x[0-9a-fA-F]+','<address>',json.dumps(value,sort_keys=True))


def compare(before,after):
    a,b=read(before/'episode.json'),read(after/'episode.json')
    initial={}
    with np.load(before/'initial_state.npz') as x,np.load(after/'initial_state.npz') as y:
        if set(x.files)!=set(y.files):initial={'keys_match':False}
        else:
            initial={k:float(np.max(np.abs(x[k]-y[k]))) if x[k].size and x[k].shape==y[k].shape else (0 if x[k].shape==y[k].shape else None) for k in x.files}
    qa,qb=rows(before/'requests.jsonl'),rows(after/'requests.jsonl')
    seeds=lambda rr:[r.get('policy_seed_used') for r in rr]
    sa,sb=seeds(qa),seeds(qb)
    seed_verified=bool(sa and sb and all(type(x)is int for x in sa+sb) and sa[:min(len(sa),len(sb))]==sb[:min(len(sa),len(sb))])
    server_a=read(before/'server.json') if (before/'server.json').exists() else {}
    server_b=read(after/'server.json') if (after/'server.json').exists() else {}
    checks={'initial_state_equal_within_1e-6':bool(initial) and all(type(v)in(int,float) and v is not False and v is not None and v<=1e-6 for v in initial.values()),
            'environment_seed_equal':a.get('seed')==b.get('seed'),
            'config_equal':comparable_config(before/'env_cfg.json')==comparable_config(after/'env_cfg.json'),
            'policy_seed_prefix_verified':seed_verified,
            'model_revision_equal':bool(server_a.get('model_revision')) and server_a.get('model_revision')==server_b.get('model_revision'),
            'instruction_equal':a.get('instruction')==b.get('instruction')}
    return {'before':str(before),'after':str(after),'checks':checks,'controlled_comparison':all(checks.values()),
            'config_exclusions':['/_instruction_variants','/recorders/dataset_export_dir_path','callable memory addresses'],
            'initial_state_max_abs_error':initial,'before_result':a,'after_result':b,
            'before_policy_seeds':sa,'after_policy_seeds':sb,
            'source_before':read(before.parent/'run.json')['environment']['source_sha256'],
            'source_after':read(after.parent/'run.json')['environment']['source_sha256'],
            'human_causal_review':None,'note':'Matching initial conditions alone does not establish a causal model improvement.'}


def bundle(episode,output,step=None):
    output.mkdir(parents=True,exist_ok=False)
    result=read(episode/'episode.json');actions=rows(episode/'applied_actions.jsonl');requests=rows(episode/'requests.jsonl')
    step=result.get('steps',0) if step is None else step
    if not 0<=step<=result.get('steps',0):raise ValueError('Step outside recorded episode')
    hz=result.get('control_hz',15);start=max(0,(step-45)/hz);duration=min(90/hz,(result.get('steps',0)+1)/hz-start)
    clips=[]
    for name in ('egocentric_mirrored_camera','wrist_cam'):
        source=episode/(name+'.mp4')
        if source.exists() and duration>0:
            target=output/(name+'_context.mp4')
            subprocess.run(['ffmpeg','-v','error','-nostdin','-ss',str(start),'-i',str(source),'-t',str(duration),
                            '-c:v','libx264','-threads','2','-preset','fast','-crf','20','-an',str(target)],check=True)
            clips.append(str(target))
    selected=[r for r in requests if max(0,step-45)<=r.get('step',0)<=step+45]
    report={'episode':str(episode),'result':result,'focus_step':step,'clip_start_sim_seconds':start,'clips':clips,
            'applied_actions':[r for r in actions if max(0,step-45)<=r['step']<=step+45],
            'requests':selected,'subtasks':[r for r in rows(episode/'subtasks.jsonl') if max(0,step-45)<=r['step']<=step+45],
            'raw_observations':[str(episode/(r['request_id']+'_observation.npz')) for r in selected],
            'error':(episode/'error.txt').read_text() if (episode/'error.txt').exists() else None,
            'classification':result['status'],'human_failure_stage':None,'human_cause':None}
    write_json(output/'evidence.json',report);manifest(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('bundle');b.add_argument('episode',type=Path);b.add_argument('--output',type=Path,required=True);b.add_argument('--step',type=int)
    c=sub.add_parser('compare');c.add_argument('before',type=Path);c.add_argument('after',type=Path);c.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='bundle':bundle(a.episode.resolve(),a.output.resolve(),a.step)
    else:
        if a.output.exists():raise FileExistsError(a.output)
        write_json(a.output,compare(a.before.resolve(),a.after.resolve()))
    print(a.output)
if __name__=='__main__':main()
