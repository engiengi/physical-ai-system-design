"""Ended-task outcome evaluation; never promote capped diagnostics into trials."""
import hashlib,json,re,subprocess
from pathlib import Path
from .artifacts import write_json

TERMINAL_PROMPT_VERSION='terminal-outcome-v1'

def completed_episode(episode):
    episode=Path(episode)
    result=json.loads((episode/'episode.json').read_text())
    run=json.loads((episode.parent/'run.json').read_text())
    state=json.loads((episode.parent/'run_status.json').read_text())
    args=run['arguments']
    if result.get('mode')!='policy' or args.get('mode')!='policy':
        raise ValueError('A policy task trial is required, not a diagnostic/forecast')
    if args.get('max_steps',0) or result.get('manual_step_cap',0):
        raise ValueError('Manually capped episodes are excluded from terminal outcome evaluation')
    if state.get('exit_code')!=0 or result.get('status') not in ('success','task_failure'):
        raise ValueError('Only completed trials without execution errors are eligible')
    expected=result['status']=='success'
    if result.get('robolab_result',{}).get('success') is not expected:
        raise ValueError('Simulator outcome evidence missing or inconsistent')
    if result.get('termination_evidence',{}).get('environment_finished') is False:
        raise ValueError('Environment did not finish')
    return result

def prepare(entries,output,tail_seconds=8,hold_seconds=1):
    """Return portable unlabeled model inputs and separate reference provenance."""
    output=Path(output)
    if tail_seconds<=0 or hold_seconds<0:raise ValueError('Positive tail and nonnegative hold required')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',e['id']) for e in entries):raise ValueError('Safe case IDs required')
    verified=[(e,completed_episode(e['episode'])) for e in entries]
    if not entries or len({e['id'] for e in entries})!=len(entries):raise ValueError('Unique nonempty cases required')
    output.mkdir(parents=True,exist_ok=False);(output/'media').mkdir()
    cases=[];references=[];evidence=[]
    for entry,result in verified:
        source=Path(entry['episode'])/'egocentric_mirrored_camera.mp4'
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(source)]))
        duration=float(probe['format']['duration']);start=max(0,duration-tail_seconds);length=duration-start
        target=output/'media'/(entry['id']+'.mp4')
        filters=f"trim=start={start},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={hold_seconds},drawtext=text='FINAL FRAME - REPEATED':x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.8:enable='gte(t,{length})'"
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(source),'-vf',filters,'-an','-c:v','libx264','-threads','2','-crf','18','-movflags','+faststart',str(target)],check=True)
        c={'id':entry['id'],'source':'media/'+target.name,'task':result['instruction'],
           'scenario_id':entry['scenario_id'],'experiment_track':entry['experiment_track'],
           'source_type':'simulation','evaluation_phase':'terminal_outcome','trial_ended':True,
           'pair_id':result['run_id']+'_'+result['episode_id'],'variant':f'terminal_tail{tail_seconds}_hold{hold_seconds}','fps':4,'seed':0,
           'source_provenance':'Final segment of an ended simulator trial. Source identity is in the separate Spark preparation evidence. '+('The labeled final frame is repeated editing, not additional physical observation.' if hold_seconds else 'Only original recorded frames; no repeated-frame hold.'),
           'evaluation_criteria':entry['criteria']+['Judge completion at the end of the ended trial, not intermediate progress.','If the final goal cannot be determined visually, answer unknown rather than inventing success or failure.',('The labeled repeated frame provides viewing time, not evidence of extra motion or stability.' if hold_seconds else 'Only the observed interval is evidence; do not assume post-trial stability or future recovery.')]}
        cases.append(c)
        references.append({'id':entry['id'],'scenario_id':entry['scenario_id'],'group_id':c['pair_id'],
            'label':'success' if result['status']=='success' else 'failure','label_source':'simulator',
            'visual_review':None,'source_episode':str(Path(entry['episode']).resolve()),
            'source_video_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'input_video_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
        evidence.append({'id':entry['id'],'source_video':str(source.resolve()),'duration_seconds':duration,
                         'source_range_seconds':[start,duration],'final_hold_seconds':hold_seconds,
                         'boundary':'ended trial; success or task time limit, reason withheld from model'})
    write_json(output/'analysis.json',{'scope':'terminal_outcome_case_study','cases':cases})
    write_json(output/'references.json',{'cases':references,'note':'Simulator references are not human ground truth. Visual review and disagreements are separate.'})
    write_json(output/'preparation.json',{'cases':evidence,'prompt_version':TERMINAL_PROMPT_VERSION if hold_seconds else 'terminal-outcome-raw-v2','no_labels_in_inputs':True})
    return cases

def compare(references,summary):
    refs={r['id']:r for r in references['cases']};seen=set();rows=[]
    for r in summary['results']:
        if r['id'] not in refs or r['id'] in seen:raise ValueError('Unexpected or duplicate result')
        seen.add(r['id']);ref=refs[r['id']];pred=(r.get('prediction') or {}).get('status')
        valid=r.get('status')=='success' and pred in {'success','failure','unknown'}
        rows.append({'id':r['id'],'scenario_id':ref['scenario_id'],'reference':ref['label'],
            'reference_source':ref['label_source'],'prediction':pred,'valid_terminal_answer':valid,
            'abstained':valid and pred=='unknown','matches_reference':valid and pred==ref['label'],
            'error_kind':r.get('error_kind') if r.get('status')!='success' else None if valid else 'non_terminal_answer',
            'seconds':r.get('seconds')})
    return {'rows':rows,'missing':sorted(set(refs)-seen),'pooled_accuracy':None,
            'note':'Case-by-case comparison, not a population accuracy claim. Unknown, format errors and in_progress non-answers stay visible; no pooling across scenarios.'}

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare');s.add_argument('catalog',type=Path);s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('compare');s.add_argument('references',type=Path);s.add_argument('summary',type=Path);s.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':
        result=prepare(json.loads(a.catalog.read_text())['cases'],a.output)
        print(json.dumps({'prepared_ids':[c['id'] for c in result],'folder':str(a.output)},indent=2))
    else:
        if a.output.exists():raise FileExistsError('Preserve existing comparisons; use a new output')
        result=compare(json.loads(a.references.read_text()),json.loads(a.summary.read_text()))
        write_json(a.output,result);print(json.dumps(result,indent=2))


def inspect_response_shape(raw):
    """Report a lossless singleton-list compatibility candidate; never reinterpret prose.

    This diagnostic does not change the strict experimental verdict or historical metrics.
    """
    answer=raw.rsplit('</think>',1)[-1].strip()
    if answer.startswith('```'):
        answer=re.sub(r'^```(?:json)?\s*|\s*```$','',answer).strip()
    obj=json.loads(answer)
    normalization=None
    if isinstance(obj,list):
        if len(obj)!=1:raise ValueError('Only a single outcome can be unwrapped')
        obj=obj[0];normalization='unwrap_singleton_list'
    if not isinstance(obj,dict) or obj.get('status') not in {'success','failure','unknown'}:
        raise ValueError('No unambiguous terminal outcome object')
    if any(not isinstance(obj.get(k),str) or not obj[k].strip() for k in ('stage','evidence')):
        raise ValueError('Missing nonempty stage/evidence')
    return {'compatible_prediction':obj,'format_normalization':normalization,
            'changes_model_judgment':False,'strict_experiment_result_unchanged':True}

if __name__=='__main__':main()
