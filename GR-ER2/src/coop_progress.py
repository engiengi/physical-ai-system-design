"""Offline ER2 video progress/moment assessment; no truth supplied to the API."""
import argparse
import base64
import json
import shutil
import subprocess
import time
from app import client,response_text
from common import ROOT,safe_child,write_json,parse_json,read_json
from coop_rules import progress_bracket
from coop_run import MODEL


def main():
    p=argparse.ArgumentParser();p.add_argument('run')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--reuse',action='store_true');mode.add_argument('--retry-incomplete',action='store_true')
    args=p.parse_args()
    directory=safe_child(ROOT,args.run);experiment=read_json(directory/'experiment.json')
    truth=[json.loads(line) for line in (directory/'truth.jsonl').read_text().splitlines()]
    duration=read_json(directory/'recording.json')['simulation_seconds'];api=None if args.reuse else client();results=[]
    task=('Each of the TWO green trays must contain one red cube and one blue cube. There are four requirements, '
          'worth 25% each: bottom red, bottom blue, top red, top blue. Cubes in the central yellow exchange do not count.'
          if experiment['scene']=='dual_franka' else
          'Transfer the red material from conveyor A through Spot to conveyor B. Four milestones worth 25% each: '
          'loaded in Spot tray, departed A while carrying, arrived B with material retained, unloaded onto conveyor B. '
          'Whole task completes only after physical unloading. The full overview is supplied for this separate OFFLINE analysis only.')
    for label,fraction in [('midpoint',.5),('final',1.)]:
        dest=directory/'video_progress'/label;dest.mkdir(parents=True,exist_ok=True)
        existing=read_json(dest/'response.json') if (dest/'response.json').exists() else None
        if args.retry_incomplete and existing is None:raise RuntimeError('No previous response to retry')
        reuse=args.reuse or (args.retry_incomplete and existing.get('status')!='incomplete')
        cutoff=min(duration,truth[-1]['elapsed_sim'])*fraction
        if not reuse and not args.retry_incomplete:
            if (dest/'response.json').exists():raise RuntimeError('Existing API result: use --reuse to preserve the original input and response')
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(directory/'input_video/overview.mp4'),
                            '-t',str(cutoff),'-vf','fps=1','-c:v','libx264','-pix_fmt','yuv420p',str(dest/'input_video.mp4')],check=True)
        prompt=task+'\nWatch only this video prefix. Return JSON: {"progress_level":"0-20|20-40|40-60|60-80|80-100",'
        prompt+='"completion_time_seconds": number or null, "evidence":"visible evidence"}. '
        prompt+='Select one of the five progress brackets at the final frame. Give the FIRST time the WHOLE objective is complete, null if not completed. Timestamps are seconds in the provided video.'
        if args.retry_incomplete:
            prompt=(dest/'prompt.txt').read_text()
        elif not reuse:(dest/'prompt.txt').write_text(prompt)
        started=time.time()
        if reuse:
            text=(dest/'response.txt').read_text()
        else:
            if args.retry_incomplete:
                for filename in ['response.json','response.txt','evaluation.json']:
                    shutil.copy2(dest/filename,dest/('previous_incomplete_'+filename))
            write_json(dest/'request.json',{'model':MODEL,'max_output_tokens':4096,'thinking_level':'medium',
                'input_video':'input_video.mp4','prompt':'prompt.txt','retry_incomplete':args.retry_incomplete,'wall_time':started})
            response=api.interactions.create(model=MODEL,input=[{'type':'video','mime_type':'video/mp4',
                'data':base64.b64encode((dest/'input_video.mp4').read_bytes()).decode()},{'type':'text','text':prompt}],
                generation_config={'thinking_level':'medium','max_output_tokens':4096})
            write_json(dest/'response.json',response.model_dump(mode='json'))
            text=response_text(response);(dest/'response.txt').write_text(text)
        try:
            parsed=parse_json(text)
            if not isinstance(parsed,dict):raise ValueError('Expected JSON object')
        except Exception:parsed={'parse_error':True}
        # Evaluator accesses truth only after the API finishes.
        reference=max((t for t in truth if t['elapsed_sim']<=cutoff),key=lambda t:t['elapsed_sim'])
        complete=next((t['elapsed_sim'] for t in truth if t['elapsed_sim']<=cutoff and
                       (t.get('physical_task_success') or t.get('station_progress') and all(v>=1 for v in t['station_progress'].values()))),None)
        if complete is not None:complete-=truth[0]['elapsed_sim']
        controller_complete=complete
        if experiment['scene']=='transport':
            # Separate visible material placement from the later controller
            # return-home milestone. The video prompt asks for unloading.
            candidates=[]
            for before,after in zip(truth,truth[1:]):
                pos=after['payload_position'];old=before['payload_position']
                placed=after.get('arrived_loaded') and abs(pos[0]-4.4)<.065 and abs(pos[1]+.65)<.13 and .746<pos[2]<.762
                released=min(after['robot_joints']['B'][-2:])>.035
                settled=sum((a-b)**2 for a,b in zip(pos,old))<.002**2
                if after['elapsed_sim']<=cutoff and placed and released and settled:candidates.append(after)
            complete=candidates[0]['elapsed_sim']-truth[0]['elapsed_sim'] if candidates else None
        reference_fraction=reference['progress_fraction']
        if experiment['scene']=='transport':
            reference_fraction=(sum(bool(reference[k]) for k in ['loaded_once','departed_loaded','arrived_loaded'])+(complete is not None))/4
        previous=read_json(dest/'evaluation.json') if reuse and (dest/'evaluation.json').exists() else {}
        predicted=parsed.get('completion_time_seconds')
        valid_time=type(predicted) in (float,int) and 0<=predicted<=cutoff
        result={'prefix':label,'cutoff_sim_seconds':cutoff,'api_seconds':previous.get('api_seconds',time.time()-started),'model':parsed,
                'reference_progress_fraction':reference_fraction,'reference_bracket':progress_bracket(reference_fraction),
                'reference_completion_time_seconds':complete,'time_resolution_seconds':1,
                'video_time_origin_elapsed_sim':truth[0]['elapsed_sim'],
                'response_status':read_json(dest/'response.json').get('status'),
                'bracket_matches':None if parsed.get('parse_error') else parsed.get('progress_level')==progress_bracket(reference_fraction),
                'completion_presence_matches':None if parsed.get('parse_error') else ('completion_time_seconds' in parsed) and ((predicted is None)==(complete is None)),
                'completion_absolute_error_seconds':abs(predicted-complete) if valid_time and complete is not None else None}
        if experiment['scene']=='transport':
            result['reference_controller_cycle_complete_seconds']=controller_complete
            result['completion_rule']='material on B endpoint, fingers open, consecutive frames stable; controller return-home time reported separately'
        write_json(dest/'evaluation.json',result);results.append(result);print(json.dumps(result),flush=True)
    write_json(directory/'video_progress/summary.json',results)


if __name__=='__main__':main()
