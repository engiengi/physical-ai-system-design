"""Bounded ER2 orchestration over asynchronous, individually addressed robots.

Standard Interactions API, NOT Gemini Live. A continuously rendered camera is
sampled into causal clips/stills; low-level physics continues during API calls.
"""
import argparse
import base64
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

from app import bridge, client, image_part, response_text, run_dir, redact
from common import ROOT, write_json

MODEL = 'gemini-robotics-er-2-preview'


def tool(name, description, properties, required):
    return {'type':'function','name':name,'description':description,
            'parameters':{'type':'object','properties':properties,'required':required}}


STR = {'type':'string'}
TOOLS = [
    tool('observe','Get a fresh RGB still and the preceding camera clip. Never contains ground truth.',
         {'camera_id':{'type':'string','enum':['overview','A','B']}},['camera_id']),
    tool('start_action','Submit asynchronous robot job; returns immediately. Multiple robots can run concurrently.',
         {'robot_id':{'type':'string','enum':['A','B']},'action':STR,
          'arguments':{'type':'object','properties':{'observation_id':STR,'pick':{'type':'array','items':{'type':'number'}},
                       'place':{'type':'array','items':{'type':'number'}},'destination':STR}}},['robot_id','action','arguments']),
    tool('get_job_status','Read motor-job status. sequence_completed does NOT certify successful object transfer.',
         {'job_id':STR},[]),
    tool('cancel_job','Cancel this robot job only.',{'job_id':STR},['job_id']),
    tool('report_progress','Report visually supported progress, assignment and uncertainty; this does not move a robot.',
         {'summary':STR,'evidence':STR,'next_plan':STR},['summary','evidence','next_plan']),
    tool('finish_task','Finish only after visual verification, or report a bounded failure.',
         {'success':{'type':'boolean'},'evidence':STR},['success','evidence']),
]

ARM_TASK = '''Two Franka arms share a table. Robot A owns the lower half of the overhead image and the bottom green tray;
robot B owns the upper half and top green tray. A initially has two red cubes and B two blue cubes.
GOAL: each green tray must contain exactly one red and one blue cube. Use the central yellow exchange area
to transfer between workspaces. Robot A cannot pick/place in the upper private half and B cannot in the lower half.
One robot at a time may use the central exchange; a busy rejection is a safety constraint, not a plan.
Choose your own assignment/order. Submit independent A/B jobs together when useful. After a transfer, inspect the
image before letting the other arm receive: a completed motor sequence can leave the cube at its source.
Tool start_action: action="pick_place", arguments={observation_id, pick:[y,x], place:[y,x]}.
Coordinates are normalized 0..1000 in the EXACT supplied overview image. Pick the TOP CENTER of a cube,
place on EMPTY flat tray/exchange floor, away from walls and other cubes. Use the current observation_id.
Each action includes approach/grasp/lift/transport/release/retreat. Low-level controllers use depth and robot state.
Do not queue another job for a busy arm. Never invent success. If the grasp failed, reobserve and try again.
'''
TRANSPORT_TASK = '''Three robots transfer ONE red material cube from conveyor A to conveyor B, about 4m apart.
Franka A loads it into the green tray on Spot's back. Spot carries it to dock B. Franka B unloads onto the green
endpoint of conveyor B. Both conveyors are stopped at handoff. The task completes when the material is on B.
Robot IDs: A and B are Frankas; Spot is the carrier. Cameras: A/B = overhead station views, cargo = tray view,
Spot = forward view. No global overview is provided. Request relevant observations using observe.
Spot begins docked at A. A/B arms face their local workspace. Station cameras look downward; identify objects visually.
pick_place for A/B takes {observation_id, pick:[y,x], place:[y,x]}, normalized0..1000 in that camera image.
Pick the visible red cube TOP CENTER. Place at EMPTY green tray CENTER for loading, green conveyor endpoint for unloading.
Observe a fresh station image before each pick. When pointing at the tray avoid Spot body/walls: use empty floor.
navigate for Spot takes {destination:"A" or "B", route:"direct" or "detour"}.
Direct route connects the docks along world y=0.30m. Detour uses a parallel aisle at world y=1.55m.
Navigation uses proprioception/known waypoints and local obstacle stop; no autonomous rerouting.
Motor sequence_completed is NOT confirmation that the material transferred. Observe cargo/station after every handoff.
Do NOT depart unless visually loaded and the arm job has ended/retreated. If first grasp misses, inspect and retry.
If navigation returns blocked, inspect forward video and choose the available detour. Do not repeat blocked direct route.
While carrier moves, B must wait; interlocks reject handoff during movement. Arrival does not prove retained cargo.
The robot controller runs asynchronously, so query jobs and reobserve as needed. Report progress with evidence at handoffs.
Do not invent visual success. At most three grasp retries per station. Finish with honest bounded failure if recovery fails.
'''

def encode_clip(frames, output, fps=5):
    if not frames:return False
    listing=output.with_suffix('.ffconcat')
    listing.write_text('ffconcat version 1.0\n'+''.join(f"file '{p.resolve()}'\nduration {1/fps}\n" for p in frames))
    subprocess.run(['ffmpeg','-y','-loglevel','error','-safe','0','-f','concat','-i',str(listing),
                    '-r',str(fps),'-c:v','libx264','-pix_fmt','yuv420p',str(output)],check=True)
    listing.unlink();return True


def capture(directory, camera_id, turn):
    obs=bridge('observe',camera_id=camera_id)
    dest=directory/'api_inputs'/f'{turn:03d}_{camera_id}_{obs["observation_id"]}';dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/obs['image'],dest/'image.jpg')
    obs['saved_image']=str((dest/'image.jpg').relative_to(directory))
    # Copy only frames already recorded before this observation. No future leakage.
    stamps=[]
    path=directory/'frames.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError:continue
            if row['wall_time']<=obs['wall_time']:stamps.append(row)
    chosen=stamps[-30:]
    if len(chosen)>=5:
        frames=[directory/'frames'/camera_id/f"{r['frame']:06d}.jpg" for r in chosen]
        if encode_clip(frames,dest/'preceding_clip.mp4'):
            obs['saved_clip']=str((dest/'preceding_clip.mp4').relative_to(directory))
            obs['clip_simulation_range']=[chosen[0]['simulation_time'],chosen[-1]['simulation_time']]
    write_json(dest/'observation.json',obs)
    return obs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=['dual_franka','transport'],required=True)
    p.add_argument('--condition',choices=['normal','exception','load_failure','obstacle'],default='normal')
    p.add_argument('--max-turns',type=int,default=55)
    p.add_argument('--max-seconds',type=int,default=1500)
    args=p.parse_args()
    if args.max_turns<1 or args.max_seconds<1:p.error('Limits must be positive')
    if args.condition not in (('normal','exception') if args.scene=='dual_franka' else ('normal','load_failure','obstacle')):
        p.error('Condition does not belong to selected scene')
    status=json.loads((ROOT/'runtime/status.json').read_text())
    if not status.get('ready') or status.get('world_scenario')!=args.scene:raise RuntimeError('Start the matching simulator scene first')
    directory=run_dir('07_cooperative/'+args.scene+'/'+args.condition)
    write_json(directory/'experiment.json',vars(args)|{'model':MODEL,'api_mode':'standard_interactions_causal_camera_clips',
        'not_a_live_api_run':True,'truth_is_evaluator_only':True})
    print(directory,flush=True)
    api=client()
    bridge('coop_reset',condition=args.condition)
    time.sleep(6 if args.scene=='transport' else 2)
    bridge('record_start',output=str(directory.relative_to(ROOT)))
    history=[]; observations={}; completed=None; started=time.time()
    task=ARM_TASK if args.scene=='dual_franka' else TRANSPORT_TASK
    tools=copy.deepcopy(TOOLS)
    fields=tools[1]['parameters']['properties']['arguments']
    allowed=['observation_id','pick','place'] if args.scene=='dual_franka' else ['destination']
    fields['properties']={k:v for k,v in fields['properties'].items() if k in allowed}
    fields['required']=allowed
    if args.scene=='transport':
        tools[0]['parameters']['properties']['camera_id']['enum']=['A','B','Spot','cargo']
        tools[1]['parameters']['properties']['robot_id']['enum']=['A','B','Spot']
        fields['properties']={'observation_id':STR,'pick':{'type':'array','items':{'type':'number'}},
                              'place':{'type':'array','items':{'type':'number'}},'destination':STR,'route':STR}
        fields.pop('required',None)
    tools[-1]['description']='Finish only after visual verification of the goal, or report bounded failure.'
    try:
        for turn in range(args.max_turns):
            if time.time()-started>args.max_seconds:break
            # Fresh overview is always shown. Robot clips are requested by the model.
            refresh=list(observations) or (['overview'] if args.scene=='dual_franka' else ['A'])
            for camera in refresh:observations[camera]=capture(directory,camera,turn)
            jobs=bridge('get_job_status')
            prompt=task+'''\nYou are the high-level ER2 orchestrator. Use the supplied tools; do not just narrate.
Report visual progress at milestones. Only RGB images/clips and public motor-job feedback are available.
You choose robots, tool order, parallel dispatch, retry and reassignment. Low-level tools do not solve the task.
Do not call finish_task before verifying the objective. Several function calls in one response are supported.
Current observations (IDs label the images in this request):\n'''+json.dumps(observations)+ '\nCurrent jobs:\n'+json.dumps(jobs)+'\nHistory:\n'+json.dumps(history[-24:])
            parts=[{'type':'text','text':prompt}]
            for camera_id,obs in observations.items():
                parts.append({'type':'text','text':f'Camera {camera_id}, observation {obs["observation_id"]}, sim time {obs["simulation_time"]}'})
                parts.append(image_part(directory/obs['saved_image']))
                if args.scene=='transport' and obs.get('saved_clip'):
                    parts.append({'type':'video','data':base64.b64encode((directory/obs['saved_clip']).read_bytes()).decode(),'mime_type':'video/mp4'})
            dest=directory/'api'/f'{turn:03d}';dest.mkdir(parents=True)
            write_json(dest/'request.json',{'model':MODEL,'prompt':prompt,'observations':observations,'tools':tools,'wall_time':time.time()})
            tick=time.time()
            response=api.interactions.create(model=MODEL,input=parts,tools=tools,
                generation_config={'thinking_level':'medium','max_output_tokens':4096})
            raw=response.model_dump(mode='json');write_json(dest/'response.json',raw)
            text=response_text(response);(dest/'response.txt').write_text(text)
            calls=[]
            for item in raw.get('outputs',raw.get('steps',[])):
                if item.get('type')=='function_call':calls.append(item)
            entry={'turn':turn,'request_wall':tick,'response_wall':time.time(),'text':text,'calls':calls,'results':[]}
            for call in calls:
                name=call['name'];kw=call.get('arguments',{})
                if isinstance(kw,str):kw=json.loads(kw)
                try:
                    if name=='observe':
                        if args.scene=='transport' and kw['camera_id'] not in ('A','B','Spot','cargo'):
                            raise ValueError('Only local task cameras are available; overview is viewer-only')
                        obs=capture(directory,kw['camera_id'],turn+1)
                        observations[kw['camera_id']]=obs;result=obs
                    elif name in ('start_action','get_job_status','cancel_job'):result=bridge(name,**kw)
                    elif name=='report_progress':result={'recorded':True}
                    elif name=='finish_task':completed=kw;result={'finished':True}
                    else:raise ValueError('Unknown tool')
                except Exception as exc:result={'error':redact(exc)}
                entry['results'].append({'name':name,'arguments':kw,'result':result,'wall_time':time.time()})
            history.append(entry);write_json(directory/'model_timeline.json',history)
            print(f'Turn {turn}: {[c["name"] for c in calls]} '+text[:100],flush=True)
            if completed is not None:break
            if not calls or all(c['name'] in ('get_job_status','report_progress') for c in calls):time.sleep(3)
        write_json(directory/'model_report.json',completed or {'success':False,'evidence':'Host bound reached; model did not finish'})
    except Exception as exc:
        write_json(directory/'error.json',{'error':redact(exc)});raise
    finally:
        bridge('hold');bridge('record_stop')
        print('Recording stopped: '+str(directory),flush=True)


if __name__=='__main__':main()
