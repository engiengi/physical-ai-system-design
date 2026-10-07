"""Execute pinned official run.py; observation-only tracing plus declared horizon intervention."""
import argparse,json,runpy,sys,time,hashlib,subprocess,os
from pathlib import Path
import cv2,numpy as np
from cosmos_spark.artifacts import write_json,append_jsonl
from cosmos_spark.runner import snapshot,provenance
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--task',required=True);p.add_argument('--horizon',type=int,choices=[8,16,32],required=True)
a=p.parse_args();out=ROOT/'runs'/a.run_id;out.mkdir(parents=True,exist_ok=False)
# OpenCV's optional preview is separate from the live Isaac Sim GUI. This ARM64
# environment has a headless OpenCV build; keep policy/physics/rendering intact.
opencv_preview=not any('GUI:' in s and s.split(':',1)[1].strip()=='NONE' for s in cv2.getBuildInformation().splitlines())
if not opencv_preview:
 cv2.imshow=lambda *args,**kwargs: None
 cv2.waitKey=lambda *args,**kwargs: -1
 print('[reference] OpenCV auxiliary preview unavailable; Isaac Sim GUI remains enabled.',flush=True)
write_json(out/'gui_compatibility.json',{'isaac_sim_headless':False,'opencv_auxiliary_preview':opencv_preview,'reason':'OpenCV GUI backend unavailable' if not opencv_preview else None})
ep=out/'episode_0000';ep.mkdir();write_json(out/'run.json',{'arguments':vars(a),'environment':provenance(),'official_entrypoint':'RoboLab/policies/cosmos3/run.py','instrumentation':'read-only traces; horizon intervention changes client.OPEN_LOOP_HORIZON'})
from robolab.eval import runner
original_eval=runner.run_evaluation
def evaluation(*args,**kwargs):
 from robolab.eval import episode
 from policies.cosmos3.client import Cosmos3Client
 from robolab.core.utils.video_utils import VideoWriter
 if a.horizon!=Cosmos3Client.OPEN_LOOP_HORIZON:Cosmos3Client.OPEN_LOOP_HORIZON=a.horizon
 orig=episode.run_episode
 def tracked(*args,**kw):
  env=kw['env'];cfg=kw['env_cfg'];client=kw['client'];steps=0;requests=0;writers={};last_obs=None;last_frames={}
  reset0=env.reset;step0=env.step;query0=client._query_server;beg0=client.begin_episode
  def record(obs):
   for group in ['image_obs','viewport_cam']:
    for name,value in obs.get(group,{}).items():
     if name not in ['egocentric_mirrored_camera','wrist_cam','over_shoulder_left_camera','over_shoulder_right_camera']:continue
     f=value[0].detach().cpu().numpy()[...,:3]
     if name not in writers:writers[name]=VideoWriter(str(ep/(name+'.mp4')),1/(cfg.sim.dt*cfg.decimation))
     writers[name].write(f)
     last_frames[name]=f.copy()
     if steps%120==0:cv2.imwrite(str(ep/(name+'_preview.png')),cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
  def reset(*args,**kw):
   nonlocal last_obs
   r=reset0(*args,**kw);last_obs=r[0];return r
  def begin(*args,**kw):
   r=beg0(*args,**kw);snapshot(env,ep/'initial_state.npz');record(last_obs);return r
  def step(actions):
   nonlocal steps
   append_jsonl(ep/'applied_actions.jsonl',{'step':steps,'action':actions.detach().cpu().numpy().tolist()})
   r=step0(actions);steps+=1;record(r[0]);snapshot(env,ep/'latest_state.npz')
   write_json(out/'progress.json',{'steps':steps,'requests':requests,'sim_seconds':steps*cfg.sim.dt*cfg.decimation});return r
  def query(req):
   nonlocal requests
   i=requests;requests+=1
   np.savez_compressed(ep/f'request_{i:05d}_observation.npz',image=req['observation/image'],joint=req['observation/joint_position'],gripper=req['observation/gripper_position'])
   t=time.perf_counter();res=query0(req);np.save(ep/f'request_{i:05d}_actions.npy',np.asarray(res['action']))
   append_jsonl(ep/'requests.jsonl',{'id':i,'step':steps,'prompt':req['prompt'],'seconds':time.perf_counter()-t});return res
  env.reset=reset;env.step=step;client._query_server=query;client.begin_episode=begin
  start=time.perf_counter()
  try:
   result=orig(*args,**kw);snapshot(env,ep/'final_state.npz')
   write_json(ep/'episode.json',{'status':'success' if all(x['success'] for x in result[0]) else 'task_failure','robolab_results':result[0],'steps':steps,'requests':requests,'simulated_elapsed_s':steps*cfg.sim.dt*cfg.decimation,'wall_elapsed_s':time.perf_counter()-start,'horizon':a.horizon,'instruction':cfg.instruction,'env_seed':cfg.seed,'official_time_limit_s':cfg.episode_length_s,'timing':result[2]})
   return result
  finally:
   for w in writers.values():w.release()
   for n,f in last_frames.items():cv2.imwrite(str(ep/(n+'_final.png')),cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
   env.reset=reset0;env.step=step0;client._query_server=query0;client.begin_episode=beg0
 episode.run_episode=tracked
 try:
  result=original_eval(*args,**kwargs)
  # SimulationApp.close() may terminate the interpreter without unwinding the
  # outer runpy finally. Commit evaluation status before official app teardown.
  write_json(out/'run_status.json',{'exit_code':0,'official_episode_present':(ep/'episode.json').exists(),'phase':'evaluation_completed_before_official_app_close'})
  return result
 except BaseException as error:
  write_json(out/'run_status.json',{'exit_code':1,'official_episode_present':(ep/'episode.json').exists(),'error':repr(error)})
  raise
runner.run_evaluation=evaluation
sys.argv=[str(ROOT/'RoboLab/policies/cosmos3/run.py'),'--task',a.task,'--remote-host',__import__('urllib.parse',fromlist=['urlparse']).urlparse(os.environ.get('COSMOS_THOR_URI','ws://thor:8000')).hostname,'--num-envs','1','--num-runs','1','--output-folder-name',a.run_id,'--kit_args','--/app/window/width=1600 --/app/window/height=900']
code=1
try:
 runpy.run_path(sys.argv[0],run_name='__main__');code=0
except SystemExit as e:
 code=e.code or 0
 if code:raise
finally:
 # Kit can suppress exceptions at exit; parent must require completed episode evidence.
 valid=(ep/'episode.json').exists();write_json(out/'run_status.json',{'exit_code':code if valid else 1,'official_episode_present':valid})
