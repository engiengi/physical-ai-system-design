"""One visible scenario trial with both actual desktop recordings."""
import argparse,json,os,signal,subprocess,sys
from pathlib import Path
from cosmos_spark.remote import ThorRecording,ROOT
from cosmos_spark.artifacts import write_json
from robolab.eval.websocket_transport import MsgPackWebSocketTransport
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--task',required=True)
p.add_argument('--mode',choices=['policy','predict'],required=True);p.add_argument('--steps',type=int,default=0)
p.add_argument('--purpose',choices=['terminal_outcome','midstream_diagnostic','forecast_comparison'])
p.add_argument('--wall-timeout',type=int,default=3600,help='Operational guard, never a task failure criterion')
p.add_argument('--report-root',type=Path,default=ROOT/'reports/scenarios_20261005')
p.add_argument('--post-terminal-seconds',type=float,default=0)
p.add_argument('--goal-variant',choices=['official','single-red-hammer-60s','single-red-hammer-no-drill-60s'],default='official')
p.add_argument('--horizon',type=int,default=16);p.add_argument('--policy-seed',type=int,default=1200)
p.add_argument('--track',choices=['standalone','banana_link'],required=True);p.add_argument('--stage',type=int,choices=[6,7],required=True)
a=p.parse_args()
a.purpose=a.purpose or ('forecast_comparison' if a.mode=='predict' else 'terminal_outcome')
if a.purpose=='terminal_outcome' and (a.mode!='policy' or a.steps):p.error('Terminal outcome requires policy mode with no early --steps cap; use the task episode limit')
if a.purpose=='midstream_diagnostic' and (a.mode!='policy' or a.steps<=0):p.error('Midstream diagnostic requires a positive explicit --steps cap')
if a.purpose=='forecast_comparison' and a.mode!='predict':p.error('Forecast comparison requires predict mode')
if a.wall_timeout<=0:p.error('Positive operational timeout required')
out=a.report_root/a.run;out.mkdir(parents=True,exist_ok=False)
transport=MsgPackWebSocketTransport(os.environ.get('COSMOS_THOR_URI','ws://thor:8000'),metadata_timeout=5,connect_kwargs={'open_timeout':5,'close_timeout':2})
try:meta=transport.connect()
finally:transport.close()
if a.mode=='predict' and not meta.get('decode_video'):raise RuntimeError('Start a decoder-enabled server before prediction')
rec=ThorRecording(a.run,meta['run_id'],ROOT/'captures'/a.run)
record={'configuration':{**vars(a),'report_root':str(a.report_root)},'server':meta,'scope':a.purpose,
        'terminal_outcome_eligible':False,'human_review':None}
cmd=[sys.executable,'-m','cosmos_spark.session','--run-id',a.run,'--task',a.task,'--mode',a.mode,'--policy-seed',str(a.policy_seed),'--execute-horizon',str(a.horizon),'--timeout','120','--kit_args','--/app/window/width=1600 --/app/window/height=900']
cmd+=['--scenario-id',a.task,'--experiment-track','new_standalone' if a.track=='standalone' else 'banana_link']
if a.mode=='policy' and a.steps:cmd+=['--max-steps',str(a.steps)]
cmd+=['--post-terminal-seconds',str(a.post_terminal_seconds),'--goal-variant',a.goal_variant]
record['command']=cmd;write_json(out/'trial.json',record)
started=False;proc=None
try:
 rec.start();started=True
 with (out/'launcher.log').open('w') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  record['exit_code']=proc.wait(timeout=a.wall_timeout)
except BaseException as e:
 record['error']=repr(e)
 if proc and proc.poll() is None:
  os.killpg(proc.pid,signal.SIGINT)
  try:proc.wait(timeout=40)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=20)
finally:
 if started:
  try:record['thor_recording']=rec.stop()
  except Exception as e:record['recording_error']=repr(e)
 ep=ROOT/'runs'/a.run/'episode_0000/episode.json'
 if ep.exists():record['episode']=json.loads(ep.read_text())
 record['terminal_outcome_eligible']=a.purpose=='terminal_outcome' and record.get('episode',{}).get('status') in ('success','task_failure') and not record.get('error') and not record.get('recording_error') and record.get('exit_code')==0
 write_json(out/'trial.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ('server','thor_recording')},indent=2))
if 'error' in record or record.get('recording_error') or record.get('exit_code') or record.get('episode',{}).get('status') in ('execution_error','incomplete'):raise SystemExit(1)
