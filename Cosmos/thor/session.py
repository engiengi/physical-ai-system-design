"""Manage one Thor experiment, real X11 recording and sampled host memory together."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import write_json
from artifacts import file_info

HERE=Path(__file__).resolve().parent


def sample(pid):
    def fields(path):
        result={}
        try:
            for line in Path(path).read_text().splitlines():
                parts=line.split()
                if len(parts)>2 and parts[2]=='kB': result[parts[0].rstrip(':')]=int(parts[1])*1024
        except FileNotFoundError: pass
        return result
    m=fields('/proc/meminfo'); p=fields(f'/proc/{pid}/status')
    return {'unix':time.time(), 'process_rss_bytes':p.get('VmRSS'), 'process_hwm_bytes':p.get('VmHWM'),
            'system_used_bytes':m.get('MemTotal',0)-m.get('MemAvailable',0)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','start','stop','status','worker'])
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--batch-id')
    p.add_argument('--no-record',action='store_true')
    p.add_argument('--interval',type=float,default=.5)
    a,command=p.parse_known_args()
    if command[:1]==['--']: command=command[1:]
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',a.run): p.error('Unsafe run name')
    if not .1 <= a.interval <= 60: p.error('interval must be .1..60 seconds')
    root=a.workspace.resolve(); out=root/'sessions'/a.run; state=out/'session.json'
    if a.mode=='status':
        print(state.read_text()); return
    if a.mode=='stop':
        r=json.loads(state.read_text()); pid=r['supervisor_pid']; proc=Path(f'/proc/{pid}/cmdline')
        if r.get('finished_at_unix'): print('Already finished'); return
        if not proc.exists() or str(HERE/'session.py').encode() not in proc.read_bytes() or a.run.encode() not in proc.read_bytes():
            raise RuntimeError('Supervisor identity mismatch; not signalling reused PID')
        os.kill(pid,signal.SIGTERM)
        for _ in range(240):
            if json.loads(state.read_text()).get('finished_at_unix'): print(state.read_text()); return
            time.sleep(.25)
        raise RuntimeError('Still finalizing; inspect session log')
    if a.mode in ['start','run','worker']:
        if '--run' not in command or command[command.index('--run')+1:command.index('--run')+2] != [a.run]:
            p.error('Experiment --run must match session --run')
        if '--workspace' not in command or Path(command[command.index('--workspace')+1]).resolve()!=root:
            p.error('Experiment workspace must match session workspace')
    if a.mode=='start':
        out.mkdir(parents=True,exist_ok=False)
        args=[sys.executable,str(HERE/'session.py'),'worker','--workspace',str(root),'--run',a.run,'--interval',str(a.interval)]
        if a.batch_id: args+=['--batch-id',a.batch_id]
        if a.no_record: args+=['--no-record']
        with (out/'supervisor.log').open('w') as log:
            proc=subprocess.Popen(args+['--']+command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'supervisor_pid':proc.pid,'session':str(state)})); return
    if a.mode=='run': out.mkdir(parents=True,exist_ok=False)
    if not command: p.error('Provide the experiment command after --')
    record={'run':a.run,'batch_id':a.batch_id,'supervisor_pid':os.getpid(),'command':command,
            'started_at_unix':time.time(),'status':'starting','output':str(root/'outputs/thor_project'/a.run),
            'memory_method':'/proc child VmRSS/VmHWM; MemTotal-MemAvailable; sampled, system includes other processes; CUDA allocator peaks in per-request/case results',
            'sample_interval_seconds':a.interval,'gui_recorded':not a.no_record}
    write_json(state,record)
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    capture=[sys.executable,str(HERE/'capture.py')]
    capture_args=['--workspace',str(root),'--output',str(out/'capture'),'--policy-run',a.run]
    child=None; errors=[]; maxima={}; rc=1
    try:
        if not a.no_record: subprocess.run(capture+['start']+capture_args,check=True,stdout=subprocess.DEVNULL)
        if stopped: raise RuntimeError('Stopped before command launch')
        with (out/'console.log').open('w') as log, (out/'memory.jsonl').open('w') as mem:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            record.update(child_pid=child.pid,status='running');write_json(state,record)
            stop_time=None
            while child.poll() is None:
                row=sample(child.pid);mem.write(json.dumps(row)+'\n');mem.flush()
                for k,v in row.items():
                    if k!='unix' and v is not None: maxima[k]=max(maxima.get(k,0),v)
                if stopped and stop_time is None:
                    child.send_signal(signal.SIGINT);stop_time=time.monotonic()
                if stop_time and time.monotonic()-stop_time>30: child.kill()
                time.sleep(a.interval)
            rc=child.returncode
    except Exception as e:
        errors.append(f'{type(e).__name__}: {e}')
    finally:
        if child and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill();child.wait()
        if (out/'capture/thor_recording.json').exists():
            try: subprocess.run(capture+['stop']+capture_args,check=True,stdout=subprocess.DEVNULL)
            except Exception as e: errors.append('recording finalization: '+str(e))
        record.update(status='stopped' if stopped and not errors else 'success' if rc==0 and not errors else 'failed',
                      exit_code=rc,errors=errors,finished_at_unix=time.time(),maximum_observed_bytes=maxima)
        if (out/'capture/thor_screen.mp4').exists(): record['screen_video']=file_info(out/'capture/thor_screen.mp4')
        write_json(state,record)
    print(json.dumps(record,indent=2))
    if errors or (rc!=0 and not stopped): raise SystemExit(1)

if __name__=='__main__': main()
