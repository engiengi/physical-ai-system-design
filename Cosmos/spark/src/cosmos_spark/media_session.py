"""Recorded Thor media jobs: checked inputs, persistent job IDs and resumable collection."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
from .remote import HOST,WORKSPACE,ROOT,ssh,remote_path,fetch_file
from .artifacts import write_json
from .settings import CODE, PYTHON
MANIFESTS={'analyze':'analysis.json','generate':'generation.json'}

def input_files(package,command):
    manifest=package/MANIFESTS[command];cases=json.loads(manifest.read_text())['cases'];files={manifest.name};ids=set()
    if not cases:raise ValueError('No input cases')
    for c in cases:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',c['id']) or c['id'] in ids:raise ValueError('Unique safe case IDs required')
        ids.add(c['id'])
        if any(k in c for k in ('label','label_source','label_kind','evidence','reviewer','simulator_status')):
            raise ValueError('Reference labels must stay on Spark')
        if command=='generate' and c.get('mode','i2v')!='i2v':raise ValueError('This transfer supports i2v only')
        path=Path(c['source'])
        if not re.fullmatch(r'media/[A-Za-z0-9_.-]+',str(path)) or not (package/path).resolve().is_relative_to(package.resolve()):
            raise ValueError('Use a simple portable media/ file path')
        if not (package/path).is_file():raise FileNotFoundError(path)
        files.add(str(path))
    return [{'path':s,'sha256':hashlib.sha256((package/s).read_bytes()).hexdigest()} for s in sorted(files)]

def start(package,command,run,folder,prompt_version='plain-json-v2',helper=None):
    if prompt_version not in ('v1','plain-json-v2','terminal-outcome-v1','terminal-outcome-raw-v2','terminal-evidence-v3'):raise ValueError('Unsupported analysis prompt version')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',run):raise ValueError('Safe run required')
    if helper is None and prompt_version=='terminal-evidence-v3':helper=ROOT/'scripts/thor/media_improvement.py'
    if helper is not None and not Path(helper).is_file():raise FileNotFoundError(helper)
    files=input_files(package,command)
    if prompt_version in ('terminal-outcome-v1','terminal-outcome-raw-v2','terminal-evidence-v3'):
        cases=json.loads((package/MANIFESTS[command]).read_text())['cases']
        if command!='analyze' or any(c.get('evaluation_phase')!='terminal_outcome' or c.get('trial_ended') is not True for c in cases):
            raise ValueError('Terminal outcome requires explicitly ended-trial analysis inputs')
    folder.mkdir(parents=True,exist_ok=False)
    remote=WORKSPACE+'/data/spark_scenarios/'+run
    job={'run':run,'command':command,'package':str(package),'files':files,'remote_input':remote,'status':'preparing'}
    write_json(folder/'job.json',job)
    ssh(['mkdir','-p',WORKSPACE+'/data/spark_scenarios']);ssh(['mkdir',remote]);ssh(['mkdir',remote+'/media'])
    for f in files:subprocess.run(['scp','-q','-o','BatchMode=yes',str(package/f['path']),HOST+':'+remote+'/'+f['path']],check=True,timeout=180)
    code='import pathlib,json,hashlib,sys; r=pathlib.Path(sys.argv[1]); fs=json.loads(sys.argv[2]); assert all(hashlib.sha256((r/f["path"]).read_bytes()).hexdigest()==f["sha256"] for f in fs); print("verified")'
    job['input_verification']=ssh(['python3','-c',code,remote,json.dumps(files)]).strip()
    ssh([PYTHON,CODE+'/scenarios.py','--manifest',remote+'/'+MANIFESTS[command],
         '--command',command,'--output',remote+'/preflight.json'])
    fetch_file(remote+'/preflight.json',folder/'preflight.json')
    release_code=ssh(['readlink','-f',CODE]).strip()
    entrypoint=release_code+'/media.py'
    if helper is not None or prompt_version in ('terminal-outcome-v1','terminal-outcome-raw-v2','terminal-evidence-v3'):
        helper=Path(helper) if helper is not None else ROOT/'scripts/thor/media_terminal_outcome.py'
        entrypoint=remote+'/media_terminal_outcome.py'
        subprocess.run(['scp','-q','-o','BatchMode=yes',str(helper),HOST+':'+entrypoint],check=True,timeout=60)
        digest=hashlib.sha256(helper.read_bytes()).hexdigest()
        remote_digest=ssh(['python3','-c','import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())',entrypoint]).strip()
        if digest!=remote_digest:raise ValueError('Analysis helper transfer hash mismatch')
        job['analysis_helper']={'path':entrypoint,'sha256':digest,'prompt_version':prompt_version}
    cmd=['env','COSMOS_THOR_CODE='+release_code,PYTHON,'-u',entrypoint,command,'--workspace',WORKSPACE,'--run',run,'--manifest',remote+'/'+MANIFESTS[command]]
    if command=='analyze':cmd+=['--prompt-version',prompt_version]
    session=['env','DISPLAY=:1','XAUTHORITY=/run/user/1000/gdm/Xauthority','DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus',
             PYTHON,release_code+'/session.py','start','--workspace',WORKSPACE,'--run',run,'--batch-id',run,'--',*cmd]
    job['experiment_command']=cmd
    job['start_response']=json.loads(ssh(session));job['status']='started';write_json(folder/'job.json',job)
    return job

def status(folder):
    job=json.loads((folder/'job.json').read_text())
    code='import pathlib,json,sys; r=pathlib.Path(sys.argv[1]); name=sys.argv[2]; print(json.dumps({k:json.loads(p.read_text()) if p.exists() else None for k,p in {"session":r/"sessions"/name/"session.json","progress":r/"outputs/thor_project"/name/"progress.json"}.items()}))'
    result=json.loads(ssh(['python3','-c',code,WORKSPACE,job['run']]))
    write_json(folder/'remote_status.json',result);return result

def collect(folder):
    state=status(folder)
    if not (state.get('session') or {}).get('finished_at_unix'):raise RuntimeError('Job is still active; collect after finalization')
    job=json.loads((folder/'job.json').read_text());run=job['run']
    for category,remote in [('output',WORKSPACE+'/outputs/thor_project/'+run),('session',WORKSPACE+'/sessions/'+run)]:
        code='''import pathlib,hashlib,json,sys
root=pathlib.Path(sys.argv[1]);rows=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 if p.suffix not in ('.json','.jsonl','.txt','.log','.mp4','.png'):continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 rows.append({'path':str(p.relative_to(root)),'sha256':h.hexdigest()})
print(json.dumps(rows))'''
        listing=json.loads(ssh(['python3','-c',code,remote],timeout=120))
        for f in listing:
            source=remote_path(remote+'/'+f['path']);target=folder/category/f['path']
            if not target.resolve().is_relative_to((folder/category).resolve()):raise ValueError('Unsafe result path')
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Existing result differs; not overwritten')
            else:fetch_file(source,target,f['sha256'])
        write_json(folder/(category+'_transfer.json'),listing)
    job['status']='collected';job['remote_result']=state['session']['status'];write_json(folder/'job.json',job);return job

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    s=sub.add_parser('start');s.add_argument('command',choices=['analyze','generate']);s.add_argument('--package',type=Path,required=True);s.add_argument('--run',required=True);s.add_argument('--folder',type=Path,required=True)
    s.add_argument('--helper',type=Path)
    s.add_argument('--prompt-version',choices=['v1','plain-json-v2','terminal-outcome-v1','terminal-outcome-raw-v2','terminal-evidence-v3'],default='plain-json-v2')
    for name in ('status','collect'):sub.add_parser(name).add_argument('folder',type=Path)
    a=p.parse_args()
    result=start(a.package.resolve(),a.command,a.run,a.folder.resolve(),a.prompt_version,a.helper) if a.op=='start' else status(a.folder) if a.op=='status' else collect(a.folder)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
