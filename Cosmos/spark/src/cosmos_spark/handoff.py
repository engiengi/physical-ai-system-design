"""Transfer only analysis inputs to Thor; reference labels stay on Spark."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from .remote import HOST,WORKSPACE,ssh
from .artifacts import write_json


def prepare_inputs(package):
    cases=json.loads((package/'analysis.json').read_text())['cases']
    if not cases:raise ValueError('No reviewed/selected cases; human review is still pending')
    files=['analysis.json']
    for c in cases:
        if any(k in c for k in ('label','label_source','evidence','reviewer','simulator_status')):raise ValueError('Ground truth must not be uploaded as model input')
        path=Path(c['source'])
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='media':raise ValueError('Expected portable media/ path')
        resolved=(package/path).resolve()
        if not resolved.is_relative_to(package.resolve()) or not resolved.is_file():raise ValueError('Invalid media path')
        files.append(str(path))
    return [{'path':p,'sha256':hashlib.sha256((package/p).read_bytes()).hexdigest()} for p in sorted(set(files))]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('package',type=Path);p.add_argument('--name',required=True)
    p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',a.name):p.error('Safe unique transfer name required')
    package=a.package.resolve();items=prepare_inputs(package);dest=WORKSPACE+'/data/spark_inputs/'+a.name
    report={'remote_manifest':dest+'/analysis.json','files':items,'ground_truth_transferred':False}
    if a.plan_only:print(json.dumps(report,indent=2));return
    ssh(['mkdir','-p',WORKSPACE+'/data/spark_inputs']);ssh(['mkdir',dest]);ssh(['mkdir',dest+'/media'])
    for item in items:
        subprocess.run(['scp','-q','-o','BatchMode=yes',str(package/item['path']),HOST+':'+dest+'/'+item['path']],check=True,timeout=180)
    code='import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); items=json.loads(sys.argv[2]); assert all(hashlib.sha256((root/i["path"]).read_bytes()).hexdigest()==i["sha256"] for i in items); print("verified")'
    report['verification']=ssh(['python3','-c',code,dest,json.dumps(items)]).strip()
    write_json(package/('transfer_'+a.name+'.json'),report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
