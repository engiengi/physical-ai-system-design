"""Instrument official server methods without changing inputs, RNG, transforms or outputs."""
import argparse,dataclasses,json,sys,time,os
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ.get('COSMOS_THOR_CODE', str(Path(__file__).resolve().parents[3] / 'thor')))
from common import configure,gpu_lock,new_run,write_json,pack
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--interval',choices=['on','off'],required=True);p.add_argument('--fixed-seed',type=int)
a=p.parse_args();root=configure(os.environ.get('COSMOS_THOR_WORKSPACE', str(Path.home() / 'cosmos_ws')));lock=gpu_lock(root);out=new_run(root,a.run)
framework=root/'external/cosmos-framework-reference-20261005';sys.path.insert(0,str(framework))
sys.path.insert(0,str(root/'external/reference-python-20261005'))
(out/'pid').write_text(str(os.getpid()))
import subprocess
from cosmos_framework.scripts import action_policy_server_robolab as official
args=official.RobolabServerArgs(checkpoint_path=str(root/'models/Cosmos3-Edge-Policy-DROID'),
 output_dir=out/'native',host='0.0.0.0',port=8000,seed=1200 if a.fixed_seed is None else a.fixed_seed,deterministic_seed=a.fixed_seed is not None,format_prompt_as_json=True,
 guidance_interval=[960,1001] if a.interval=='on' else None)
original_init=official.RobolabPolicyService.__init__;original_infer=official.RobolabPolicyService.infer
original_seed=official.RobolabPolicyService._next_seed
def init(self,args):
 original_init(self,args)
 self.audit_counter=0;self.audit_seed=None
 write_json(out/'runtime.json',{'framework_revision':subprocess.check_output(['git','-C',str(framework),'rev-parse','HEAD'],text=True).strip(),
  'policy':dataclasses.asdict(self.cfg),'args':args.model_dump(mode='json'),'instrumentation':'Read-only logging around original init/infer/next_seed; official serve and WebSocket implementation unchanged'})
def seed(self):
 value=original_seed(self);self.audit_seed=value;return value
def infer(self,obs):
 i=self.audit_counter;self.audit_counter+=1;d=out/'requests'/f'{i:06d}';d.mkdir(parents=True)
 (d/'observation.msgpack').write_bytes(pack(obs));t=time.perf_counter()
 try:
  result=original_infer(self,obs);arr=np.asarray(result['action']);np.save(d/'action.npy',arr)
  write_json(d/'result.json',{'request_id':i,'status':'success','inference_seconds':time.perf_counter()-t,'action_shape':list(arr.shape),'seed':self.audit_seed,'prompt':obs['prompt']})
  return result
 except BaseException as e:
  write_json(d/'result.json',{'request_id':i,'status':'error','error':repr(e)});raise
official.RobolabPolicyService.__init__=init;official.RobolabPolicyService.infer=infer;official.RobolabPolicyService._next_seed=seed
official.serve(args)
