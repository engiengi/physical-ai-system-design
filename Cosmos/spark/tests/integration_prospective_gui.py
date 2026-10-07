"""GUI protocol fixture: NOT Cosmos inference or prediction quality evaluation.

Uses actual RoboLab, WebSocket, SSH artifact transfer and GUI recording. The
fixture returns held joint positions and a conspicuously labeled static video.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid
import cv2
import numpy as np
from test_client import policy_server
from cosmos_spark.remote import HOST,WORKSPACE,ssh

root=Path(__file__).resolve().parents[1]
run_id='spark_s4_PROTOCOL_FIXTURE_'+uuid.uuid4().hex[:8]
output=root/'reports/spark_extensions_20261005'/run_id
output.mkdir()
remote=WORKSPACE+'/data/spark_protocol_fixtures/'+run_id
ssh(['mkdir','-p',remote])

def respond(request):
    image=cv2.cvtColor(request['observation/image'],cv2.COLOR_RGB2BGR)
    cv2.putText(image,'PROTOCOL FIXTURE - NOT COSMOS',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
    cv2.imwrite(str(output/'fixture.png'),image)
    video=output/'fixture.mp4'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-loop','1','-framerate','15','-i',str(output/'fixture.png'),
                    '-frames:v','7','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(video)],check=True)
    subprocess.run(['scp','-q','-o','BatchMode=yes',str(video),HOST+':'+remote+'/fixture.mp4'],check=True)
    action=np.concatenate([request['observation/joint_position'],[0.0]]).astype(np.float32)
    return {'action':np.tile(action,(6,1)),'policy_seed':request['policy_seed'],
            'run_id':'PROTOCOL_FIXTURE','request_id':0,'session_id':request['session_id'],
            'prediction':{'path':remote+'/fixture.mp4','sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'frames':7,'fps':15}}

with policy_server(respond) as (uri,requests):
    with (output/'launcher.log').open('w') as log:
        cp=subprocess.run([sys.executable,'-m','cosmos_spark.session','--run-id',run_id,'--mode','predict',
            '--uri',uri,'--policy-seed','123','--execute-horizon','6','--timeout','60',
            '--kit_args','--/app/window/width=1600 --/app/window/height=900'],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    assert cp.returncode==0,(cp.returncode,output/'launcher.log')
ep=root/'runs'/run_id/'episode_0000'
result=json.loads((ep/'episode.json').read_text())
comparison=json.loads((ep/'prospective_comparison.json').read_text())
assert result['status']=='diagnostic_complete' and result['steps']==6
assert len(requests)==1 and comparison['prediction_saved_before_action'] and comparison['same_action_prefix_verified']
report={'passed':True,'kind':'protocol fixture, NOT Cosmos prediction','run_id':run_id,'requests':len(requests),
        'result':result,'gui':str(root/'captures'/run_id/'gui_wallclock.mp4')}
(output/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'passed':True,'run_id':run_id,'report':str(output/'validation.json')}))
