"""Persist and verify Thor prediction before executing the corresponding action chunk."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from .artifacts import write_json
from .remote import fetch_file, ssh, remote_path, WORKSPACE


def probe_video(path):
    p=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(path)],text=True))
    s=p['streams'][0];a,b=map(int,s['r_frame_rate'].split('/'))
    return {'frames':int(s['nb_frames']),'fps':a/b,'width':s['width'],'height':s['height'],'duration':float(p['format']['duration'])}


def comparison_registration(raw_rgb, predicted_rgb):
    """Validate the known decoder crop using frame zero only; never stretch frames."""
    import cv2
    if raw_rgb.ndim != 3 or predicted_rgb.ndim != 3 or raw_rgb.shape[2] != 3 or predicted_rgb.shape[2] != 3:
        raise ValueError('Expected RGB registration images')
    if raw_rgb.shape == predicted_rgb.shape:
        return {'mode':'native_equal_dimensions','crop_xy':[0,0],
                'width':raw_rgb.shape[1],'height':raw_rgb.shape[0],
                'limit':'Equal dimensions alone do not establish camera calibration'}
    if raw_rgb.shape != (540,640,3) or predicted_rgb.shape != (528,640,3):
        raise ValueError('Unsupported prediction geometry; explicit registration review required')
    mae=lambda a,b:float(np.mean(np.abs(a.astype(np.float32)-b.astype(np.float32))))
    crops=[mae(raw_rgb[y:y+528],predicted_rgb) for y in range(13)]
    resize=mae(cv2.resize(raw_rgb,(640,528),interpolation=cv2.INTER_AREA),predicted_rgb)
    if np.argmin(crops)!=0 or crops[0]>12 or min(crops[1:]+[resize])-crops[0]<0.5:
        raise ValueError('Initial frame does not establish unambiguous top-left crop; review alignment')
    return {'mode':'verified_initial_frame_top_left_common_region','crop_xy':[0,0],
            'width':640,'height':528,'actual_bottom_pixels_excluded':12,
            'candidate_crop_mae':crops,'resize_mae':resize,'fit_frames':[0],
            'limit':'Frame-zero correspondence only; not independent camera calibration or physical accuracy'}


def first_rgb(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        ok,frame=cap.read()
        if not ok:raise ValueError('Missing first video frame')
        return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    finally:cap.release()


class PredictionReceiver:
    def __init__(self,episode,horizon,expected_fps,*,source='response',fetcher=fetch_file):
        self.episode=Path(episode);self.horizon=horizon;self.fps=expected_fps;self.source=source;self.fetcher=fetcher;self.receipt=None

    def __call__(self,response,request_id,actions,request):
        if self.receipt is not None:raise ValueError('Prospective comparison permits exactly one policy request')
        if not isinstance(response,dict) or len(actions)<self.horizon:raise ValueError('Action chunk shorter than comparison horizon')
        if response.get('session_id')!=request['session_id']:raise ValueError('Prediction/action session mismatch')
        if self.source=='response':
            artifact=response.get('artifacts')
            if artifact is not None:
                if not isinstance(artifact,dict):raise ValueError('Invalid Thor artifacts manifest')
                for key in ('run_id','request_id','session_id','policy_seed'):
                    if artifact.get(key)!=response.get(key):raise ValueError('Artifact/response identity mismatch: '+key)
                descriptor=artifact.get('prediction')
                action_file=artifact.get('action')
                if not isinstance(descriptor,dict):raise ValueError('Thor prediction metadata unavailable; action withheld')
                if not isinstance(action_file,dict) or not re.fullmatch(r'[0-9a-f]{64}',str(action_file.get('sha256',''))):
                    raise ValueError('Action artifact checksum required')
                remote_path(action_file['path'])
                action_target=self.episode/(request_id+'_thor_actions.npy')
                self.fetcher(action_file['path'],action_target,action_file['sha256'])
                saved_actions=np.load(action_target,allow_pickle=False)
                if list(saved_actions.shape)!=action_file.get('shape'):raise ValueError('Action artifact shape mismatch')
                np.testing.assert_array_equal(saved_actions.astype(np.float32),actions)
                write_json(self.episode/(request_id+'_thor_artifacts.json'),artifact)
            else:
                descriptor=response.get('prediction')
            if not isinstance(descriptor,dict):raise ValueError('Thor prediction metadata unavailable; action withheld')
            descriptor={k:descriptor[k] for k in ('path','sha256','frames','fps')}
        else:
            server_run=response.get('run_id');rid=response.get('request_id')
            if not isinstance(server_run,str) or not re.fullmatch(r'[A-Za-z0-9_+-]+',server_run) or type(rid)is not int or rid<0:
                raise ValueError('Missing safe server request IDs')
            video=remote_path(f'{WORKSPACE}/outputs/thor_project/{server_run}/requests/{rid:06d}/prediction.mp4')
            runtime=f'{WORKSPACE}/outputs/thor_project/{server_run}/runtime.json'
            code="import pathlib,json,hashlib,sys; p=pathlib.Path(sys.argv[1]); r=json.loads(pathlib.Path(sys.argv[2]).read_text()); assert r['decode_video']; print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fps':r['conditioning_fps']}))"
            descriptor=json.loads(ssh(['python3','-c',code,video,runtime]))
        if not re.fullmatch(r'[0-9a-f]{64}',str(descriptor.get('sha256',''))):raise ValueError('Prediction SHA256 required')
        remote_path(descriptor['path'])
        target=self.episode/(request_id+'_prediction.mp4')
        digest=self.fetcher(descriptor['path'],target,descriptor['sha256'])
        probe=probe_video(target)
        if abs(probe['fps']-self.fps)>1e-6 or abs(float(descriptor['fps'])-self.fps)>1e-6:
            raise ValueError('Prediction/control FPS mismatch')
        if probe['frames']<self.horizon+1:raise ValueError('Prediction too short for initial frame plus actions')
        if self.source=='response' and (type(descriptor['frames'])is not int or descriptor['frames']!=probe['frames']):
            raise ValueError('Prediction frame-count metadata mismatch')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(target),'-f','null','-'],check=True,capture_output=True)
        self.receipt={'request_id':request_id,'server_run':response.get('run_id'),'server_request_id':response.get('request_id'),
                      'session_id':request['session_id'],'metadata_source':self.source,'remote':descriptor,
                      'local_video':str(target),'sha256':digest,'probe':probe,'requested_horizon':self.horizon,
                      'raw_actions_sha256':hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
                      'saved_at_ns':time.time_ns(),'scope':'joint action/video prediction, not arbitrary-action conditioning',
                      'frame_alignment_assumption':'generated frame 0 corresponds to input observation; requires visual verification'}
        write_json(self.episode/'prediction_receipt.json',self.receipt)
        return self.receipt

    def finalize(self):
        if self.receipt is None:raise ValueError('No saved prediction')
        rows=[json.loads(s) for s in (self.episode/'applied_actions.jsonl').read_text().splitlines()]
        if not rows or rows[0].get('action_started_at_ns',0)<=self.receipt['saved_at_ns']:raise ValueError('Prediction must be saved BEFORE first action')
        if any(r['request_id']!=self.receipt['request_id'] or r['chunk_index']!=i for i,r in enumerate(rows)):
            raise ValueError('Actions replanned or not the saved chunk prefix')
        raw=np.load(self.episode/(self.receipt['request_id']+'_actions.npy'))
        expected=raw[:len(rows)].copy();expected[:,-1]=expected[:,-1]>.5
        np.testing.assert_array_equal(expected,np.asarray([r['action'] for r in rows],dtype=np.float32))
        actual=self.episode/'policy_view.mp4';pred=Path(self.receipt['local_video'])
        actual_probe=probe_video(actual)
        if actual_probe['frames']!=len(rows)+1:raise ValueError('Actual video/action alignment failed')
        if abs(actual_probe['fps']-self.fps)>1e-6:raise ValueError('Actual/control FPS mismatch')
        pred_first=first_rgb(pred);actual_first=first_rgb(actual)
        observation=self.episode/(self.receipt['request_id']+'_observation.npz')
        if pred_first.shape!=actual_first.shape and not observation.exists():
            raise ValueError('Original observation required for unequal-geometry registration')
        raw=np.load(observation)['image'] if observation.exists() else actual_first
        if raw.shape!=actual_first.shape:raise ValueError('Observation/actual camera geometry mismatch')
        if float(np.mean(np.abs(raw.astype(float)-actual_first.astype(float))))>12:
            raise ValueError('Actual frame zero differs from recorded observation')
        registration=comparison_registration(raw,pred_first)
        count=len(rows)+1;size=f"{registration['width']}:{registration['height']}:0:0"
        filters=f"[0:v]trim=end_frame={count},setpts=PTS-STARTPTS,crop={size},setsar=1,drawtext=text='PREDICTED':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7[p];[1:v]setpts=PTS-STARTPTS,crop={size},setsar=1,drawtext=text='ACTUAL SIMULATION':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7[a];[p][a]hstack=shortest=1[v]"
        subprocess.run(['ffmpeg','-v','error','-nostdin','-filter_complex_threads','2','-i',str(pred),'-i',str(actual),
                        '-filter_complex',filters,'-map','[v]','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20',
                        '-movflags','+faststart',str(self.episode/'prospective_comparison.mp4')],check=True)
        result={**self.receipt,'applied_steps':len(rows),'compared_frames':count,'initial_frame_included':True,
                'prediction_saved_before_action':True,'same_action_prefix_verified':True,
                'shortened_by_termination':len(rows)<self.horizon,'actual_video_probe':actual_probe,
                'spatial_registration':registration,'human_alignment_review':None,'human_prediction_quality':None}
        write_json(self.episode/'prospective_comparison.json',result)
        return result
