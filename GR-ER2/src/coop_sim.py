"""Two-robot Isaac Sim runtime with non-blocking, individually addressable jobs."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--scenario',choices=['dual_franka','transport'],required=True)
p.add_argument('--gui',action='store_true');p.add_argument('--stream',action='store_true');p.add_argument('--stream-ip',default='')
args,unknown=p.parse_known_args();sys.argv=[sys.argv[0]]+unknown
if args.stream_ip: sys.argv += [f'--/exts/omni.kit.livestream.app/primaryStream/publicIp={args.stream_ip}']
from isaacsim import SimulationApp
app=SimulationApp({'headless':not args.gui,'hide_ui':not(args.gui or args.stream),'width':1280,'height':720,'renderer':'RaytracedLighting'},
                  experience='/isaac-sim/apps/isaacsim.exp.full.streaming.kit' if args.stream else '')
import numpy as np
from PIL import Image
from pxr import UsdLux
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.viewports import set_camera_view
from common import ROOT,read_json,write_json,safe_child,uid,pixel
from coop_rules import validate_job,public_job,ROBOTS
from coop_scenes import DualFranka
from transport_scene import Transport
from transport_geometry import refine_block_top

runtime=ROOT/'runtime'
for folder in ['queue','replies','coop_observations']: (runtime/folder).mkdir(parents=True,exist_ok=True)
for path in (runtime/'queue').glob('*.json'):
    write_json(runtime/'replies'/path.name,{'error':'Simulator restarted'});path.unlink()
for name in ['STOP','SHUTDOWN']: (runtime/name).unlink(missing_ok=True)
write_json(runtime/'status.json',{'ready':False,'world_scenario':args.scenario,'updated':time.time()})
cls=DualFranka if args.scenario=='dual_franka' else Transport
dt=1/cls.hz
world=World(stage_units_in_meters=1,physics_dt=dt,rendering_dt=dt)
record=None;jobs={};step=0


def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def event(kind,**data):
    if record: append(record/'events.jsonl',{'kind':kind,'wall_time':time.time(),'simulation_time':world.current_time,**data})


scene=cls(world,event)
UsdLux.DomeLight.Define(world.stage,'/World/Light').CreateIntensityAttr(1800)
cameras={name:Camera('/World/Camera'+name,resolution=(640,480)) for name in scene.camera_eyes}
side=Camera('/World/Side',resolution=(960,640))
for name,cam in cameras.items():
    set_camera_view(eye=np.array(scene.camera_eyes[name]),target=np.array(scene.camera_targets[name]),camera_prim_path=cam.prim_path)
set_camera_view(eye=np.array(scene.overview_eye),target=np.array(scene.look),camera_prim_path=side.prim_path)
set_camera_view(eye=np.array(scene.overview_eye),target=np.array(scene.look))
world.reset();scene.initialize()
scene.reset('normal')
for cam in [*cameras.values(),side]:
    cam.initialize();cam.set_clipping_range(.02,30);cam.set_focal_length(2.4);cam.set_horizontal_aperture(2.4)
    if cam in cameras.values(): cam.add_distance_to_image_plane_to_frame()
render_every=max(1,cls.hz//30);record_every=cls.hz//5


def save(cam,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    rgb=np.asarray(cam.get_rgba())[:,:,:3].astype(np.uint8)
    tmp=path.with_name(path.stem+'.tmp'+path.suffix);Image.fromarray(rgb).save(tmp);os.replace(tmp,path)


def observation(camera_id):
    if camera_id not in cameras:raise ValueError('camera_id must be overview, A, B')
    obs=uid();dest=runtime/'coop_observations'/obs;dest.mkdir()
    cam=cameras[camera_id];save(cam,dest/'image.jpg')
    meta={'observation_id':obs,'camera_id':camera_id,'wall_time':time.time(),'simulation_time':world.current_time}
    if camera_id != 'Spot':
        np.save(dest/'depth.npy',cam.get_depth())
        meta.update(intrinsics=cam.get_intrinsics_matrix().tolist(),camera_pose=[x.tolist() for x in cam.get_world_pose()])
    write_json(dest/'meta.json',meta)
    return {k:meta[k] for k in ['observation_id','camera_id','wall_time','simulation_time']}|{'image':str((dest/'image.jpg').relative_to(ROOT))}


def targets(arguments):
    dest=runtime/'coop_observations'/arguments['observation_id'];meta=read_json(dest/'meta.json')
    if time.time()-meta['wall_time']>180:raise ValueError('Expired observation; observe again')
    cam=cameras[meta['camera_id']];depth=np.load(dest/'depth.npy');result=[]
    for key in ['pick','place']:
        x,y=pixel(arguments[key],640,480);ix,iy=round(x),round(y)
        values=depth[max(0,iy-1):iy+2,max(0,ix-1):ix+2]
        valid=values[np.isfinite(values)&(values>.01)&(values<4)]
        if len(valid)<3:raise ValueError('Invalid depth')
        selected_depth=float(np.median(valid))
        if args.scenario=='transport' and key=='pick':
            requested=[x,y]
            x,y,selected_depth=refine_block_top(depth,x,y)
            event('depth_point_refinement',observation_id=arguments['observation_id'],requested_pixel=requested,refined_pixel=[x,y],method='local raised depth plateau; no object truth')
        xyz=cam.get_world_points_from_image_coords(np.array([[x,y]]),np.array([selected_depth]))[0]
        if args.scenario=='transport':
            if not .70<=xyz[2]<=.95:raise ValueError('Select material top / empty tray or conveyor surface')
            xyz[2] += -.025 if key=='pick' else .025
        elif key=='pick':
            if not .035<=xyz[2]<=.10:raise ValueError('Pick the visible block TOP center; background/robot excluded')
            xyz[2]-=.025
        else:
            if not -.01<=xyz[2]<=.035:raise ValueError('Place on an empty tray/exchange floor')
            xyz[2]+=.025
        result.append(xyz)
    return result


def finish(job_id,status):
    jobs[job_id].update(status=status,ended_wall=time.time(),ended_sim=world.current_time)
    event('job_end',job=public_job(jobs[job_id]))


def process(req):
    global record,jobs,frame,start_sim
    command=req['command']
    if time.time()-req['created']>300:raise ValueError('Expired request')
    if command=='shutdown':
        (runtime/'SHUTDOWN').touch();return {'shutdown':True}
    if command in ('hold','stop'):
        for r,s in list(scene.active.items()):
            job=s['job']['job_id'];scene.cancel(r);finish(job,'cancelled')
        return {'stopped':True}
    if command=='coop_reset':
        if record:raise ValueError('Stop recording first')
        if req.get('condition') not in (('normal','exception') if args.scenario=='dual_franka' else ('normal','load_failure','obstacle')):raise ValueError('Invalid condition')
        scene.reset(req['condition']);jobs={};return {'reset':True}
    if command=='record_start':
        if record:raise ValueError('Already recording')
        record=safe_child(ROOT/'outputs',req['output'].removeprefix('outputs/'));record.mkdir(parents=True,exist_ok=True)
        frame=0;start_sim=world.current_time
        write_json(record/'physics_config.json',{'scenario':args.scenario,'physics_hz':cls.hz,'record_fps':5,
            'state_access':'Controllers use simulator state/depth; truth.jsonl is evaluator-only',
            'isaac_image':'nvcr.io/nvidia/isaac-sim:6.0.0-dev2',
            'controller':'Isaac Sim Franka PickPlaceController/RMPFlow' if isinstance(scene,DualFranka) else 'Isaac Sim SpotFlatTerrainPolicy, policy decimation 10, waypoint steering/raycast safety',
            'plant_process':'dynamic payload with physical tray/gripper contacts; conveyors stopped at handoff'})
        return {'recording':True}
    if command=='record_stop':
        if not record:raise ValueError('No recording')
        write_json(record/'jobs.json',jobs)
        write_json(record/'recording.json',{'frames':frame,'fps':5,'simulation_seconds':world.current_time-start_sim})
        result={'recording':False,'frames':frame};record=None;return result
    if command=='observe':return observation(req.get('camera_id','overview'))
    if command=='get_job_status':
        job_id=req.get('job_id','all')
        if job_id=='all':return {'jobs':[public_job(v) for v in jobs.values()]}
        if job_id not in jobs:raise ValueError('Unknown job_id')
        return public_job(jobs[job_id])
    if command=='cancel_job':
        job=jobs[req['job_id']]
        if job['status']=='running':scene.cancel(job['robot_id']);finish(job['job_id'],'cancelled')
        return public_job(job)
    if command=='start_action':
        r,action,params=req['robot_id'],req['action'],req['arguments']
        validate_job(args.scenario,r,action,params)
        if args.scenario=='transport' and action=='pick_place':
            meta=read_json(runtime/'coop_observations'/params['observation_id']/'meta.json')
            if meta['camera_id']!=r:raise ValueError('Use the matching stationary camera A/B for arm pointing')
        if r in scene.active:raise ValueError('Robot busy; query its job or assign another robot')
        job={'job_id':uid(),'robot_id':r,'action':action,'arguments':params,'status':'running','started_wall':time.time(),'started_sim':world.current_time}
        scene.start(job,targets(params) if action=='pick_place' else None)
        jobs[job['job_id']]=job;event('job_start',job=public_job(job));return public_job(job)
    if command=='transport_smoke_action' and args.scenario=='transport':
        # Explicit controller diagnostic only. Never registered in ER2 tools.
        r=req['robot_id'];p,yaw=scene.pose();source=scene.payload.get_world_pose()[0]
        place=p+np.array([0,0,.285]) if r=='A' else np.array([4.4,-.65,.76])
        job={'job_id':uid(),'robot_id':r,'action':'pick_place','arguments':{'diagnostic_only':True},'status':'running','started_wall':time.time(),'started_sim':world.current_time}
        if r in scene.active:raise ValueError('Busy')
        scene.start(job,[source,place]);jobs[job['job_id']]=job;event('job_start',job=public_job(job));return public_job(job)
    if command=='coop_truth': # Smoke/offline evaluation only. Not registered as a model tool.
        return scene.truth()|{'camera_poses':{name:[x.tolist() for x in cam.get_world_pose()] for name,cam in cameras.items()}}
    raise ValueError('Unsupported command')


try:
    # Camera initialization may render before the controller starts. Restore
    # the initial pose afterwards and refresh Fabric at EVERY warm-up tick:
    # the Spot policy reads experimental-API proprioception from Fabric.
    scene.reset('normal')
    for warm in range(cls.hz*3):
        scene.tick(dt);world.step(render=warm%render_every==0,update_fabric=True)
    # Match the explicit reset used by every tested runner AFTER assets, camera
    # pipelines and policy buffers are fully warmed. This also makes opening
    # the standalone GUI safe without first submitting an experiment command.
    scene.reset('normal')
    print('GR_ER2_READY '+args.scenario,flush=True)
    while app.is_running() and not(runtime/'SHUTDOWN').exists():
        begin=time.monotonic()
        for job,status in scene.tick(dt):finish(job,status)
        if step%render_every==0:scene.update_cameras(cameras)
        world.step(render=step%render_every==0,update_fabric=True);step+=1
        if step%record_every==0:
            if record:
                for name,cam in cameras.items():save(cam,record/'frames'/name/f'{frame:06d}.jpg')
                save(side,record/'frames'/'side'/f'{frame:06d}.jpg')
                stamp={'frame':frame,'wall_time':time.time(),'simulation_time':world.current_time,'elapsed_sim':world.current_time-start_sim}
                append(record/'frames.jsonl',stamp);append(record/'truth.jsonl',{**stamp,**scene.truth()});frame+=1
            write_json(runtime/'status.json',{'ready':True,'world_scenario':args.scenario,'updated':time.time(),'busy':bool(scene.active)})
            for path in sorted((runtime/'queue').glob('*.json')):
                try:write_json(runtime/'replies'/path.name,process(read_json(path)))
                except Exception as exc:
                    traceback.print_exc();event('request_rejected',command=read_json(path).get('command'),reason=str(exc))
                    write_json(runtime/'replies'/path.name,{'error':str(exc)})
                finally:path.unlink(missing_ok=True)
        time.sleep(max(0,dt-(time.monotonic()-begin)))
except Exception:
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();raise
finally:
    write_json(runtime/'status.json',{'ready':False,'world_scenario':args.scenario,'updated':time.time()})
    app.close()
