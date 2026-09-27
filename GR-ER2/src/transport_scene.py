"""Physical Franka -> payload-carrying Spot -> Franka material transfer.

Controllers use calibrated geometry and proprioception. ER2 receives RGB only.
The payload is NEVER teleported or attached after reset. The carrier tray is a
light rigid attachment to Spot; parcel/tray/gripper interaction uses contacts.
"""
import math
import numpy as np
from pxr import UsdPhysics, PhysxSchema, Gf
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.policy.examples.robots.spot import SpotFlatTerrainPolicy
from isaacsim.core.deprecation_manager import import_module
from omni.physx import get_physx_scene_query_interface
from coop_scenes import box


class Transport:
    hz = 500
    look = [2, 0, .6]
    overview_eye = [6, -7, 5]
    camera_eyes = {'overview':[2,0,6.5], 'A':[.35,-.15,2.8], 'B':[4.35,-.15,2.8], 'Spot':[.9,0,1.0], 'cargo':[.45,0,1.8]}
    camera_targets = {'overview':[2,0,0], 'A':[.35,-.15,.65], 'B':[4.35,-.15,.65], 'Spot':[2,0,.6], 'cargo':[.45,0,.7]}
    docks = {'A':np.array([.45,.30]), 'B':np.array([4.45,.30])}

    def __init__(self, world, event):
        self.world, self.event = world, event
        self.torch = import_module('torch')
        world.scene.add_default_ground_plane()
        self.bases = {r:np.array([x,-.25,.72]) for r,x in [('A',0),('B',4)]}
        self.arms = {r:world.scene.add(Franka('/World/Franka'+r,name='franka'+r,position=p,
            orientation=np.array([1.,0.,0.,0.]))) for r,p in self.bases.items()}
        for r,x in [('A',0),('B',4)]:
            box(world,'Pedestal'+r,[x,-.25,.36],[.20,.20,.72],[.32,.37,.43])
            box(world,'Conveyor'+r,[x-.15,-.65,.66],[1.3,.36,.12],[.15,.19,.24])
            for i in range(10):
                box(world,f'Roller{r}{i}',[x-.72+.12*i,-.65,.722],[.055,.33,.004],[.48,.50,.53])
            box(world,'Endpoint'+r,[x+.40,-.65,.725],[.15,.33,.006],[.12,.55,.2] if r=='B' else [.85,.55,.05])
            box(world,'Dock'+r,[x+.45,.30,.002],[.95,.6,.004],[.14,.36,.65])
        self.initial_payload=np.array([.40,-.65,.758])
        self.payload=box(world,'Material',self.initial_payload,[.05]*3,[.9,.10,.03],True)
        self.spot=SpotFlatTerrainPolicy('/World/Carrier',position=[.45,.30,.65],orientation=[1.,0.,0.,0.])
        # Compound collider, attached as a rigid child of the robot trunk after asset load.
        stage=get_current_stage()
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/Carrier') and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateSleepThresholdAttr(0.)
        roots=[p for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Carrier/') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        body=next((p for p in roots if p.GetName() in ('body','base','trunk')),None)
        if body is None:raise RuntimeError('Cannot resolve Spot trunk: '+str([str(p.GetPath()) for p in roots]))
        self.body_path=str(body.GetPath())
        # Author collider shapes in trunk-local coordinates; no separate rigid body/mass.
        from pxr import UsdGeom
        for name,pos,size in [('Floor',[0,0,.23],[.38,.30,.018]),
                              ('Left',[0,.155,.265],[.40,.018,.07]),('Right',[0,-.155,.265],[.40,.018,.07]),
                              ('Front',[.20,0,.265],[.018,.30,.07]),('Rear',[-.20,0,.265],[.018,.30,.07])]:
            cube=UsdGeom.Cube.Define(stage,self.body_path+'/CarrierTray'+name)
            cube.CreateSizeAttr(1.);cube.AddTranslateOp().Set(Gf.Vec3d(*pos));cube.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(.1,.55,.2)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        self.barrier=box(world,'RouteBarrier',[2,4,.5],[.22,1.1,1.],[.95,.35,.02])
        self.active={};self.condition='normal';self.injected=False;self.ticks=0
        self.hold_target=self.docks['A'].copy();self.loaded_once=False;self.departed_loaded=False;self.arrived_loaded=False;self.unloaded=False

    def initialize(self):
        self.spot.initialize();self.spot.post_reset()
        # Policy initialize reapplies YAML articulation properties, overriding USD.
        self.spot.robot.set_sleep_thresholds([0.])
        self.home={r:b.get_joint_positions().copy() for r,b in self.arms.items()}
        for q in self.home.values():q[-2:]=.04
        self.ctrl={r:PickPlaceController(name='transport_'+r,gripper=b.gripper,robot_articulation=b,
            end_effector_initial_height=.98) for r,b in self.arms.items()}

    def pose(self):
        pos,q=self.spot.robot.get_world_poses();p,v=pos.numpy()[0],q.numpy()[0]
        return p,math.atan2(2*(v[0]*v[3]+v[1]*v[2]),1-2*(v[2]**2+v[3]**2))

    def reset(self,condition):
        self.hold_target=self.docks['A'].copy();self.condition=condition;self.injected=False;self.active.clear();self.ticks=0
        self.loaded_once=self.departed_loaded=self.arrived_loaded=self.unloaded=False
        self.spot.post_reset();self.spot._policy_counter=0;self.spot._previous_action.zero_()
        self.spot.robot.set_world_poses(positions=[[.45,.30,.65]],orientations=[[1.,0,0,0]])
        self.spot.robot.set_velocities(linear_velocities=[[0.,0,0]],angular_velocities=[[0.,0,0]])
        for r,b in self.arms.items():
            b.set_joint_positions(self.home[r]);b.set_joint_velocities(np.zeros(9));self.ctrl[r].reset()
            b.get_articulation_controller().apply_action(ArticulationAction(joint_positions=self.home[r]))
        self.payload.set_world_pose(self.initial_payload,np.array([1.,0,0,0]))
        self.payload.set_linear_velocity(np.zeros(3));self.payload.set_angular_velocity(np.zeros(3))
        self.barrier.set_world_pose(np.array([2.,4.,.5]))

    def start(self,job,targets=None):
        r=job['robot_id'];p,yaw=self.pose()
        if r=='Spot':
            if any(k in self.active for k in ['A','B']):raise ValueError('Arm handoff active; Spot must remain stopped')
            dest=job['arguments']['destination'];route=job['arguments']['route'];target=self.docks[dest]
            points=[target.tolist()] if route=='direct' else [[float(p[0]),1.55],[float(target[0]),1.55],target.tolist()]
            self.active[r]={'job':job,'route':points,'index':0,'start':self.world.current_time,'destination':dest}
            return
        if 'Spot' in self.active:raise ValueError('Carrier moving; wait for docking')
        if np.linalg.norm(p[:2]-self.docks[r])>.20 or abs(yaw)>.15:raise ValueError('Carrier must dock at this station before handoff')
        pick,place=map(np.array,targets)
        origin=0 if r=='A' else 4
        if any(not origin-.35<=v[0]<=origin+.7 or not -.85<=v[1]<=.5 or not .68<=v[2]<=.90 for v in [pick,place]):
            raise ValueError('Outside calibrated handoff workspace')
        fail=self.condition=='load_failure' and r=='A' and not self.injected
        if fail:
            self.injected=True;self.event('grasp_failure_injected',robot_id=r,job_id=job['job_id'],mechanism='fingers held open for first loading attempt')
        self.ctrl[r].reset()
        self.active[r]={'job':job,'pick':pick,'place':place,'start':self.world.current_time,'failure':fail,'home_ticks':0}

    def cancel(self,r):
        self.active.pop(r,None)
        if r=='Spot':self.hold_target=self.pose()[0][:2].copy()
        if r in self.arms:
            b=self.arms[r];b.get_articulation_controller().apply_action(ArticulationAction(joint_positions=b.get_joint_positions()))

    def loaded(self):
        p,yaw=self.pose();o=self.payload.get_world_pose()[0];delta=o-p
        local=np.array([math.cos(yaw)*delta[0]+math.sin(yaw)*delta[1],-math.sin(yaw)*delta[0]+math.cos(yaw)*delta[1]])
        return bool(abs(local[0])<.18 and abs(local[1])<.14 and .23<delta[2]<.34)

    def tick(self,dt):
        self.ticks+=1;done=[];p,yaw=self.pose();command=[0.,0.,0.]
        loaded=self.loaded()
        if loaded and 'A' not in self.active:self.loaded_once=True
        if self.loaded_once and loaded and p[0]>1:self.departed_loaded=True
        if self.departed_loaded and loaded and 'Spot' not in self.active and np.linalg.norm(p[:2]-self.docks['B'])<.20:self.arrived_loaded=True
        o=self.payload.get_world_pose()[0]
        self.unloaded=bool(self.arrived_loaded and 'B' not in self.active and abs(o[0]-4.4)<.065 and abs(o[1]+.65)<.13 and .73<o[2]<.82)
        s=self.active.get('Spot')
        if s:
            status=None
            if self.condition=='obstacle' and not self.injected and p[0]>.85:
                self.injected=True;self.barrier.set_world_pose(np.array([2.,.30,.5]))
                self.event('route_obstacle_inserted',robot_id='Spot',mechanism='physical barrier across direct lane')
            target=np.array(s['route'][s['index']]);delta=target-p[:2];distance=np.linalg.norm(delta)
            tolerance=.15 if s['index']==len(s['route'])-1 else .12
            if distance<tolerance:
                if s['index']+1<len(s['route']):s['index']+=1
                else:s['align']=True
            if s.get('align') and abs(yaw)<.08:
                status='arrived' if distance<.20 else 'docking_failed'
                self.hold_target=p[:2].copy()
            desired=0. if s.get('align') else math.atan2(delta[1],delta[0])
            error=math.atan2(math.sin(desired-yaw),math.cos(desired-yaw))
            command=[min(.6,max(.35,distance))*max(0.,math.cos(error)) if abs(error)<.5 and distance>=tolerance and not s.get('align') else 0.,
                     0.,float(np.clip(error*1.8,-.8,.8))]
            speed=np.linalg.norm(command[:2])
            if speed>.03:
                direction=np.array([math.cos(yaw)*command[0]-math.sin(yaw)*command[1],math.sin(yaw)*command[0]+math.cos(yaw)*command[1],0.]);direction/=np.linalg.norm(direction)
                for lateral in [-.19,0,.19]:
                    origin=p+direction*.50+np.array([-direction[1]*lateral,direction[0]*lateral,0]);origin[2]=.52
                    hit=get_physx_scene_query_interface().raycast_closest(tuple(map(float,origin)),tuple(map(float,direction)),.38)
                    if hit.get('hit') and '/Carrier' not in str(hit.get('rigidBody','')):status='blocked';break
            if p[2]<.25:status='fallen'
            if self.world.current_time-s['start']>120:status='timeout'
            if status:
                if status!='arrived':self.hold_target=p[:2].copy()
                self.event('navigation_end',robot_id='Spot',destination=s['destination'],status=status)
                done.append((s['job']['job_id'],status));self.active.pop('Spot');command=[0.,0.,0.]
        if 'Spot' not in self.active:
            delta=self.hold_target-p[:2]
            command=[float(np.clip(delta[0]*1.5,-.12,.12)),float(np.clip(delta[1]*1.5,-.12,.12)),float(np.clip(-yaw*1.5,-.4,.4))]
        self.last_command=command
        self.spot.forward(dt,self.torch.tensor(command,dtype=self.torch.float32))
        if self.ticks%8==0:
            for r in ['A','B']:
                s=self.active.get(r)
                if not s:continue
                b,c=self.arms[r],self.ctrl[r];status=None
                if self.world.current_time-s['start']>70:status='timeout'
                elif not c.is_done():
                    action=c.forward(picking_position=s['pick'],placing_position=s['place'],current_joint_positions=b.get_joint_positions(),end_effector_offset=np.array([0,0,0]))
                    b.get_articulation_controller().apply_action(action)
                    if s['failure']:b.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.array([.04,.04]),joint_indices=np.array([7,8])))
                elif s['home_ticks']<180:
                    if not s['home_ticks']:s['home_start']=b.get_joint_positions().copy()
                    s['home_ticks']+=1;u=s['home_ticks']/180;mix=u*u*u*(10+u*(-15+6*u))
                    q=s['home_start']*(1-mix)+self.home[r]*mix;q[-2:]=.04
                    b.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
                else:status='sequence_completed'
                if status:done.append((s['job']['job_id'],status));self.active.pop(r)
        return done

    def truth(self):
        p,yaw=self.pose()
        return {'payload_position':self.payload.get_world_pose()[0].tolist(),'carrier_position':p.tolist(),'carrier_yaw':yaw,
            'loaded':self.loaded(),'loaded_once':self.loaded_once,'departed_loaded':self.departed_loaded,'arrived_loaded':self.arrived_loaded,
            'physical_task_success':self.unloaded,'progress_fraction':sum([self.loaded_once,self.departed_loaded,self.arrived_loaded,self.unloaded])/4,
            'injected':self.injected,'robot_joints':{r:b.get_joint_positions().tolist() for r,b in self.arms.items()},
            'controller_phases':{r:int(c.get_current_event()) for r,c in self.ctrl.items()},'tray_body_path':self.body_path,
            'locomotion_command':getattr(self,'last_command',None),'locomotion_action':self.spot._previous_action.tolist(),
            'policy_decimation':self.spot._decimation,'policy_dt':self.spot._dt,
            'spot_dof_names':self.spot.robot.dof_names}

    def update_cameras(self,cameras):
        p,yaw=self.pose();forward=np.array([math.cos(yaw),math.sin(yaw),0.])
        cameras['Spot'].set_world_pose(position=p+.48*forward+np.array([0,0,.32]),
            orientation=euler_angles_to_quat(np.array([0.,.1,yaw])),camera_axes='world')
        cameras['cargo'].set_world_pose(position=p+np.array([0,0,1.3]))
