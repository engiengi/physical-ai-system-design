"""Isaac-only physical worlds for cooperative experiments.

Scene state here is privileged. Only public jobs and rendered observations leave
the simulator; object/station truth is recorded separately for evaluation.
"""
import math
import time
import numpy as np
from pxr import UsdLux
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.policy.examples.robots.spot import SpotFlatTerrainPolicy
from isaacsim.core.deprecation_manager import import_module
from omni.physx import get_physx_scene_query_interface
from coop_rules import ROBOTS, TRAYS, evaluate_arm
from scenario_rules import steering


def box(world, name, position, scale, color, dynamic=False):
    cls = DynamicCuboid if dynamic else FixedCuboid
    return world.scene.add(cls('/World/'+name, name=name, position=np.array(position, float),
        scale=np.array(scale, float), color=np.array(color), **({'mass':.05} if dynamic else {})))


class DualFranka:
    hz = 60
    look = [.45, 0, .0]
    overview_eye = [1.9, 2.5, 2.1]
    camera_eyes = {'overview':[.45,0,2.65], 'A':[.45,-.4,1.6], 'B':[.45,.4,1.6]}
    camera_targets = {'overview':[.45,0,0], 'A':[.45,-.3,0], 'B':[.45,.3,0]}

    def __init__(self, world, event):
        self.world, self.event = world, event
        world.scene.add_default_ground_plane(z_position=-.06)
        box(world, 'Worktop', [.4,0,-.025], [1.2,2.1,.05], [.58,.63,.68])
        self.bases = {'A': np.array([0.,-.45,0.]), 'B':np.array([0.,.45,0.])}
        self.robots = {r:world.scene.add(Franka('/World/Franka'+r, name='franka'+r, position=p)) for r,p in self.bases.items()}
        self.initial = {'red_1':[.36,-.37,.026], 'red_2':[.56,-.38,.026],
                        'blue_1':[.36,.37,.026], 'blue_2':[.56,.38,.026]}
        self.objects = {n:box(world,n,p,[.05]*3,[.85,.04,.02] if n.startswith('red') else [.03,.14,.85],True)
                        for n,p in self.initial.items()}
        for r,(x,y) in TRAYS.items():
            box(world,'Tray'+r,[x,y,.005],[.32,.22,.01],[.07,.5,.18])
            for i,(dx,dy,sx,sy) in enumerate([(-.165,0,.01,.23),(.165,0,.01,.23),(0,-.115,.34,.01),(0,.115,.34,.01)]):
                box(world,f'Tray{r}Wall{i}',[x+dx,y+dy,.025],[sx,sy,.05],[.04,.37,.12])
        box(world,'Exchange',[.45,0,.004],[.28,.24,.008],[.9,.65,.05])
        self.active = {}
        self.shared_owner = None
        self.condition = 'normal'
        self.injected = False

    def initialize(self):
        self.home = {r:bot.get_joint_positions().copy() for r,bot in self.robots.items()}
        for q in self.home.values(): q[-2:] = [.04,.04]
        self.ctrl = {r:PickPlaceController(name='coop_pick_'+r,gripper=b.gripper,robot_articulation=b,
                                         end_effector_initial_height=.25) for r,b in self.robots.items()}
        for bot in self.robots.values(): bot.gripper.set_default_state(bot.gripper.joint_opened_positions)

    def reset(self, condition):
        self.condition, self.injected, self.shared_owner = condition, False, None
        self.active.clear()
        for r,bot in self.robots.items():
            bot.set_joint_positions(self.home[r]); bot.set_joint_velocities(np.zeros(9))
            bot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=self.home[r]))
            self.ctrl[r].reset()
        for n,o in self.objects.items():
            o.set_world_pose(np.array(self.initial[n]),np.array([1.,0,0,0]))
            o.set_linear_velocity(np.zeros(3)); o.set_angular_velocity(np.zeros(3))

    def start(self, job, targets):
        r = job['robot_id']
        pick, place = targets
        for p in targets:
            if not .25 <= p[0] <= .7 or not -.88 <= p[1] <= .88:
                raise ValueError('Outside calibrated workspace')
            if r == 'A' and p[1] > .14 or r == 'B' and p[1] < -.14:
                raise ValueError('Other robot owns that side. Use the central exchange area.')
        shared = any(abs(p[1]) < .17 for p in targets)
        if shared and self.shared_owner:
            raise ValueError('Central exchange occupied by '+self.shared_owner+'; retry after its job ends')
        if shared: self.shared_owner = r
        self.ctrl[r].reset()
        failure = self.condition == 'exception' and not self.injected and abs(place[1]) < .14
        if failure:
            self.injected = True
            self.event('grasp_failure_injected', robot_id=r, job_id=job['job_id'], mechanism='Keep fingers open for one exchange-bound attempt; no object teleport')
        self.active[r] = {'job':job,'pick':np.array(pick),'place':np.array(place),'ticks':0,'home_ticks':0,
                          'failure':failure,'shared':shared,'start':self.world.current_time}

    def cancel(self, r):
        if r in self.active:
            bot = self.robots[r]
            bot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=bot.get_joint_positions()))
            self.active.pop(r)
            if self.shared_owner == r: self.shared_owner = None

    def tick(self, dt):
        done = []
        for r,s in list(self.active.items()):
            bot, ctrl = self.robots[r], self.ctrl[r]
            s['ticks'] += 1
            status = None
            if s['ticks'] > 2400: status = 'timeout'
            elif not ctrl.is_done():
                action = ctrl.forward(picking_position=s['pick'],placing_position=s['place'],
                                      current_joint_positions=bot.get_joint_positions(),end_effector_offset=np.array([0,.005,0]))
                bot.get_articulation_controller().apply_action(action)
                if s['failure']:
                    bot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.array([.04,.04]),joint_indices=np.array([7,8])))
            elif s['home_ticks'] < 180:
                if s['home_ticks'] == 0: s['home_start'] = bot.get_joint_positions().copy()
                s['home_ticks'] += 1
                u = s['home_ticks']/180
                mix = u*u*u*(10+u*(-15+6*u))
                target = s['home_start']*(1-mix)+self.home[r]*mix
                target[-2:] = [.04,.04]
                bot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=target))
            elif self.world.current_time-s.setdefault('settle_start',self.world.current_time) >= 1.:
                status = 'sequence_completed'
            if status:
                done.append((s['job']['job_id'],status)); self.active.pop(r)
                if self.shared_owner == r: self.shared_owner = None
        return done

    def truth(self):
        objects = {n:o.get_world_pose()[0].tolist() for n,o in self.objects.items()}
        return {'objects':objects, **evaluate_arm(objects), 'injected':self.injected,
                'robot_joints':{r:b.get_joint_positions().tolist() for r,b in self.robots.items()},
                'controller_phases':{r:int(self.ctrl[r].get_current_event()) for r in ROBOTS}}

    def update_cameras(self, cameras): pass
