"""Calibrated Cartesian drawer skill with physical contact, not an action policy.

Uses NVIDIA RMPFlow for joint targets. ER2 selects this whole skill; it does not
generate these waypoints. No drawer pose writes occur during opening.
"""
import numpy as np
from pxr import PhysxSchema
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.experimental.prims import Articulation, RigidPrim, XformPrim
from isaacsim.core.utils.rotations import quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import RmpFlow, interface_config_loader


class DrawerMotion:
    _dt = 1/120
    _decimation = 1
    _indices = list(range(8))
    default_pos = np.array([0., -.785, 0., -2.356, 0., 1.571, .785, .04, .04])

    def __init__(self, prim_path, cabinet, assets):
        self.classic = Franka(prim_path=prim_path, name="drawer_franka")
        self.robot = Articulation(paths=prim_path)
        self.cabinet = cabinet
        self.handle = XformPrim("/World/cabinet/drawer_handle_top")
        for link in ("panda_leftfinger", "panda_rightfinger"):
            PhysxSchema.PhysxContactReportAPI.Apply(get_prim_at_path(prim_path+"/"+link)).CreateThresholdAttr(0.)
        self.fingers = RigidPrim([prim_path+"/panda_leftfinger", prim_path+"/panda_rightfinger"],
                                contact_filter_paths=["/World/cabinet/drawer_handle_top"], max_contact_count=128)
        config = interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmp = RmpFlow(**config)
        self.rmp.set_ignore_state_updates(True)
        self.orientation = np.array([.5, .5, .5, .5])
        self.phase = "idle"
        self.elapsed = 0.
        self.result = None
        self.command_speed = 0.
        self.reset_quality()

    def reset_quality(self):
        self.peak_arm_speed = self.peak_bottom = self.peak_other_doors = self.peak_tcp_speed = 0.

    def initialize(self):
        self.classic.initialize()
        self.art = self.classic.get_articulation_controller()
        self.robot.set_dof_drive_types("force")
        self.robot.set_solver_iteration_counts(position_counts=[32], velocity_counts=[8])
        self.art.set_gains(kps=np.array([3000.]*7+[2000.]*2), kds=np.array([180.]*7+[100.]*2))
        self.robot.set_dof_max_efforts([87.]*4+[12.]*3+[40.]*2)
        self.robot.set_dof_max_velocities([.7]*7+[.04]*2)
        self.robot.set_default_state(dof_positions=self.default_pos, dof_velocities=np.zeros(9))
        self.top_index = self.cabinet.get_dof_indices("drawer_top_joint")
        self.bottom_index = self.cabinet.get_dof_indices("drawer_bottom_joint")
        self.post_reset()

    def post_reset(self):
        # Reset only before recording; never use joint-position writes to recover.
        self.robot.set_dof_positions(self.default_pos)
        self.robot.set_dof_velocities(np.zeros(9))
        self.robot.set_dof_position_targets(self.default_pos[:8], dof_indices=self._indices)
        self.robot.set_dof_velocity_targets(np.zeros(8), dof_indices=self._indices)
        self.q_command = self.default_pos[:7].copy()
        self.v_command = np.zeros(7)
        self.finger_target = .04
        self.rmp.reset()
        self.rmp.set_ignore_state_updates(True)
        self.rmp.set_robot_base_pose(np.zeros(3), np.array([1., 0, 0, 0]))
        self.target, self.initial_rotation = self.rmp.get_end_effector_pose(self.q_command)
        self.target = np.asarray(self.target).copy()
        self.phase, self.elapsed, self.result = "idle", 0., None
        self.grasp_frames = 0
        self.last_tcp = self.target.copy()
        self.tcp_speed = 0.
        self.command_speed = 0.
        self.reset_quality()

    def handle_center(self):
        p, q = self.handle.get_world_poses()
        return p.numpy()[0]+quat_to_rot_matrix(q.numpy()[0])@np.array([.305, 0, .01])

    def tcp(self):
        q = self.robot.get_dof_positions().numpy().ravel()[:7]
        return np.asarray(self.rmp.get_end_effector_pose(q)[0])

    def contacts(self, dt):
        forces = self.fingers.get_contact_force_matrix(dt=dt).numpy()
        return np.linalg.norm(forces[:, 0, :], axis=1)

    def begin(self):
        self.result = None
        self.grasp_frames = 0
        self.grasp_confirmed = False
        self.open_start = float(self.cabinet.get_dof_positions(dof_indices=self.top_index).numpy().ravel()[0])
        self.grasp_point = self.handle_center()
        self.start_orientation = rot_matrix_to_quat(self.rmp.get_end_effector_pose(self.robot.get_dof_positions().numpy().ravel()[:7])[1])
        self.pregrasp = self.grasp_point+np.array([-.13, 0, 0])
        self.pull_end = self.grasp_point+np.array([-.22, 0, 0])
        self.enter("ungrip", self.tcp(), 2.5)

    def enter(self, phase, target, duration):
        self.phase, self.elapsed, self.duration = phase, 0., duration
        self.start = self.target.copy()
        self.goal = np.array(target).copy()
        self.finger_start = self.finger_target
        self.stable = 0

    def stop(self, reason="stopped_by_user"):
        # RMPFlow holds the attained TCP with the SAME gains, no home sweep.
        self.target = self.tcp().copy()
        self.rmp.reset()
        self.rmp.set_ignore_state_updates(True)
        self.rmp.set_robot_base_pose(np.zeros(3), np.array([1., 0, 0, 0]))
        self.q_command = self.robot.get_dof_positions().numpy().ravel()[:7].copy()
        self.v_command = np.zeros(7)
        self.phase, self.result = "stopped", reason

    def tick(self, dt):
        q = self.robot.get_dof_positions().numpy().ravel()[:7]
        qd = self.robot.get_dof_velocities().numpy().ravel()[:7]
        tcp = self.tcp()
        self.tcp_speed = float(np.linalg.norm(tcp-self.last_tcp)/dt)
        self.last_tcp = tcp.copy()
        self.peak_arm_speed = max(self.peak_arm_speed, float(np.max(np.abs(qd))))
        self.peak_tcp_speed = max(self.peak_tcp_speed, self.tcp_speed)
        cabinet_q = dict(zip(self.cabinet.dof_names, self.cabinet.get_dof_positions().numpy().ravel()))
        self.peak_bottom = max(self.peak_bottom, abs(float(cabinet_q["drawer_bottom_joint"])))
        self.peak_other_doors = max(self.peak_other_doors, max((abs(float(v)) for k, v in cabinet_q.items() if k.startswith("door_")), default=0))
        moving = self.phase not in ("idle", "done", "stopped")
        if moving and np.max(np.abs(qd)) > 1.0:
            self.stop("excess_joint_speed")
            moving = False
        bottom = float(self.cabinet.get_dof_positions(dof_indices=self.bottom_index).numpy().ravel()[0])
        if moving and abs(bottom) > .01:
            self.stop("non_target_drawer_moved")
            moving = False
        if moving:
            self.elapsed += dt
            u = min(1., self.elapsed/self.duration)
            smooth = u*u*u*(10+u*(-15+6*u))
            self.target = self.start+(self.goal-self.start)*smooth
            desired_finger = 0.0 if self.phase in ("grasp", "pull") else .04
            self.finger_target += np.clip(desired_finger-self.finger_target, -.02*dt, .02*dt)
            if self.phase == "grasp":
                self.grasp_frames = self.grasp_frames+1 if np.min(self.contacts(dt)) > .2 else 0
                if self.grasp_frames >= round(.2/dt):
                    self.grasp_confirmed = True
            error = float(np.linalg.norm(tcp-self.goal))
            if self.elapsed >= self.duration and error < .015 and self.tcp_speed < .08:
                self.stable += 1
            else:
                self.stable = 0
            if self.stable >= round(.25/dt):
                if self.phase == "ungrip": self.enter("lift", np.array([min(self.tcp()[0], self.pregrasp[0]), self.grasp_point[1], self.grasp_point[2]+.10]), 3.)
                elif self.phase == "lift": self.enter("align", self.pregrasp, 3.)
                elif self.phase == "align": self.enter("approach", self.grasp_point, 3.)
                elif self.phase == "approach": self.enter("grasp", self.grasp_point, 2.5)
                elif self.phase == "grasp":
                    if self.grasp_confirmed: self.enter("pull", self.pull_end, 8.)
                    else: self.stop("grasp_unconfirmed")
                elif self.phase == "pull": self.enter("release", self.pull_end, 2.5)
                elif self.phase == "release": self.enter("retreat", self.pull_end+np.array([-.10, 0, .07]), 3.)
                elif self.phase == "retreat": self.enter("settle", self.goal, 2.)
                elif self.phase == "settle": self.phase, self.result = "done", "completed"
            if moving and self.elapsed > self.duration+8:
                self.stop("tracking_timeout")
        orient = None
        if self.phase in ("ungrip", "lift"):
            orient = self.start_orientation
        elif self.phase != "idle":
            orient = self.orientation
            if self.phase == "align":
                initial = self.start_orientation.copy()
                if np.dot(initial, orient) < 0: initial = -initial
                mix = min(1., self.elapsed/3.)
                mix = mix*mix*(3-2*mix)
                orient = (1-mix)*initial+mix*orient
                orient /= np.linalg.norm(orient)
        self.rmp.set_end_effector_target(self.target, orient)
        desired, velocity = self.rmp.compute_joint_targets(q, qd, np.empty(0), np.empty(0), dt)
        delta = desired-self.q_command
        scale = min(1., .5*dt/max(1e-9, float(np.max(np.abs(delta)))))
        self.v_command = delta*scale/dt
        self.q_command += delta*scale
        self.command_speed = float(np.max(np.abs(self.v_command)))
        self.art.apply_action(ArticulationAction(joint_positions=np.r_[self.q_command, self.finger_target, self.finger_target],
                                                joint_velocities=np.r_[self.v_command, 0., 0.]))

    def metrics(self, dt):
        qd = self.robot.get_dof_velocities().numpy().ravel()
        return {"drawer_bottom_open_m": float(self.cabinet.get_dof_positions(dof_indices=self.bottom_index).numpy().ravel()[0]),
                "cabinet_joints": dict(zip(self.cabinet.dof_names, self.cabinet.get_dof_positions().numpy().ravel().tolist())),
                "arm_joint_positions": self.robot.get_dof_positions().numpy().ravel()[:7].tolist(),
                "arm_joint_speed_max_rad_s": float(np.max(np.abs(qd[:7]))),
                "command_speed_max_rad_s": self.command_speed,
                "arm_joint_target": self.q_command.tolist(),
                "physics_peak_arm_speed_rad_s": self.peak_arm_speed,
                "physics_peak_bottom_displacement_m": self.peak_bottom,
                "physics_peak_other_doors_rad": self.peak_other_doors,
                "physics_peak_tcp_speed_m_s": self.peak_tcp_speed,
                "tcp_position": self.tcp().tolist(), "tcp_speed_m_s": self.tcp_speed,
                "finger_positions": self.robot.get_dof_positions().numpy().ravel()[7:].tolist(),
                "finger_handle_contact_n": self.contacts(dt).tolist(),
                "controller_phase": self.phase, "grasp_confirmed": getattr(self, "grasp_confirmed", False)}
