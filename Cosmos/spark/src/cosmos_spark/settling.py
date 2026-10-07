"""Separate post-terminal physics observation; preserves original task outcome/trajectory."""
import math
from pathlib import Path
import numpy as np
from .artifacts import append_jsonl,write_json


def observe(env,cfg,observation,last_action,output,seconds):
    import torch
    from robolab.core.utils.video_utils import VideoWriter
    output=Path(output);output.mkdir(exist_ok=False)
    dt=cfg.sim.dt*cfg.decimation;n=round(seconds/dt)
    if n<1 or not math.isfinite(seconds):raise ValueError('Positive finite observation duration required')
    # Hold the measured terminal arm pose and retain the terminal gripper command.
    # Direct physics stepping avoids RoboLab replacing frozen-env actions with all-zero joint targets.
    held=np.concatenate((observation['proprio_obs']['arm_joint_pos'][0].cpu().numpy(),np.asarray(last_action)[-1:]))
    action=torch.as_tensor(held,device=env.device).reshape(1,8)
    writers={};rows=[]
    success_cfg=cfg.terminations.success
    targets=success_cfg.params.get('object',[])
    targets=[targets] if isinstance(targets,str) else targets
    def record(obs,i):
        for group in ('image_obs','viewport_cam'):
            for name,tensor in obs.get(group,{}).items():
                frame=tensor[0].detach().cpu().numpy()[...,:3]
                if name not in writers:writers[name]=VideoWriter(str(output/(name+'.mp4')),1/dt)
                writers[name].write(frame)
        states={name:{'position':asset.data.root_pos_w[0].cpu().numpy(),
                      'linear_velocity':asset.data.root_lin_vel_w[0].cpu().numpy(),
                      'angular_velocity':asset.data.root_ang_vel_w[0].cpu().numpy()}
                for name,asset in env.scene.rigid_objects.items()}
        contact={}
        for name in dict.fromkeys([*targets,*getattr(cfg,'contact_object_list',[])]):
            contact[name]={}
            for finger in ('gripper','audit_right'):
                sensor=getattr(env.scene,'sensors',{}).get(finger+'__'+name)
                contact[name][finger]=float(sensor.data.force_matrix_w[0].abs().max().item()) if sensor is not None else None
        row={'finger_contact_max_abs_component_N':contact,
             'gripper_position':obs['proprio_obs']['gripper_pos'][0].detach().cpu().numpy() if 'gripper_pos' in obs.get('proprio_obs',{}) else None,
             'frame':i,'seconds_after_terminal':i*dt,'objects':states,
             'original_success_predicate':bool(success_cfg.func(env,**success_cfg.params)[0])}
        append_jsonl(output/'observations.jsonl',row);rows.append(row)
    try:
        record(observation,0)
        for i in range(1,n+1):
            env.action_manager.process_action(action)
            for _ in range(cfg.decimation):
                env._sim_step_counter+=1
                env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step(render=False)
                if env._sim_step_counter%cfg.sim.render_interval==0:env.sim.render()
                env.scene.update(dt=cfg.sim.dt)
            env.common_step_counter+=1
            obs=env.observation_manager.compute(update_history=True)
            record(obs,i)
    finally:
        for writer in writers.values():writer.release()
    tail=rows[-max(1,round(1/dt)):]
    objects=success_cfg.params.get('object',[])
    objects=[objects] if isinstance(objects,str) else objects
    speed={name:max(float(np.linalg.norm(r['objects'][name]['linear_velocity'])) for r in tail)
           for name in objects if name in tail[0]['objects']}
    measured=bool(targets) and all(all(value is not None for value in r['finger_contact_max_abs_component_N'][name].values()) for r in tail for name in targets)
    detached=all(all(value <= 0.1 for value in r['finger_contact_max_abs_component_N'][name].values()) for r in tail for name in targets) if measured else None
    result={'two_finger_detached_final_second':detached,
            'contact_audit':'Original left inner finger plus supplemental right inner finger, abs force component <=0.1N; not all robot surfaces',
            'seconds':n*dt,'frames':len(rows),'action':held,'policy_requests':0,
            'control':'hold measured terminal arm joints and retain final gripper command; physics continues',
            'original_task_outcome_unchanged':True,
            'predicate_true_all_frames':all(r['original_success_predicate'] for r in rows),
            'predicate_true_final_second':all(r['original_success_predicate'] for r in tail),
            'target_max_linear_speed_final_second_m_s':speed,
            'secondary_stability_check':bool(speed) and all(r['original_success_predicate'] for r in tail) and max(speed.values())<0.02,
            'secondary_criterion':'Original goal predicate holds every frame in last 1s and each target linear speed <0.02m/s; not full geometry containment or indefinite stability'}
    result['strict_secondary_check']=result['secondary_stability_check'] and detached if measured else None
    write_json(output/'summary.json',result)
    return result
