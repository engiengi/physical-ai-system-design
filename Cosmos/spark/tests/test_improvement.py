import ast,json
from pathlib import Path
import pytest
from cosmos_spark.media_session import start

def test_improved_terminal_analysis_rejects_midstream(tmp_path):
    p=tmp_path/'inputs';(p/'media').mkdir(parents=True)
    (p/'media/x.mp4').write_bytes(b'video')
    (p/'analysis.json').write_text(json.dumps({'cases':[{'id':'x','source':'media/x.mp4','trial_ended':False}]}))
    with pytest.raises(ValueError,match='ended-trial'):start(p,'analyze','test',tmp_path/'job','terminal-evidence-v3')
    assert not (tmp_path/'job').exists()

def test_concise_judgment_keeps_failure_and_unknown():
    tree=ast.parse(Path('scripts/thor/media_improvement.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='task_prompt')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'prompt','exec'),ns)
    s=ns['task_prompt']('put both hammers into the left bin',version='terminal-evidence-v3')
    assert 'both hammers' in s and 'ended trial' in s and 'failure' in s and 'unknown' in s

@pytest.mark.parametrize('right_force',[None,0.0,0.2])
@pytest.mark.parametrize('speed,goal,expected',[(0.001,True,True),(0.1,True,False),(0.001,False,False)])
def test_post_terminal_steps_real_physics_with_held_pose_and_separate_metrics(tmp_path,speed,goal,expected,right_force):
    from types import SimpleNamespace as N
    import torch
    from cosmos_spark.settling import observe
    counts={'physics':0,'action':[]}
    def physics(**kw):counts['physics']+=1
    manager=N(process_action=lambda a:counts['action'].append(a.clone()),apply_action=lambda:None)
    obs={'proprio_obs':{'arm_joint_pos':torch.tensor([[1.,2.,3.,4.,5.,6.,7.]]),'gripper_pos':torch.tensor([[0.5]])}}
    asset=N(data=N(root_pos_w=torch.zeros((1,3)),root_lin_vel_w=torch.tensor([[speed,0,0]]),root_ang_vel_w=torch.zeros((1,3))))
    env=N(device='cpu',action_manager=manager,_sim_step_counter=0,common_step_counter=0,
          sim=N(step=physics,render=lambda:None),scene=N(write_data_to_sim=lambda:None,update=lambda **kw:None,rigid_objects={'target':asset}),
          observation_manager=N(compute=lambda **kw:obs))
    if right_force is not None:
        env.scene.sensors={name+'__target':N(data=N(force_matrix_w=torch.tensor([[[[force,0.,0.]]]]))) for name,force in [('gripper',0.),('audit_right',right_force)]}
    cfg=N(sim=N(dt=1/120,render_interval=8),decimation=8,
          terminations=N(success=N(func=lambda env,**kw:[goal],params={'object':'target'})))
    r=observe(env,cfg,obs,[0,0,0,0,0,0,0,1],tmp_path/'settle',1.2)
    assert counts['physics']==144 and len(counts['action'])==18
    assert counts['action'][0].tolist()==[[1,2,3,4,5,6,7,1]]
    assert r['strict_secondary_check'] is (None if right_force is None else expected and right_force<=0.1)
    assert r['frames']==19 and r['secondary_stability_check'] is expected and r['policy_requests']==0

def test_singleton_format_diagnostic_preserves_verdict_without_scoring_as_strict_valid():
    from cosmos_spark.outcomes import inspect_response_shape
    raw='```json\n[{"status":"failure","stage":"holding","evidence":"object outside bin"}]\n```'
    d=inspect_response_shape(raw)
    assert d['compatible_prediction']['status']=='failure' and d['format_normalization']=='unwrap_singleton_list'
    assert d['strict_experiment_result_unchanged'] and not d['changes_model_judgment']

@pytest.mark.parametrize('raw',['No','[{"status":"success"},{"status":"failure"}]','{"status":"in_progress","stage":"moving","evidence":"arm moves"}'])
def test_format_diagnostic_does_not_invent_or_merge_outcomes(raw):
    from cosmos_spark.outcomes import inspect_response_shape
    with pytest.raises(ValueError):inspect_response_shape(raw)
