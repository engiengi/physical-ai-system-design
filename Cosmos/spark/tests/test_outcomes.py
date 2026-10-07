import ast,json,subprocess,sys
from pathlib import Path
import pytest
from cosmos_spark.outcomes import completed_episode,prepare,compare
from cosmos_spark.media_session import start

def episode(tmp_path,*,status='success',cap=0,mode='policy',exit_code=0):
    root=tmp_path/'trial';ep=root/'episode_0000';ep.mkdir(parents=True)
    (root/'run.json').write_text(json.dumps({'arguments':{'mode':mode,'max_steps':cap}}))
    (root/'run_status.json').write_text(json.dumps({'exit_code':exit_code}))
    (ep/'episode.json').write_text(json.dumps({'run_id':'trial','episode_id':'episode_0000','instruction':'Put both hammers into the left bin','mode':mode,'status':status,'robolab_result':{'success':status=='success'}}))
    return ep

@pytest.mark.parametrize('settings',[{'cap':96},{'status':'incomplete'},{'mode':'predict'},{'exit_code':1},{'status':'execution_error'}])
def test_unfinished_and_invalid_episodes_never_become_outcomes(tmp_path,settings):
    ep=episode(tmp_path,**settings)
    with pytest.raises(ValueError):prepare([{'id':'case','episode':ep}],tmp_path/'package')
    assert not (tmp_path/'package').exists()

@pytest.mark.parametrize('status',['success','task_failure'])
def test_natural_endings_are_eligible(tmp_path,status):
    assert completed_episode(episode(tmp_path,status=status))['status']==status

def test_unknown_and_progress_are_not_counted_as_correct():
    refs={'cases':[{'id':n,'scenario_id':'tools','label':'failure','label_source':'simulator'} for n in 'abcd']}
    results={'results':[{'id':'a','status':'success','prediction':{'status':'unknown'}},
                        {'id':'b','status':'success','prediction':{'status':'in_progress'}},
                        {'id':'c','status':'failed','error_kind':'format_error'}]}
    report=compare(refs,results);a,b,c=report['rows']
    assert a['abstained'] and not a['matches_reference']
    assert not b['valid_terminal_answer'] and b['error_kind']=='non_terminal_answer'
    assert c['error_kind']=='format_error' and report['missing']==['d']
    assert report['pooled_accuracy'] is None

def test_terminal_prompt_says_ended_goal_and_retains_uncertainty():
    tree=ast.parse(Path('scripts/thor/media_terminal_outcome.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='task_prompt')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'prompt','exec'),namespace)
    prompt=namespace['task_prompt']('Put both hammers into left bin',version='terminal-outcome-v1')
    assert all(s in prompt for s in ['ENDED','complete goal','unknown','Do not answer in_progress','repeated'])

@pytest.mark.parametrize('version',['terminal-outcome-v1','terminal-outcome-raw-v2'])
def test_transfer_refuses_midstream_for_terminal_prompt(tmp_path,version):
    package=tmp_path/'package';(package/'media').mkdir(parents=True)
    (package/'media/a.mp4').write_bytes(b'video')
    (package/'analysis.json').write_text(json.dumps({'cases':[{'id':'a','source':'media/a.mp4','evaluation_phase':'midstream_diagnostic','trial_ended':False}]}))
    with pytest.raises(ValueError,match='ended-trial'):start(package,'analyze','trial',tmp_path/'job',version)
    assert not (tmp_path/'job').exists()

def test_prepare_keeps_references_off_model_input_and_marks_hold(tmp_path):
    ep=episode(tmp_path)
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=blue:s=320x240:r=15:d=1','-c:v','libx264','-threads','1',str(ep/'egocentric_mirrored_camera.mp4')],check=True)
    output=tmp_path/'package'
    cases=prepare([{'id':'case','episode':ep,'scenario_id':'tool_sorting','experiment_track':'new_standalone','criteria':['Both hammers inside left bin, gripper detached.']}],output)
    assert cases[0]['trial_ended'] is True and 'label' not in cases[0]
    assert 'success' not in cases[0]['source_provenance']
    assert json.loads((output/'references.json').read_text())['cases'][0]['label']=='success'
    assert json.loads((output/'preparation.json').read_text())['cases'][0]['final_hold_seconds']==1
