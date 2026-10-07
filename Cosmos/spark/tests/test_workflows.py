import json
from pathlib import Path
import numpy as np
import pytest
from cosmos_spark.client import PolicyError, RecordedCosmosClient
from cosmos_spark.evaluate import item_completed, summarize_batch, build_command, digest
from cosmos_spark.remote import remote_path, fetch_file
from test_client import policy_server, observation


def test_policy_seed_roundtrip_and_refuse_ignored_seed(tmp_path):
    actions=np.zeros((1,8),dtype=np.float32)
    with policy_server(lambda r:{'action':actions,'policy_seed':r['policy_seed']}) as (uri,requests):
        c=RecordedCosmosClient(uri,tmp_path/'ok','run','ep',policy_seed=900)
        try:
            c.infer(observation(),'task');c.infer(observation(),'task')
        finally:c.close()
        assert [r['policy_seed'] for r in requests]==[900,901]
    with policy_server(lambda r:{'action':actions}) as (uri,_):
        c=RecordedCosmosClient(uri,tmp_path/'bad','run','ep',policy_seed=900)
        try:
            with pytest.raises(PolicyError,match='acknowledge'):c.infer(observation(),'task')
            assert not c._chunks
        finally:c.close()
    row=json.loads((tmp_path/'bad/requests.jsonl').read_text())
    assert row['status']=='execution_error' and row['policy_seed_requested']==900
    assert (tmp_path/'bad/request_00000_actions.npy').exists()


def test_resume_never_replaces_task_failure_and_reports_infra_attempts():
    plan={'items':[{'id':'a'},{'id':'b'},{'id':'c'}],'config':{'conditions':[{'name':'default'}]}}
    def attempt(item,status):return {'item_id':item,'result':{'condition':'default','status':status}}
    rows=[attempt('a','task_failure'),attempt('a','success'),attempt('b','execution_error'),attempt('b','success')]
    report=summarize_batch(plan,rows)
    assert item_completed(rows,'a')
    assert not item_completed([attempt('c','incomplete')],'c')
    assert report['primary']['success_rate']==.5
    assert report['all_attempts']['execution_errors']==1
    assert report['retry_attempts']==2 and report['not_started']==1


def test_gui_command_and_seed_are_fixed_by_plan():
    p={'gui':True,'uri':'ws://localhost:8000','config':{'task':'BananaInBowlTask','execute_horizon':16}}
    item={'environment_seed':4,'condition':'base','pose_range_m':0,'policy_seed':100}
    c=build_command(p,item,'unique')
    assert 'cosmos_spark.session' in c and '--headless' not in c
    assert c[c.index('--policy-seed')+1]=='100'
    original=digest(p);p['config']['execute_horizon']=32
    assert digest(p)!=original


@pytest.mark.parametrize('s',['/etc/passwd','/home/geunpil/cosmos_ws/../secret','/home/geunpil/cosmos_ws/$(whoami)','/home/geunpil/cosmos_ws/a;true'])
def test_remote_artifact_paths_are_bounded(s):
    with pytest.raises(ValueError):remote_path(s)


def test_review_http_and_export_keep_ground_truth_out_of_inputs(tmp_path):
    import hashlib,threading,urllib.request,urllib.parse
    from http.server import ThreadingHTTPServer
    from cosmos_spark.review import handler_for,export,latest_reviews
    video=tmp_path/'fixture.mp4';video.write_bytes(b'test video bytes')
    reference=tmp_path/'reference.png';reference.write_bytes(b'fixture image')
    case={'id':'fixture','group_id':'fixture','video':str(video),'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
          'source_type':'synthetic','task':'move object','variant':'generated',
          'reference_image':str(reference),'reference_image_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
          'intent_ko':'조명 <변화>','preserve_ko':['물체 유지']}
    dataset=tmp_path/'dataset.json';dataset.write_text(json.dumps({'cases':[case]}))
    srv=ThreadingHTTPServer(('127.0.0.1',0),handler_for(dataset,'test-token'))
    thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{srv.server_port}'
    try:
        page=urllib.request.urlopen(url).read().decode()
        assert '검토 대기' in page and '조명 &lt;변화&gt;' in page and '입력 원본 이미지' in page
        with urllib.request.urlopen(url+'/reference?id=fixture') as r:
            assert r.headers['Content-Type']=='image/png' and r.read()==b'fixture image'
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(url+'/reference?id=missing')
        assert error.value.code==404
        reference.write_bytes(b'changed image')
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(url+'/reference?id=fixture')
        assert error.value.code==409
        request=urllib.request.Request(url+'/media?id=fixture',headers={'Range':'bytes=0-3'})
        with urllib.request.urlopen(request) as r:assert r.status==206 and r.read()==b'test'
        form={'token':'test-token','id':'fixture','reviewer':'TEST FIXTURE ONLY','status':'failure','failure_stage':'',
              'evidence':'test evidence','requested_event_present':'true','object_identity_consistent':'true','motion_plausible':'true','accepted':'true'}
        with urllib.request.urlopen(urllib.request.Request(url+'/review',data=urllib.parse.urlencode(form).encode())) as r:assert r.status==200
        assert latest_reviews(tmp_path)['fixture']['reviewer_kind']=='human'
        assert latest_reviews(tmp_path)['fixture']['failure_stage']=='unspecified'
        assert latest_reviews(tmp_path)['fixture']['failure_stage_known'] is False
        result=export(dataset,tmp_path/'export')
        assert result['selected']==1
        inputs=(tmp_path/'export/analysis.json').read_text()
        assert 'label' not in inputs and 'evidence' not in inputs and 'TEST FIXTURE' not in inputs
        assert json.loads((tmp_path/'export/references.json').read_text())['cases'][0]['label']=='failure'
        video.write_bytes(b'changed')
        with pytest.raises(ValueError,match='changed'):export(dataset,tmp_path/'bad-export')
    finally:srv.shutdown();thread.join();srv.server_close()


def test_model_review_does_not_pass_human_export_gate(tmp_path):
    from cosmos_spark.review import export,validate_review
    video=tmp_path/'v.mp4';video.write_bytes(b'test')
    case={'id':'fixture','source_type':'synthetic','video':str(video),'sha256':'bad'}
    dataset=tmp_path/'dataset.json';dataset.write_text(json.dumps({'cases':[case]}))
    (tmp_path/'review_events.jsonl').write_text(json.dumps({'id':'fixture','reviewer_kind':'model_assisted','accepted':True})+'\n')
    assert export(dataset,tmp_path/'export')=={'selected':0,'pending':1,'rejected':0}
    with pytest.raises(ValueError,match='pass all criteria'):
        validate_review(case,{'reviewer_kind':'human','reviewer':'TEST','status':'success','evidence':'test','accepted':True,
                             'requested_event_present':False,'object_identity_consistent':True,'motion_plausible':True})


def test_delegated_reviews_keep_provenance_and_human_priority(tmp_path):
    from cosmos_spark.review import latest_reviews,export
    delegated={'id':'fixture','reviewer_kind':'ai_delegated','status':'in_progress','accepted':False}
    (tmp_path/'delegated_review_events.jsonl').write_text(json.dumps(delegated)+'\n')
    assert latest_reviews(tmp_path)['fixture']['reviewer_kind']=='ai_delegated'
    dataset=tmp_path/'dataset.json';dataset.write_text(json.dumps({'cases':[{'id':'fixture'}]}))
    assert export(dataset,tmp_path/'export')=={'selected':0,'pending':1,'rejected':0}
    human={'id':'fixture','reviewer_kind':'human','status':'success'}
    (tmp_path/'review_events.jsonl').write_text(json.dumps(human)+'\n')
    assert latest_reviews(tmp_path)['fixture']==human


def test_analysis_errors_never_disappear_from_denominators(tmp_path):
    from cosmos_spark.review import assess
    refs=tmp_path/'refs.json';summary=tmp_path/'summary.json'
    refs.write_text(json.dumps({'cases':[{'id':i,'label':'failure'} for i in ['a','b','c']]}))
    summary.write_text(json.dumps({'results':[{'id':'a','status':'success','prediction':{'status':'success'}},
                                            {'id':'b','status':'failed','error':'invalid JSON'}]}))
    r=assess(refs,summary,tmp_path/'out.json')
    assert r['valid_failure_denominator']==1 and r['failure_reference_count']==3
    assert r['missing']==['c'] and len(r['errors'])==1 and r['false_success_numerator']==1


def test_prediction_saved_before_actions_and_prefix_checked(tmp_path):
    import hashlib,shutil,time,subprocess
    from cosmos_spark.prediction import PredictionReceiver
    source=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=640x540:r=15',
                    '-frames:v','3','-c:v','libx264','-threads','1',str(source)],check=True)
    episode=tmp_path/'ep';episode.mkdir()
    def local_fetch(remote,target,digest):
        shutil.copyfile(source,target)
        actual=hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual==digest
        return actual
    receiver=PredictionReceiver(episode,2,15,fetcher=local_fetch)
    actions=np.zeros((2,8),np.float32);actions[1,-1]=.8
    request={'session_id':'test'}
    response={'session_id':'test','run_id':'TEST_ONLY','request_id':0,
              'prediction':{'path':'/home/geunpil/cosmos_ws/test/prediction.mp4','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frames':3,'fps':15}}
    receipt=receiver(response,'request_00000',actions,request)
    np.save(episode/'request_00000_actions.npy',actions)
    shutil.copyfile(source,episode/'policy_view.mp4')
    applied=actions.copy();applied[:,-1]=applied[:,-1]>.5
    rows=[{'request_id':'request_00000','chunk_index':i,'action':row.tolist(),'action_started_at_ns':time.time_ns(),'wall_time_ns':time.time_ns()} for i,row in enumerate(applied)]
    (episode/'applied_actions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=receiver.finalize()
    assert result['prediction_saved_before_action'] and result['compared_frames']==3
    rows[0]['action_started_at_ns']=receipt['saved_at_ns']-1
    (episode/'applied_actions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='BEFORE'):receiver.finalize()
    with pytest.raises(ValueError,match='exactly one'):receiver(response,'request_00001',actions,request)


def test_missing_prediction_withholds_action(tmp_path):
    from cosmos_spark.prediction import PredictionReceiver
    with policy_server(lambda r:{'action':np.zeros((2,8),np.float32),'session_id':r['session_id']}) as (uri,_):
        c=RecordedCosmosClient(uri,tmp_path,'run','ep',execute_horizon=2,prediction_receiver=PredictionReceiver(tmp_path,2,15))
        try:
            with pytest.raises(PolicyError,match='metadata unavailable'):c.infer(observation(),'test')
            assert not c._chunks
        finally:c.close()


def test_handoff_rejects_ground_truth_and_path_escape(tmp_path):
    from cosmos_spark.handoff import prepare_inputs
    (tmp_path/'media').mkdir();(tmp_path/'media/v.mp4').write_bytes(b'fixture')
    case={'id':'a','source':'media/v.mp4','task':'test'}
    f=tmp_path/'analysis.json';f.write_text(json.dumps({'cases':[case]}))
    assert len(prepare_inputs(tmp_path))==2
    case['label']='failure';f.write_text(json.dumps({'cases':[case]}))
    with pytest.raises(ValueError,match='Ground truth'):prepare_inputs(tmp_path)
    del case['label'];case['source']='../secret.mp4';f.write_text(json.dumps({'cases':[case]}))
    with pytest.raises(ValueError,match='portable'):prepare_inputs(tmp_path)


def test_interrupted_attempt_is_saved_and_not_counted_as_failure(tmp_path,monkeypatch):
    from cosmos_spark import evaluate
    monkeypatch.setattr(evaluate,'ROOT',tmp_path)
    sent=[];monkeypatch.setattr(evaluate.os,'killpg',lambda pid,sig:sent.append((pid,sig)))
    class Interrupted:
        pid=123
        calls=0
        def wait(self,timeout=None):
            self.calls+=1
            if self.calls==1:raise KeyboardInterrupt()
            return 130
        def poll(self):return None
    monkeypatch.setattr(evaluate.subprocess,'Popen',lambda *a,**k:Interrupted())
    plan={'batch_id':'test','record_thor':False,'gui':False,'uri':'ws://localhost:8000','config':{'task':'BananaInBowlTask','execute_horizon':16}}
    item={'id':'c0','condition':'base','environment_seed':0,'pose_range_m':0,'policy_seed':12}
    row=evaluate.run_item(plan,item,tmp_path,0)
    assert row['result']['status']=='incomplete' and sent
    assert not evaluate.item_completed([row],'c0')
    assert json.loads((tmp_path/'attempts/00000.json').read_text())['finished_at']


def test_thor_v2_artifacts_verify_action_and_identity(tmp_path):
    import hashlib,shutil,subprocess
    from cosmos_spark.prediction import PredictionReceiver
    video=tmp_path/'video.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=640x540:r=15','-frames:v','3','-c:v','libx264','-threads','1',str(video)],check=True)
    actions=np.zeros((2,8),np.float32);action=tmp_path/'action.npy';np.save(action,actions)
    def info(p):return {'path':'/home/geunpil/cosmos_ws/'+p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    manifest={'run_id':'v2','request_id':4,'session_id':'test','policy_seed':7,
              'prediction':{**info(video),'frames':3,'fps':15},'action':{**info(action),'shape':[2,8]}}
    response={**{k:manifest[k] for k in ('run_id','request_id','session_id','policy_seed')},'artifacts':manifest}
    def fetch(remote,target,digest):
        source=tmp_path/Path(remote).name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        shutil.copyfile(source,target);return digest
    ep=tmp_path/'ep';ep.mkdir()
    assert PredictionReceiver(ep,2,15,fetcher=fetch)(response,'request_0',actions,{'session_id':'test'})['probe']['frames']==3
    assert (ep/'request_0_thor_artifacts.json').exists()
    manifest['policy_seed']=8
    with pytest.raises(ValueError,match='identity mismatch'):
        PredictionReceiver(tmp_path/'bad',2,15,fetcher=fetch)(response,'r',actions,{'session_id':'test'})
    manifest['policy_seed']=7
    bad=tmp_path/'bad_action';bad.mkdir()
    with pytest.raises(AssertionError):
        PredictionReceiver(bad,2,15,fetcher=fetch)(response,'r',actions+1,{'session_id':'test'})


def test_authorized_delegated_export_is_separate_from_human_truth(tmp_path):
    import hashlib
    from cosmos_spark.review import export
    video=tmp_path/'video.mp4';video.write_bytes(b'fixture')
    digest=hashlib.sha256(video.read_bytes()).hexdigest()
    case={'id':'accepted','video':str(video),'sha256':digest,'source_type':'synthetic','task':'place objects','group_id':'scene','variant':'dim'}
    rejected={**case,'id':'rejected'}
    dataset=tmp_path/'dataset.json';dataset.write_text(json.dumps({'cases':[case,rejected]}))
    decision={'id':'accepted','reviewer':'Codex','reviewer_kind':'ai_delegated','status':'unknown','evidence':'scene visible','video_sha256':digest,'accepted':True,'requested_event_present':True,'object_identity_consistent':True,'motion_plausible':True}
    denied={**decision,'id':'rejected','accepted':False,'object_identity_consistent':False}
    (tmp_path/'delegated_review_events.jsonl').write_text(json.dumps(decision)+'\n'+json.dumps(denied)+'\n')
    assert export(dataset,tmp_path/'human_only')['selected']==0
    result=export(dataset,tmp_path/'delegated',delegated_authorization='User explicitly delegated review')
    assert result=={'selected':1,'pending':0,'rejected':1}
    refs=json.loads((tmp_path/'delegated/references.json').read_text())
    inputs=json.loads((tmp_path/'delegated/analysis.json').read_text())
    assert refs['cases'][0]['label_source']=='ai_delegated'
    assert 'label' not in inputs['cases'][0] and 'evidence' not in inputs['cases'][0]
