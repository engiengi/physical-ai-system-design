import json
from pathlib import Path
import pytest
from cosmos_spark.scenarios import prepare_analysis,group_results
from cosmos_spark.media_session import input_files
from cosmos_spark.corpus import build

def test_new_and_banana_analysis_preserve_tasks_without_labels(tmp_path):
    video=tmp_path/'v.mp4';video.write_bytes(b'fixture')
    cases=[{'id':str(i),'scenario_id':s,'track':t,'task':task,'source':str(video),
            'source_type':'simulation','group_id':s,'variant':'original','label':'success',
            'source_provenance':'Test fixture','evaluation_criteria':['Visible task completion']}
           for i,s,t,task in [(0,'tools','standalone','Sort the hammers'),(1,'banana','banana_link','Move banana')]]
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps({'cases':cases}))
    inputs=prepare_analysis(catalog,tmp_path/'out')
    assert [c['task'] for c in inputs]==['Sort the hammers','Move banana']
    assert [c['experiment_track'] for c in inputs]==['new_standalone','banana_link']
    assert all(c['source_type']=='simulation' for c in inputs)
    assert all('label' not in c for c in inputs)
    summary=tmp_path/'summary.json';summary.write_text(json.dumps({'results':[{'id':'0','status':'success','prediction':{'status':'in_progress'}},{'id':'1','status':'failed','error_kind':'format_error'}]}))
    grouped=group_results(tmp_path/'out/analysis.json',summary)
    assert grouped['groups']['standalone/tools']['valid']==1
    assert grouped['groups']['banana_link/banana']['errors']==1
    assert all(g['human_accuracy'] is None for g in grouped['groups'].values())
    cases[0]['source_type']='synthetic';catalog.write_text(json.dumps({'cases':cases}))
    with pytest.raises(ValueError,match='Unreviewed generated'):prepare_analysis(catalog,tmp_path/'blocked')
    assert not (tmp_path/'blocked').exists()

def test_media_upload_excludes_refs_and_rejects_unsafe_input(tmp_path):
    (tmp_path/'media').mkdir();(tmp_path/'media/input.png').write_bytes(b'image fixture')
    (tmp_path/'references.json').write_text('secret labels')
    c={'id':'new','source':'media/input.png','mode':'i2v','task':'Sort tools','prompt':'Room dims'}
    p=tmp_path/'generation.json';p.write_text(json.dumps({'cases':[c]}))
    files=input_files(tmp_path,'generate')
    assert {f['path'] for f in files}=={'generation.json','media/input.png'}
    c['label']='failure';p.write_text(json.dumps({'cases':[c]}))
    with pytest.raises(ValueError,match='Reference labels'):input_files(tmp_path,'generate')
    del c['label'];c['source']='media/../../private';p.write_text(json.dumps({'cases':[c]}))
    with pytest.raises(ValueError,match='portable'):input_files(tmp_path,'generate')

def test_synthetic_corpus_never_silently_assigns_banana(tmp_path):
    run=tmp_path/'generated';(run/'tools/native').mkdir(parents=True)
    (run/'tools/native/vision.mp4').write_bytes(b'fixture video')
    (run/'summary.json').write_text(json.dumps({'results':[{'id':'tools','status':'success'}]}))
    with pytest.raises(ValueError,match='never assume banana'):build([], [run],tmp_path/'bad')
    (run/'manifest.json').write_text(json.dumps({'cases':[{'id':'tools','task':'Sort the hammers','track':'standalone','scenario_id':'tools'}]}))
    result=build([], [run],tmp_path/'ok')
    assert result['cases'][0]['task']=='Sort the hammers'
    assert result['cases'][0]['track']=='standalone'

def test_policy_scenario_identity_reaches_thor(tmp_path):
    import numpy as np
    from cosmos_spark.client import RecordedCosmosClient
    from test_client import policy_server,observation
    context={'scenario_id':'cube_stack','experiment_track':'new_standalone','source_type':'simulation'}
    with policy_server(lambda r:{'action':np.zeros((1,8),np.float32),'policy_seed':r['policy_seed'],'client_context':context}) as (uri,requests):
        client=RecordedCosmosClient(uri,tmp_path,'run','ep',policy_seed=42,scenario_context=context)
        try:client.infer(observation(),'Stack the cubes')
        finally:client.close()
        assert {k:requests[0][k] for k in context}==context
        assert requests[0]['prompt']=='Stack the cubes'
    with pytest.raises(ValueError,match='scenario context'):
        RecordedCosmosClient('ws://unused:1',tmp_path/'invalid','run','ep',scenario_context={**context,'prompt':'override'})

def test_human_accuracy_is_not_pooled_across_scenarios(tmp_path):
    from cosmos_spark.review import assess
    refs=tmp_path/'refs.json';summary=tmp_path/'summary.json'
    refs.write_text(json.dumps({'cases':[{'id':'a','label':'failure','scenario_id':'tools','track':'standalone','source_type':'simulation'},
        {'id':'b','label':'success','scenario_id':'banana','track':'banana_link','source_type':'simulation'}]}))
    summary.write_text(json.dumps({'results':[{'id':'a','status':'success','prediction':{'status':'success'}},
        {'id':'b','status':'success','prediction':{'status':'success'}}]}))
    report=assess(refs,summary,tmp_path/'report.json')
    assert report['accuracy_valid'] is None
    assert {g['scenario_id']:g['accuracy_valid'] for g in report['groups']}=={'tools':0,'banana':1}
