"""Scenario-aware exploratory analysis inputs; human reference labels remain separate."""
import argparse,hashlib,json,re,shutil
from pathlib import Path
from .artifacts import write_json

TRACKS={'standalone','banana_link'}
def identity(case):
    for key in ('id','scenario_id'):
        if not isinstance(case.get(key),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',case[key]):
            raise ValueError('Safe '+key+' required')
    if case.get('track') not in TRACKS:raise ValueError('Explicit standalone or banana_link track required')
    if not isinstance(case.get('task'),str) or not case['task'].strip():raise ValueError('Scenario task required')
    return {**{k:case[k] for k in ('scenario_id','track','task')},
            'experiment_track':'new_standalone' if case['track']=='standalone' else 'banana_link'}

def prepare_analysis(catalog,output):
    data=json.loads(Path(catalog).read_text());cases=data['cases'];ids=set();inputs=[];review=[]
    for case in cases:
        identity(case)
        if case['id'] in ids:raise ValueError('Duplicate case id')
        ids.add(case['id'])
        if case.get('source_type') not in ('simulation','real'):
            raise ValueError('Unreviewed generated video cannot enter analysis; use human review export')
        if not Path(case['source']).is_file():raise FileNotFoundError(case['source'])
        if not case.get('source_provenance') or not case.get('evaluation_criteria'):
            raise ValueError('Source provenance and observable evaluation criteria required')
    if not cases:raise ValueError('Empty catalog')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'media').mkdir()
    for case in cases:
        source=Path(case['source']);target=output/'media'/(case['id']+'.mp4');shutil.copyfile(source,target)
        sha=hashlib.sha256(target.read_bytes()).hexdigest()
        inputs.append({'id':case['id'],'source':str(target.relative_to(output)),**identity(case),
                       'pair_id':case['group_id'],'variant':case['variant'],'fps':case.get('fps',4),
                       'condition':case['scenario_id']+'_'+case['variant'],
                       **{k:case[k] for k in ('source_type','source_provenance','evaluation_criteria')},
                       **({'view_layout':case['view_layout']} if case.get('view_layout') else {})})
        review.append({**case,'video':str(target.resolve()),'sha256':sha,'human_status':None,
                       'note':case.get('note','')+' Exploratory model output; human reference pending.'})
    write_json(output/'analysis.json',{'scope':'exploratory; no human accuracy claims','cases':inputs})
    write_json(output/'dataset.json',{'schema_version':2,'cases':review,'human_review':None})
    write_json(output/'preparation.json',{'source_catalog':str(Path(catalog).resolve()),'scope':'exploratory',
               'cases':len(cases),'human_ground_truth_supplied':False,'synthetic_inputs_allowed':False})
    return inputs

def group_results(input_manifest,summary):
    cases={c['id']:c for c in json.loads(Path(input_manifest).read_text())['cases']}
    results=json.loads(Path(summary).read_text())['results'];groups={};seen=set()
    for r in results:
        if r['id'] in seen or r['id'] not in cases:raise ValueError('Duplicate or unexpected result')
        seen.add(r['id']);c=cases[r['id']];identity(c)
        key=c['track']+'/'+c['scenario_id']
        g=groups.setdefault(key,{'track':c['track'],'scenario_id':c['scenario_id'],'cases':[],
            'valid':0,'errors':0,'human_accuracy':None})
        valid=r.get('status')=='success' and r.get('prediction',{}).get('status') in {'success','failure','in_progress','unknown'}
        g['valid' if valid else 'errors']+=1
        g['cases'].append({'id':r['id'],'group_id':c.get('pair_id'),'variant':c.get('variant'),
                           'prediction':r.get('prediction'),'error_kind':r.get('error_kind'),
                           'error':r.get('error'),'seconds':r.get('seconds')})
    return {'groups':groups,'missing':sorted(set(cases)-seen),'human_review':None,
            'note':'Model output validity is not task success or human-ground-truth accuracy. Paired variants are not independent episodes.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('catalog',type=Path);a.add_argument('--output',type=Path,required=True)
    g=sub.add_parser('summarize');g.add_argument('--manifest',type=Path,required=True);g.add_argument('--summary',type=Path,required=True);g.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':print(json.dumps({'cases':len(prepare_analysis(a.catalog,a.output))}))
    else:
        if a.output.exists():raise FileExistsError(a.output)
        write_json(a.output,group_results(a.manifest,a.summary));print(a.output)
if __name__=='__main__':main()
