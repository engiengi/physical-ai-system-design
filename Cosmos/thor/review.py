"""Summarize explicit reviews of generated scenarios without auto-acceptance."""
import argparse
import json
from pathlib import Path
from common import write_json
from artifacts import file_info
from evaluation import STATUS
from scenarios import context, group_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reviews', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analysis-manifest', type=Path, help='Optional manifest containing accepted videos only')
    p.add_argument('--task', help='Task to evaluate on accepted videos')
    a=p.parse_args()
    if a.analysis_manifest and a.analysis_manifest.exists():
        p.error('A new analysis manifest path is required')
    if a.output.exists(): raise FileExistsError(a.output)
    generated=json.loads((a.run/'summary.json').read_text())['results']
    reviews=json.loads(a.reviews.read_text())['cases']
    ids=[r['id'] for r in reviews]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate review id')
    generated_by_id={r['id']:r for r in generated}
    known={r['id'] for r in generated if r['status']=='success'}
    if not set(ids)<=known: raise ValueError('Review references unknown or failed generation')
    accepted=[]
    rejected=[]
    for r in reviews:
        if r.get('accepted') is None: continue
        if type(r['accepted']) is not bool or not r.get('reviewer') or not r.get('evidence'):
            raise ValueError('Completed review needs boolean accepted, reviewer and evidence')
        if r.get('reviewer_kind') not in ['human','model_assisted']:
            raise ValueError('Set reviewer_kind to human or model_assisted')
        criteria=['requested_event_present','object_identity_consistent','motion_plausible']
        if any(type(r.get(k)) is not bool for k in criteria): raise ValueError('Complete every review criterion')
        if r['accepted'] and not all(r[k] for k in criteria): raise ValueError('Accepted case must pass all criteria')
        (accepted if r['accepted'] else rejected).append(r['id'])
    human_accepted = [r['id'] for r in reviews if r['id'] in accepted and r['reviewer_kind'] == 'human']
    human_completed = [r['id'] for r in reviews if r.get('accepted') is not None and r.get('reviewer_kind') == 'human']
    if a.analysis_manifest:
        for r in reviews:
            if r['id'] in human_accepted and (r.get('label') not in STATUS or not r.get('label_evidence')):
                raise ValueError('Human accepted cases need label and label_evidence from the viewed video')
            if r['id'] in human_accepted:
                item=generated_by_id[r['id']]
                if context(item)['scenario_id']!='legacy_unspecified' and not item.get('task'):
                    raise ValueError('Scenario generation must preserve its own task')
                if not (item.get('task') or a.task): raise ValueError('Legacy case requires --task')
                if a.task and item.get('task') and a.task!=item['task']:
                    raise ValueError('Global --task conflicts with the scenario task')
    conditions = {}
    for item in generated:
        key = group_key(item)+'::'+item.get('condition', item['id'])
        g = conditions.setdefault(key, {'generated':0, 'execution_errors':0, 'human_accepted':0, 'human_rejected':0, 'human_pending':0})
        g['generated'] += 1
        g['execution_errors'] += item['status'] != 'success'
        g['human_accepted'] += item['id'] in human_accepted
        g['human_rejected'] += item['id'] in human_completed and item['id'] not in human_accepted
        g['human_pending'] += item['status'] == 'success' and item['id'] not in human_completed
    write_json(a.output, {'human_accepted': human_accepted,
        'human_pending': sorted(known-set(human_completed)), 'by_condition': conditions,
        'human_acceptance_fraction':len(human_accepted)/len(generated) if generated else None,
        'note':'accepted/rejected retain all reviewer kinds for compatibility; only human_accepted enters the analysis manifest.',
        'review_source': file_info(a.reviews), 'total_generated':len(generated),'execution_success':len(known),
        'reviewed':len(accepted)+len(rejected),'accepted':accepted,'rejected':rejected,
        'pending':sorted(known-set(accepted)-set(rejected)),
        'accepted_fraction_of_all_attempts':len(accepted)/len(generated) if generated else None,
        'reviews':reviews})
    if a.analysis_manifest:
        review_map = {r['id']:r for r in reviews}
        write_json(a.analysis_manifest, {'review_source': file_info(a.reviews),
            'note': 'Only human accepted videos; zero cases means no reevaluation is possible yet.',
            'cases':[{'id':r['id'], 'source':r['video'], 'task':r.get('task') or a.task, **context(r), 'source_type':'synthetic',
                'evaluation_criteria':r.get('evaluation_criteria'),
                'source_provenance':r.get('source_provenance'), 'parent_case_id':r['id'],
                'parent_generation_run':str(a.run.resolve()),
                'condition':r.get('condition', r['id']), 'label':review_map[r['id']]['label'],
                'label_kind':'human', 'label_source':review_map[r['id']]['reviewer'],
                'label_evidence':review_map[r['id']]['label_evidence'],
                'label_stage':review_map[r['id']].get('label_stage'),
                'label_evidence_times_seconds':review_map[r['id']].get('label_evidence_times_seconds'),
                'video_artifact':file_info(r['video'])} for r in generated if r['id'] in human_accepted]})

if __name__=='__main__': main()
