"""Scenario identity and conservative support checks shared by experiment tools."""
import argparse
import json
from pathlib import Path
import re
from common import write_json
from artifacts import file_info

FIELDS=('scenario_id','experiment_track','source_type')
TRACKS={'new_standalone','banana_link'}
SOURCES={'real','simulation','synthetic'}


def context(case):
    if case.get('scenario_id')=='legacy_unspecified' and case.get('experiment_track')=='legacy_unspecified':
        return {k:case[k] for k in FIELDS}
    if not any(k in case for k in FIELDS):
        return dict(scenario_id='legacy_unspecified',experiment_track='legacy_unspecified',source_type=case.get('source_type','unspecified'))
    # Old manifests may only specify source_type; keep them explicitly unassigned.
    if not any(k in case for k in FIELDS[:2]):
        return dict(scenario_id='legacy_unspecified',experiment_track='legacy_unspecified',source_type=case['source_type'])
    if not isinstance(case.get('scenario_id'),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',case['scenario_id']):
        raise ValueError('scenario_id must be a safe nonempty identifier')
    if case.get('experiment_track') not in TRACKS or case.get('source_type') not in SOURCES:
        raise ValueError('Scenario cases need experiment_track new_standalone/banana_link and source_type real/simulation/synthetic')
    return {k:case[k] for k in FIELDS}


def group_key(case):
    return json.dumps(context(case),sort_keys=True,separators=(',',':'))


def support_check(case, command):
    ctx=context(case)
    if ctx['scenario_id']=='legacy_unspecified': return {'status':'legacy_unverified','reason':'No explicit scenario metadata'}
    if not isinstance(case.get('source_provenance'),str) or not case['source_provenance'].strip():
        raise ValueError('Scenario case requires source_provenance')
    if not isinstance(case.get('task'),str) or not case['task'].strip():
        raise ValueError('Scenario case requires its own task (also for generation reevaluation)')
    criteria=case.get('evaluation_criteria')
    if not isinstance(criteria,list) or not criteria or any(not isinstance(s,str) or not s.strip() for s in criteria):
        raise ValueError('Scenario case requires observable evaluation_criteria; never supply answer labels here')
    if command=='generate' and case.get('mode')=='forward_dynamics':
        if case.get('domain_name')!='droid_lerobot':
            raise ValueError('Only the locally verified droid_lerobot forward-dynamics path is allowed')
        if case.get('prediction_timing') not in ['retrospective','prospective']:
            raise ValueError('Specify retrospective or prospective prediction_timing')
        import numpy as np
        motion=np.asarray(json.loads(Path(case['action_path']).read_text()),dtype=float)
        if motion.shape!=(case.get('action_chunk_size',16),10) or not np.isfinite(motion).all():
            raise ValueError('Verified DROID recorded motion requires action_chunk_size x 10 finite values')
        if case['prediction_timing']=='prospective':
            raise ValueError('This recorded-motion path is retrospective. Use the policy action/video path with a validated DROID scene for prospective comparison.')
    return {'status':'input_contract_checked','model_scene_accuracy':'not_verified',
            'reason':'Schema compatibility does not establish support or accuracy for a new scene'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--command',choices=['analyze','generate'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from media import cases_from
    manifest,cases=cases_from(a.manifest.resolve(),a.command)
    write_json(a.output,{'manifest':file_info(a.manifest),'cases':[{'id':c['id'],**context(c),
        'input':file_info(c['source']),'support':support_check(c,a.command)} for c in cases],
        'note':'Preflight only. No model run, human labels, or scenario success claim.'})

if __name__=='__main__':main()
