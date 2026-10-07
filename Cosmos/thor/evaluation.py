"""Denominator-explicit video judgments; invalid answers never become task failures."""
STATUS = {'success', 'failure', 'in_progress', 'unknown'}


def _metrics(results):
    valid = [r for r in results if r.get('prediction', {}).get('status') in STATUS and r['status'] == 'success']
    labeled = [r for r in results if r.get('label') in STATUS and r.get('label_source')]
    scored = [r for r in labeled if r in valid]
    failed_truth = [r for r in labeled if r['label'] == 'failure']
    wrong_success = [r for r in failed_truth if r.get('prediction', {}).get('status') == 'success']
    correct = sum(r['prediction']['status'] == r['label'] for r in scored)
    def ratio(n, d):
        return {'numerator': n, 'denominator': d, 'value': n/d if d else None}
    human = [r for r in results if r.get('label_kind') == 'human']
    groups = {}
    for r in results:
        key = str(r.get('condition', 'unspecified'))
        g = groups.setdefault(key, {'total':0, 'valid':0, 'format_errors':0, 'inference_errors':0, 'content_mismatches':0})
        g['total'] += 1
        g['valid'] += r in valid
        g['format_errors'] += r.get('error_kind') == 'format_error'
        g['inference_errors'] += r.get('error_kind') == 'inference_error'
        g['content_mismatches'] += r in scored and r['prediction']['status'] != r['label']
    return {'valid_response': ratio(len(valid), len(results)),
            'format_errors': sum(r.get('error_kind') == 'format_error' for r in results),
            'inference_errors': sum(r.get('error_kind') == 'inference_error' for r in results),
            'content_mismatches': len(scored)-correct,
            'labeled_total': len(labeled), 'labeled_completed': len(scored),
            'accuracy': correct/len(scored) if scored else None,
            'accuracy_on_valid_labeled': ratio(correct, len(scored)),
            'correct_among_all_labeled': ratio(correct, len(labeled)),
            'false_success': ratio(len(wrong_success), len(failed_truth)),
            'failure_cases_without_valid_response': sum(r not in valid for r in failed_truth),
            'false_success_among_failures': len(wrong_success)/len(failed_truth) if failed_truth else None,
            'label_kinds': sorted({str(r.get('label_kind', 'unspecified')) for r in labeled}),
            'human_only': {
                'labeled': len([r for r in human if r in labeled]),
                'accuracy_on_valid_labeled': ratio(sum(r['prediction']['status']==r['label'] for r in human if r in scored), len([r for r in human if r in scored])),
                'false_success': ratio(len([r for r in human if r in wrong_success]), len([r for r in human if r in failed_truth]))},
            'human_labeled_total': sum(r.get('label') in STATUS and bool(r.get('label_source')) for r in human),
            'by_condition': groups,
            'note': 'No failure labels means false-success rate is unmeasured. Invalid responses stay in total denominators. Simulator agreement is not human-rated accuracy.'}


def analyze_metrics(results):
    from scenarios import group_key
    groups={}
    for r in results: groups.setdefault(group_key(r),[]).append(r)
    summary=_metrics(results)
    summary['by_scenario']={key:_metrics(rows) for key,rows in groups.items()}
    summary['pooled_performance_allowed']=len(groups)<=1
    if len(groups)>1:
        for key in ['accuracy','accuracy_on_valid_labeled','correct_among_all_labeled','false_success','false_success_among_failures','human_only']:
            summary[key]=None
        summary['by_condition']={}
        summary['note']='Different scenarios/tracks/source types are reported separately in by_scenario. No pooled accuracy or false-success rate.'
    return summary
