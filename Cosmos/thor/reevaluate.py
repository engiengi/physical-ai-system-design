"""Compare preserved analysis runs by case pair and variant, without pooling retries."""
import argparse
import json
from pathlib import Path
from common import write_json
from artifacts import file_info
from evaluation import analyze_metrics
from scenarios import group_key, context


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summaries', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--review-summary', type=Path)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    runs=[];pairs={}
    for path in a.summaries:
        data=json.loads(path.read_text())
        if data['command'] != 'analyze': raise ValueError('Expected analysis summary')
        results=data['results']
        runs.append({'source':file_info(path), 'metrics':analyze_metrics(results)})
        for r in results:
            pairs.setdefault(group_key(r)+'::'+(r.get('pair_id') or r['id']), []).append({
                **context(r), 'run':str(path.parent), 'id':r['id'], 'variant':r.get('variant','original'),
                'retry_of':r.get('retry_of'), 'status':r['status'], 'error_kind':r.get('error_kind'),
                'prediction':r.get('prediction'), 'label':r.get('label'), 'label_source':r.get('label_source'),
                'seconds':r['seconds'], 'source_artifact':r.get('source_artifact')})
    review=json.loads(a.review_summary.read_text()) if a.review_summary else None
    write_json(a.output, {'runs':runs, 'pairs':pairs, 'generation_review':review,
        'note':'Each run has separate denominators. Paired variants and retries are not independent episodes. Review fields retain reviewer kind.'})

if __name__=='__main__': main()
