"""Frozen GUI evaluation plans, append-only attempts and explicit resume/retry."""
from .settings import URI
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
from .artifacts import summarize, write_json

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_plan(config, uri, batch_id):
    from .network import urlsplit
    from .runner import provenance
    parsed = urlsplit(uri)
    if parsed.scheme not in ('ws', 'wss') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use a credential-free ws(s) URI')
    count = config['episodes_per_condition']
    if type(count) is not int or count < 1:
        raise ValueError('Positive episodes_per_condition required')
    if type(config['seed_start']) is not int or config['seed_start'] < 0:
        raise ValueError('Nonnegative integer environment seed required')
    if type(config['execute_horizon']) is not int or config['execute_horizon'] < 1:
        raise ValueError('Positive integer execute_horizon required')
    if not np.isfinite(config.get('timeout',60)) or config.get('timeout',60)<=0:
        raise ValueError('Positive finite timeout required')
    names = [c['name'] for c in config['conditions']]
    if not names or len(names) != len(set(names)):
        raise ValueError('Unique condition names required')
    mode = config.get('mode', 'policy')
    if mode not in ('policy', 'smoke'):
        raise ValueError('Batch mode must be policy or smoke')
    seed_mode = config.get('policy_seed_mode', 'explicit')
    if seed_mode not in ('explicit', 'legacy'):
        raise ValueError('policy_seed_mode must be explicit or legacy')
    if config.get('scope') == 'formal' and (seed_mode != 'explicit' or config.get('max_steps', 0)):
        raise ValueError('Formal evaluation requires explicit seeds and uncapped episodes')
    items = []
    for ci, c in enumerate(config['conditions']):
        if not isinstance(c['name'], str) or not c['name'] or not np.isfinite(c['pose_range_m']) or c['pose_range_m'] < 0:
            raise ValueError('Invalid condition')
        for n in range(count):
            seed = config.get('policy_seed_start', 1000) + n * 10000
            if not 0 <= seed < 2**31:
                raise ValueError('Policy seed out of range')
            items.append({'id': f'c{ci:02d}_e{n:03d}', 'condition': c['name'],
                          'pose_range_m': c['pose_range_m'], 'environment_seed': config['seed_start'] + n,
                          'policy_seed': seed if seed_mode == 'explicit' and mode == 'policy' else None})
    return {'schema_version': 1, 'batch_id': batch_id, 'created_at': datetime.now().astimezone().isoformat(),
            'config': config, 'uri': uri, 'items': items, 'environment': provenance(),
            'seed_rule': 'episode policy_seed + request_index modulo 2**31; same episode bases across conditions',
            'record_thor': bool(config.get('record_thor', mode == 'policy')),
            'gui': not config.get('headless', False)}


def attempts_at(folder):
    return [json.loads(p.read_text()) for p in sorted((folder/'attempts').glob('*.json'))]


def item_completed(attempts, item_id):
    return any(a['item_id'] == item_id and a.get('result', {}).get('status') in ('success', 'task_failure', 'diagnostic_complete') for a in attempts)


def summarize_batch(plan, attempts):
    primary = []
    for item in plan['items']:
        rows = [a for a in attempts if a['item_id'] == item['id']]
        terminal = [a for a in rows if a.get('result', {}).get('status') in ('success', 'task_failure', 'diagnostic_complete')]
        if rows:
            # A task failure is final: resuming never silently replaces it with a later success.
            primary.append((terminal[0] if terminal else rows[0])['result'])
    all_results = [a['result'] for a in attempts]
    by_condition = {c['name']: summarize([r for r in primary if r['condition'] == c['name']]) for c in plan['config']['conditions']}
    values = [rtt for a in attempts for rtt in a.get('rtt_ms', [])]
    return {'planned': len(plan['items']), 'attempts': len(attempts), 'primary': summarize(primary),
            'all_attempts': summarize(all_results), 'conditions': by_condition,
            'not_started': sum(not any(a['item_id'] == i['id'] for a in attempts) for i in plan['items']),
            'retry_attempts': sum(max(0,n-1) for n in Counter(a['item_id'] for a in attempts).values()),
            'recording_errors': sum(bool(a.get('recording_errors')) for a in attempts),
            'rtt_ms_all_attempts': ({'count': len(values), 'mean': float(np.mean(values)),
                                   'p50': float(np.percentile(values,50)), 'p95': float(np.percentile(values,95))} if values else None),
            'human_review': None}


def build_command(plan, item, run_id):
    c = plan['config']
    command = [sys.executable, '-u', '-m', 'cosmos_spark.session' if plan['gui'] else 'cosmos_spark.runner',
               '--run-id', run_id, '--mode', c.get('mode','policy'), '--uri', plan['uri'],
               '--task', c['task'], '--episodes', '1', '--seed', str(item['environment_seed']),
               '--condition', item['condition'], '--pose-range-m', str(item['pose_range_m']),
               '--execute-horizon', str(c['execute_horizon']), '--timeout', str(c.get('timeout', 60))]
    if item['policy_seed'] is not None:
        command += ['--policy-seed', str(item['policy_seed'])]
    if c.get('max_steps'):
        command += ['--max-steps', str(c['max_steps'])]
    if not plan['gui']:
        command += ['--headless']
    else:
        command += ['--kit_args', '--/app/window/width=1600 --/app/window/height=900']
    return command


def run_item(plan, item, folder, index):
    from .remote import ThorRecording
    from robolab.eval.websocket_transport import MsgPackWebSocketTransport
    run_id = f"{plan['batch_id']}_{item['id']}_a{index:03d}"
    out = folder/'attempts'/f'{index:05d}.json'
    attempt = {'item_id': item['id'], 'run_id': run_id, 'attempt': index, 'started_at': time.time(),
               'result': {'status': 'incomplete', 'condition': item['condition']}, 'recording_errors': []}
    attempt['command'] = build_command(plan, item, run_id)
    write_json(out, attempt)  # Resumable evidence exists before any external process starts.
    recorder = process = None
    try:
        if plan['record_thor']:
            transport = MsgPackWebSocketTransport(plan['uri'], api_token=os.environ.get('COSMOS3_API_TOKEN'),
                metadata_timeout=5, connect_kwargs={'open_timeout':5,'close_timeout':2})
            try:
                metadata = transport.connect()
            finally:
                transport.close()
            recorder = ThorRecording(run_id, metadata['run_id'], ROOT/'captures'/run_id)
            attempt['thor_start'] = recorder.start()
        with (folder/f'{run_id}.log').open('w') as log:
            process = subprocess.Popen(attempt['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            attempt['exit_code'] = process.wait()
    except BaseException as exc:
        attempt['error'] = f'{type(exc).__name__}: {exc}'
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
        if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            attempt['result']['status'] = 'execution_error'
    finally:
        if recorder and 'thor_start' in attempt:
            try:
                attempt['thor_stop'] = recorder.stop()
            except Exception as exc:
                attempt['recording_errors'].append('Thor: '+str(exc))
        episode = ROOT/'runs'/run_id/'episode_0000'
        if (episode/'episode.json').exists():
            attempt['result'] = {**json.loads((episode/'episode.json').read_text()), 'condition': item['condition']}
        if (episode/'requests.jsonl').exists():
            attempt['rtt_ms'] = [r['rtt_ms'] for line in (episode/'requests.jsonl').read_text().splitlines()
                                 if (r:=json.loads(line)).get('rtt_ms') is not None]
        if plan['gui']:
            path=ROOT/'captures'/run_id/'session.json'
            session=json.loads(path.read_text()) if path.exists() else {}
            if not session.get('recording_valid') or session.get('recording_stopped_before_run_status'):
                attempt['recording_errors'].append('Spark GUI recording incomplete/missing')
        attempt['finished_at']=time.time()
        write_json(out,attempt)
    return attempt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path)
    p.add_argument('--uri',default=URI)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--retry-errors',action='store_true')
    p.add_argument('--max-items',type=int,default=0,help='Stop between completed items; 0 means all')
    a=p.parse_args()
    if a.max_items<0: p.error('max-items must be nonnegative')
    if a.resume:
        if a.config or a.plan_only: p.error('Resume uses the saved plan, omit --config/--plan-only')
        folder=a.resume.resolve()
        envelope=json.loads((folder/'plan.json').read_text());plan=envelope['plan']
        if envelope['sha256'] != digest(plan): raise ValueError('Saved plan changed; create a new batch')
    else:
        if not a.config: p.error('--config or --resume is required')
        batch='eval_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
        plan=make_plan(json.loads(a.config.read_text()),a.uri,batch)
        folder=ROOT/'runs'/batch;folder.mkdir()
        write_json(folder/'plan.json',{'plan':plan,'sha256':digest(plan)})
    print(f'Frozen batch: {folder}',flush=True)
    if a.plan_only:return
    with (folder/'batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.resume:
            from .runner import provenance
            current=provenance()
            if current['packages'] != plan['environment']['packages'] or current['robolab_commit'] != plan['environment']['robolab_commit'] or current['source_sha256'] != plan['environment']['source_sha256']:
                raise ValueError('Runtime/source changed since plan; create a separate batch instead of mixing implementations')
        done=0
        for item in plan['items']:
            attempts=attempts_at(folder)
            if item_completed(attempts,item['id']):continue
            if any(x['item_id']==item['id'] for x in attempts) and not a.retry_errors:
                raise SystemExit('Unfinished/error attempt exists; inspect it and use --retry-errors explicitly')
            row=run_item(plan,item,folder,len(attempts))
            write_json(folder/'summary.json',summarize_batch(plan,attempts_at(folder)))
            if row['result']['status'] not in ('success','task_failure','diagnostic_complete') or row.get('recording_errors') or row.get('exit_code',1):
                raise SystemExit('Stopped after incomplete/error attempt; raw evidence retained in '+str(folder))
            done+=1
            if a.max_items and done>=a.max_items:break
        write_json(folder/'summary.json',summarize_batch(plan,attempts_at(folder)))

if __name__=='__main__':main()
