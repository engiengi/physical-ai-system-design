"""Serve the official DROID policy with recorded OpenPI requests and timing."""
from __future__ import annotations
import argparse
import concurrent.futures
import dataclasses
from http import HTTPStatus
import json
import os
from pathlib import Path
import secrets
import threading
import time
import uuid

import numpy as np
from artifacts import policy_seed, file_info, video_info
from scenarios import context as scenario_context
from common import configure, gpu_lock, new_run, pack, unpack, provenance, validate_observation, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--decode-video', action='store_true')
    p.add_argument('--deadline-seconds', type=float, default=120)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if not 0 < a.deadline_seconds or not 1 <= a.port <= 65535:
        p.error('Positive deadline and a valid port required')
    root = configure(a.workspace)
    lockfile = gpu_lock(root)
    out = new_run(root, a.run)
    checkpoint = root / 'models/Cosmos3-Edge-Policy-DROID'
    token = os.environ.get('COSMOS_POLICY_TOKEN')
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.common.init import init_output_dir
    from cosmos_framework.scripts.action_policy_server_utils import disable_runtime_ema_for_frozen_config
    class LocalPolicyService(RobolabPolicyService):
        def _build_setup_args(self, args):
            # Same controlled local-input setup as the existing Thor experiments.
            # The optional text/video guardrail checkpoint is separately gated.
            setup = OmniSetupOverrides(checkpoint_path=args.checkpoint_path,
                output_dir=args.output_dir, sampler=args.sampler, guardrails=False).build_setup()
            init_output_dir(setup.output_dir)
            return disable_runtime_ema_for_frozen_config(setup)
    from websockets.sync.server import serve
    import torch
    args = RobolabServerArgs(checkpoint_path=str(checkpoint), output_dir=out / 'native',
                            format_prompt_as_json=True, decode_video=a.decode_video,
                            deterministic_seed=True, seed=a.seed, host=a.host, port=a.port)
    started = time.perf_counter()
    # A single worker owns CUDA initialization and inference, including thread-local state.
    worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        service = worker.submit(LocalPolicyService, args).result()
    except Exception as e:
        write_json(out / 'failure.json', {'stage': 'model_load', 'error': str(e)})
        worker.shutdown()
        raise
    metadata = {'run_id': a.run, 'model': 'Cosmos3-Edge-Policy-DROID',
                'model_revision': json.loads((checkpoint / 'download.json').read_text())['revision'],
                'action_shape': [32, 8], 'conditioning_fps': 15, 'decode_video': a.decode_video,
                'state_conditioned': True, 'protocol_version': 2,
                'policy_seed': {'field': 'policy_seed', 'range': [0, 2147483647],
                                'fallback': '(server_seed + global_request_id) % 2147483648'}}
    write_json(out / 'runtime.json', {**provenance(root), **metadata,
        'load_seconds': time.perf_counter() - started, 'policy': dataclasses.asdict(service.cfg),
        'host': a.host, 'port': a.port, 'deadline_seconds': a.deadline_seconds,
        'authentication_enabled': bool(token), 'optional_guardrail_model': False})
    gate = threading.Lock()
    counter = 0
    counter_lock = threading.Lock()

    def process_request(connection, request):
        if token and not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + token):
            return connection.respond(HTTPStatus.UNAUTHORIZED, 'Unauthorized\n')
        if request.path == '/healthz':
            return connection.respond(HTTPStatus.OK, 'ready\n')
        if request.path != '/':
            return connection.respond(HTTPStatus.NOT_FOUND, 'Unknown endpoint\n')

    def handler(ws):
        nonlocal counter
        connection_id = uuid.uuid4().hex
        ws.send(pack(metadata))
        for payload in ws:
            with counter_lock:
                idx = counter
                counter += 1
            dest = out / 'requests' / f'{idx:06d}'
            dest.mkdir(parents=True)
            record = {'run_id': a.run, 'request_id': idx, 'connection_id': connection_id,
                      'received_at_unix': time.time(), 'status': 'received'}
            acquired = False
            started = time.perf_counter()
            try:
                if not isinstance(payload, bytes):
                    raise ValueError('Binary MessagePack request required')
                obs = validate_observation(unpack(payload))
                session = obs.get('session_id', connection_id)
                if not isinstance(session, str) or len(session) > 256:
                    raise ValueError('session_id must be a short string')
                seed, seed_rule = policy_seed(obs, a.seed, idx)
                context = {}
                if "scenario_id" in obs or "experiment_track" in obs:
                    context.update(scenario_context(obs))
                for key in ['batch_id', 'condition_id', 'episode_id']:
                    if key in obs:
                        if not isinstance(obs[key], str) or not 0 < len(obs[key]) <= 256:
                            raise ValueError(f'{key} must be a nonempty string up to 256 characters')
                        context[key] = obs[key]
                if 'episode_request_index' in obs:
                    value = obs['episode_request_index']
                    if type(value) is not int or value < 0:
                        raise ValueError('episode_request_index must be a nonnegative integer')
                    context['episode_request_index'] = value
                record.update(session_id=session, prompt=obs['prompt'], policy_seed=seed, seed_rule=seed_rule,
                              client_context=context)
                (dest / 'observation.msgpack').write_bytes(payload)
                acquired = gate.acquire(blocking=False)
                if not acquired:
                    raise RuntimeError('Policy busy; run one environment or serialize requests')
                service.cfg = dataclasses.replace(service.cfg, seed=seed)
                def infer():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t = time.perf_counter()
                    result = service.infer(obs)
                    torch.cuda.synchronize()
                    return result, time.perf_counter() - t, {
                        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                        'method': 'PyTorch allocator peak; reset immediately before request; includes resident model'}
                future = worker.submit(infer)
                try:
                    result, seconds, memory = future.result(timeout=a.deadline_seconds)
                except concurrent.futures.TimeoutError:
                    # CUDA work cannot be cancelled safely. Keep the gate locked until it finishes.
                    future.add_done_callback(lambda _: gate.release())
                    acquired = False
                    raise TimeoutError('Inference deadline exceeded; action withheld; GPU remains busy until completion')
                action = np.asarray(result['action'])
                if action.shape != (32, 8) or not np.isfinite(action).all():
                    raise ValueError(f'Invalid policy action: {action.shape}')
                np.save(dest / 'action.npy', action)
                write_json(dest / 'action.json', action.tolist())
                record.update(seed=seed, inference_seconds=seconds, memory=memory, action_shape=list(action.shape))
                if 'video' in result:
                    import imageio.v3 as iio
                    iio.imwrite(dest / 'prediction.mp4', result['video'], fps=15, macro_block_size=1)
                artifact = {**metadata, 'request_id': idx, 'session_id': session,
                    'policy_seed': seed, 'seed_rule': seed_rule, 'client_context': context,
                    'observation': file_info(dest / 'observation.msgpack'),
                    'action': {**file_info(dest / 'action.npy'), 'shape': list(action.shape),
                               'semantics': '7 absolute joint positions (rad), raw gripper; Spark records threshold >0.5 postprocessing'},
                    'prediction': video_info(dest / 'prediction.mp4') if 'video' in result else None,
                    'saved_at_unix': time.time(), 'inference_seconds': seconds,
                    'comparison_kind': 'jointly_generated_action_and_video',
                    'transfer': 'Copy files separately (e.g. scp); verify SHA256 before applying actions.'}
                write_json(dest / 'artifacts.json', artifact)
                record['artifacts'] = artifact
                # Files are saved before the response. No future measured state enters this request.
                response = {'action': action, 'server_timing': {'infer_ms': seconds * 1000},
                            'run_id': a.run, 'request_id': idx, 'session_id': session,
                            'policy_seed': seed, 'seed_rule': seed_rule, 'artifacts': artifact,
                            'memory': memory, 'client_context': context}
                ws.send(pack(response))
                record['status'] = 'success'
            except Exception as e:
                record.update(status='error', error=f'{type(e).__name__}: {e}')
                try:
                    ws.send(pack({'type': 'error', 'message': str(e), 'run_id': a.run, 'request_id': idx}))
                except Exception:
                    pass
            finally:
                if acquired:
                    gate.release()
                record['handler_seconds'] = time.perf_counter() - started
                write_json(dest / 'result.json', record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    try:
        with serve(handler, a.host, a.port, compression=None, max_size=32 * 1024**2,
                   process_request=process_request, ping_timeout=180) as server:
            print(f'READY ws://{a.host}:{a.port} run={a.run}', flush=True)
            server.serve_forever()
    finally:
        worker.shutdown(wait=True)
        lockfile.close()

if __name__ == '__main__':
    main()
