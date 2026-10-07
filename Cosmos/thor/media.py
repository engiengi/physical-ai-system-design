"""Batch task-video analysis and generation using the local Cosmos Edge model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time
import subprocess
from artifacts import file_info, video_info
from evaluation import analyze_metrics
from scenarios import context, support_check

from common import configure, gpu_lock, new_run, provenance, write_json

STATUS = {'success', 'failure', 'in_progress', 'unknown'}


def final_json(text):
    answer = text.rsplit('</think>', 1)[-1].strip()
    if answer.startswith('```'):
        answer = re.sub(r'^```(?:json)?\s*|\s*```$', '', answer).strip()
    obj = json.loads(answer)
    if not isinstance(obj, dict) or obj.get('status') not in STATUS:
        raise ValueError('Expected a JSON object with a valid status')
    if any(not isinstance(obj.get(k), str) or not obj[k].strip() for k in ['stage', 'evidence']):
        raise ValueError('Nonempty stage and evidence required')
    return obj


def task_prompt(task, layout=None, version="v1"):
    if version == "plain-json-v2":
        camera = "The panels are synchronized views of the same single robot. " if layout == "droid_concat" else ""
        return (f"Watch the video. Task: {task}. {camera}"
            "Classify only the visible final task state. Use success if completion is visible, failure if a failed attempt is visible, "
            "in_progress if still underway, or unknown if evidence is insufficient. "
            "Return exactly one JSON object with exactly three string fields: status, stage, evidence. "
            "Allowed status values: success, failure, in_progress, unknown. stage is a short task stage; evidence describes visible facts. "
            "Do not add markdown fences, commentary, extra braces, or text outside the JSON object.")
    layout_note = 'The three panels show ONE physical robot through synchronized cameras: wrist view on top, two exterior views below. Do not count the panels as separate robots.' if layout == 'droid_concat' else ''
    return f'''Observe this recording. Requested task: {task}
{layout_note}
Identify the current stage and whether the requested task is visibly complete.
Do not assume objects or actions outside the camera view. The operator may be a human or a robot.
Use success only when the requested final state is visible, failure for a visible failed attempt,
in_progress when the visible task is still underway, and unknown when evidence is insufficient.
A short clip may not show the start or end of the task. A still image cannot prove a movement sequence.
Return your final answer as one JSON object with these string fields:
{{"status":"success|failure|in_progress|unknown","stage":"short stage","evidence":"visible evidence and uncertainty"}}'''


def cases_from(path, command):
    manifest = json.loads(path.read_text())
    cases = manifest['cases']
    if not isinstance(cases, list) or not cases:
        raise ValueError('Manifest must contain a nonempty cases list')
    names = set()
    for case in cases:
        name = case['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', name) or name in names:
            raise ValueError('Each case needs a unique safe id')
        names.add(name)
        source = Path(case['source'])
        case['source'] = str((path.parent / source).resolve() if not source.is_absolute() else source.resolve())
        if not Path(case['source']).is_file():
            raise FileNotFoundError(case['source'])
        if command == 'analyze':
            if not isinstance(case.get('task'), str) or not case['task'].strip():
                raise ValueError('A task description is required')
            if case.get('label') is not None and (case['label'] not in STATUS or not case.get('label_source')):
                raise ValueError('Labels require a valid status and explicit label_source')
            if not isinstance(case.get('fps', 4), (int, float)) or not 0 < case.get('fps', 4) <= 120:
                raise ValueError('Sampling fps must be in (0,120]')
            interval = case.get('input_range_seconds')
            if interval is not None and (len(interval) != 2 or not 0 <= interval[0] < interval[1]):
                raise ValueError('input_range_seconds must be [start,end], 0 <= start < end')
        if command == 'generate':
            if not isinstance(case.get('prompt'), str) or not case['prompt'].strip():
                raise ValueError('A nonempty generation prompt is required')
            if case.get('mode', 'i2v') not in ['i2v', 'forward_dynamics']:
                raise ValueError('Supported modes: i2v, forward_dynamics')
            if case.get('mode') == 'forward_dynamics':
                if not case.get('action_semantics'):
                    raise ValueError('Document action units/convention in action_semantics')
                action = Path(case['action_path'])
                case['action_path'] = str((path.parent / action).resolve() if not action.is_absolute() else action.resolve())
                if not Path(case['action_path']).is_file():
                    raise FileNotFoundError(case['action_path'])
                if not case.get('domain_name'):
                    raise ValueError('Forward dynamics requires the correct domain_name')
        support_check(case, command)
    return manifest, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['analyze', 'generate'])
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--tokens', type=int, default=1024)
    p.add_argument('--prompt-version', choices=['v1','plain-json-v2'], default='v1')
    a = p.parse_args()
    if a.tokens <= 0:
        p.error('--tokens must be positive')
    manifest, cases = cases_from(a.manifest.resolve(), a.command)
    root = configure(a.workspace)
    lock = gpu_lock(root)
    out = new_run(root, a.run)
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'progress.json', {'phase': 'loading', 'command': a.command, 'run': a.run})
    import torch
    from cosmos_framework.inference.common.init import init_script, init_output_dir
    from cosmos_framework.inference.args import OmniSetupOverrides, OmniSampleOverrides
    init_script()
    init_output_dir(out / 'runtime')
    setup = OmniSetupOverrides(checkpoint_path=str(root / 'models/Cosmos3-Edge'),
        output_dir=out / 'native', parallelism_preset='latency', guardrails=False).build_setup()
    if a.command == 'analyze':
        setup.experiment_overrides.append('model.config.load_vision_tokenizer=false')
    started = time.perf_counter()
    try:
        pipe = setup.get_inference_cls().create(setup)
    except Exception as e:
        write_json(out / 'failure.json', {'stage': 'load', 'error': str(e)})
        raise
    torch.cuda.synchronize()
    write_json(out / 'runtime.json', {**provenance(root), 'load_seconds': time.perf_counter() - started,
        'model': 'Cosmos3-Edge', 'model_revision': 'a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba',
        'optional_guardrail_model': False})
    results = []
    reviews = []
    for case in cases:
        dest = out / case['id']
        dest.mkdir()
        seed = int(case.get('seed', 0))
        source = file_info(case['source'])
        input_path = case['source']
        interval = case.get('input_range_seconds')
        if interval is not None:
            original = video_info(input_path)
            if interval[1] > original['duration_seconds'] + 0.05:
                raise ValueError('Input range exceeds source duration')
            input_path = str(dest / 'input_clip.mp4')
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', case['source'],
                '-ss', str(interval[0]), '-t', str(interval[1]-interval[0]), '-an',
                '-c:v', 'libx264', '-crf', '18', input_path], check=True)
        sample = {'name': case['id'], 'vision_path': input_path, 'seed': seed,
                  'output_dir': str(dest / 'native')}
        if a.command == 'analyze':
            sample.update(model_mode='reasoner', prompt=task_prompt(case['task'], case.get('view_layout'), a.prompt_version),
                          video_fps=case.get('fps', 4), max_new_tokens=a.tokens, do_sample=False)
        else:
            mode = case.get('mode', 'i2v')
            sample.update(model_mode='image2video' if mode == 'i2v' else mode, prompt=case['prompt'], resolution=case.get('resolution', '480'))
            if mode == 'i2v':
                sample['num_frames'] = case.get('num_frames', 121)
            else:
                sample.update(action_path=case['action_path'], domain_name=case['domain_name'],
                    action_chunk_size=case.get('action_chunk_size', 16), fps=case.get('fps', 15),
                    image_size=case.get('image_size', 480), view_point=case.get('view_point', 'concat_view'))
        if a.command == 'analyze' and case.get('evaluation_criteria'):
            sample['prompt'] += '\nObservable evaluation criteria:\n' + '\n'.join('- '+c for c in case['evaluation_criteria'])
        write_json(dest / 'input.json', sample)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        stage = 'inference_error'
        write_json(out / 'progress.json', {'phase': 'inference', 'command': a.command, 'run': a.run, 'case': case['id']})
        record = {**context(case), 'task':case.get('task'),
                  'evaluation_criteria':case.get('evaluation_criteria'),
                  'source_provenance':case.get('source_provenance'),
                  'label_stage':case.get('label_stage'), 'label_evidence':case.get('label_evidence'),
                  'label_evidence_times_seconds':case.get('label_evidence_times_seconds'),
                  'support':support_check(case,a.command),
                  'prediction_timing':case.get('prediction_timing'),
                  'id': case['id'], 'source': case['source'], 'seed': seed, 'status': 'failed',
                  'source_artifact': source, 'input_artifact': file_info(input_path),
                  'input_range_seconds': interval, 'sampling_fps': sample.get('video_fps'),
                  'prompt': sample['prompt'], 'prompt_version': a.prompt_version, 'condition': case.get('condition', case['id']),
                  'pair_id': case.get('pair_id'), 'variant': case.get('variant', 'original'),
                  'retry_of': case.get('retry_of'), 'label': case.get('label'),
                  'label_source': case.get('label_source'), 'label_kind': case.get('label_kind', 'unspecified')}
        try:
            args = OmniSampleOverrides.model_validate(sample)
            args.download(dest / 'inputs')
            native = pipe.generate([args.build_sample(model_config=pipe.model_config)])
            torch.cuda.synchronize()
            record['native'] = [x.model_dump(mode='json') for x in native]
            if not native or any(x.status != 'success' for x in native):
                raise RuntimeError('Model execution failed')
            if a.command == 'analyze':
                texts = list((dest / 'native').rglob('reasoner_text.txt'))
                if len(texts) != 1:
                    raise RuntimeError('Expected one reasoner output')
                raw = texts[0].read_text()
                (dest / 'answer.txt').write_text(raw)
                stage = 'format_error'
                record['prediction'] = final_json(raw)
                record['label'] = case.get('label')
                record['label_source'] = case.get('label_source')
            else:
                videos = list((dest / 'native').rglob('vision.mp4'))
                if len(videos) != 1:
                    raise RuntimeError('Expected one generated video')
                record['video'] = str(videos[0])
                record['video_artifact'] = video_info(videos[0])
                reviews.append({'id': case['id'], 'requested_event_present': None,
                    'object_identity_consistent': None, 'motion_plausible': None,
                    'accepted': None, 'reviewer': '', 'reviewer_kind': None, 'evidence': '',
                    'label': None, 'label_evidence': ''})
            record['status'] = 'success'
        except Exception as e:
            record['error'] = f'{type(e).__name__}: {e}'
            record['error_kind'] = stage
        record['seconds'] = time.perf_counter() - started
        record['memory'] = {'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                            'method': 'PyTorch allocator peak reset before case; includes resident model'}
        write_json(dest / 'result.json', record)
        results.append(record)
        print(json.dumps({k: record[k] for k in ['id', 'status', 'seconds']}, ensure_ascii=False), flush=True)
    summary = {'scenario_groups': sorted({__import__('scenarios').group_key(r) for r in results}), 'command': a.command, 'total': len(results), 'completed': sum(x['status'] == 'success' for x in results), 'results': results}
    if a.command == 'analyze':
        summary.update(analyze_metrics(results))
    else:
        write_json(out / 'review.template.json', {'cases': reviews, 'note': 'Fill after viewing every generated video; never infer acceptance from successful execution.'})
    write_json(out / 'summary.json', summary)
    write_json(out / 'progress.json', {'phase': 'finished', 'command': a.command, 'run': a.run, 'completed': summary['completed'], 'total': summary['total']})
    lock.close()
    if summary['completed'] != summary['total']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
