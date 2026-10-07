"""Local human review UI, append-only decisions, and ground-truth-separated export."""
import argparse
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .artifacts import append_jsonl, manifest, write_json

STATUSES={'success','failure','in_progress','unknown'}
CRITERIA=('requested_event_present','object_identity_consistent','motion_plausible')


def latest_reviews(folder):
    delegated=folder/'delegated_review_events.jsonl'
    combined={r['id']:r for r in (json.loads(s) for s in delegated.read_text().splitlines())} if delegated.exists() else {}
    path=folder/'review_events.jsonl'
    events=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    combined.update({r['id']:r for r in events})
    return combined


def validate_review(case,review,*,delegated_authorization=None):
    allowed={'human','ai_delegated'} if delegated_authorization else {'human'}
    if review.get('reviewer_kind') not in allowed or not str(review.get('reviewer','')).strip():raise ValueError('Human reviewer name required, or explicit delegated authorization')
    if review.get('status') not in STATUSES or not str(review.get('evidence','')).strip():raise ValueError('Status and visible evidence required')
    if review['status']=='failure' and not str(review.get('failure_stage','')).strip():
        review={**review,'failure_stage':'unspecified','failure_stage_known':False}
    if case['source_type']=='synthetic':
        if type(review.get('accepted'))is not bool or any(type(review.get(k))is not bool for k in CRITERIA):
            raise ValueError('Complete acceptance and all generated-video criteria')
        if review['accepted'] and not all(review[k] for k in CRITERIA):raise ValueError('Accepted video must pass all criteria')
    return {**review,'id':case['id'],'video_sha256':case['sha256'],'saved_at_unix':time.time()}


def export(dataset,output,*,delegated_authorization=None):
    if delegated_authorization is not None and not str(delegated_authorization).strip():raise ValueError('Nonempty delegated authorization required')
    data=json.loads(dataset.read_text());reviews=latest_reviews(dataset.parent)
    output.mkdir(parents=True,exist_ok=False)
    inputs=[];references=[];pending=[];rejected=[]
    for case in data['cases']:
        r=reviews.get(case['id'])
        allowed={'human','ai_delegated'} if delegated_authorization else {'human'}
        if not r or r.get('reviewer_kind') not in allowed:pending.append(case['id']);continue
        validate_review(case,r,delegated_authorization=delegated_authorization)
        if r.get('video_sha256')!=case['sha256'] or hashlib.sha256(Path(case['video']).read_bytes()).hexdigest()!=case['sha256']:
            raise ValueError('Reviewed video changed: '+case['id'])
        if case['source_type']=='synthetic' and not r['accepted']:rejected.append(case['id']);continue
        if not re.fullmatch(r'[A-Za-z0-9_+.-]+',case['id']):raise ValueError('Unsafe case ID')
        media=output/'media'/(case['id']+'.mp4');media.parent.mkdir(exist_ok=True)
        shutil.copyfile(case['video'],media)
        inputs.append({'id':case['id'],'source':str(media.relative_to(output)),'task':case['task'],'source_type':case['source_type'],'fps':4,
                       'pair_id':case['group_id'],'variant':case['variant'],
                       **{k:case[k] for k in ('scenario_id','track','experiment_track','view_layout','source_provenance','evaluation_criteria') if k in case}})
        references.append({'id':case['id'],'label':r['status'],'label_source':r['reviewer_kind'],'reviewer':r['reviewer'],
                           'failure_stage':r.get('failure_stage'),'evidence':r['evidence'],'video_sha256':case['sha256'],
                           'group_id':case['group_id'],'variant':case['variant'],
                           **{k:case[k] for k in ('scenario_id','track','experiment_track','source_type') if k in case}})
    write_json(output/'analysis.json',{'cases':inputs})
    write_json(output/'references.json',{'cases':references,'delegated_authorization':delegated_authorization})
    write_json(output/'selection.json',{'total':len(data['cases']),'selected':len(inputs),'pending':pending,'rejected':rejected,
               'ready_for_analysis':bool(inputs),'labels_are_not_model_inputs':True,'delegated_authorization':delegated_authorization})
    manifest(output)
    return {'selected':len(inputs),'pending':len(pending),'rejected':len(rejected)}


def assess(references,summary,output):
    refs={r['id']:r for r in json.loads(references.read_text())['cases']}
    rows=json.loads(summary.read_text())['results'];seen=set();valid=[];errors=[]
    for row in rows:
        ident=row['id']
        if ident in seen:raise ValueError('Duplicate result ID')
        seen.add(ident)
        if ident not in refs:raise ValueError('Result has no reference: '+ident)
        prediction=row.get('prediction',{})
        if row.get('status')!='success' or prediction.get('status') not in STATUSES:
            errors.append({'id':ident,'error':row.get('error','invalid response')});continue
        valid.append({'id':ident,'label':refs[ident]['label'],'predicted':prediction['status'],
                      'variant':refs[ident].get('variant'),'group_id':refs[ident].get('group_id')})
    failures=[r for r in valid if r['label']=='failure']
    result={'references':len(refs),'attempted':len(rows),'valid':len(valid),'errors':errors,
            'missing':sorted(set(refs)-seen),'coverage':len(valid)/len(refs) if refs else None,
            'accuracy_valid':sum(r['label']==r['predicted'] for r in valid)/len(valid) if valid else None,
            'false_success_numerator':sum(r['predicted']=='success' for r in failures),'valid_failure_denominator':len(failures),
            'failure_reference_count':sum(r['label']=='failure' for r in refs.values()),
            'false_success_rate_valid_failures':sum(r['predicted']=='success' for r in failures)/len(failures) if failures else None,
            'rows':valid,'note':'Missing and invalid results are reported separately; variants are paired by group_id, not independent episodes.'}
    keys={(r.get('track','legacy'),r.get('scenario_id','legacy'),r.get('source_type','unspecified'),r.get('label_source','unspecified')) for r in refs.values()}
    groups=[]
    for track,scenario,source,label_source in sorted(keys):
        ids={i for i,r in refs.items() if (r.get('track','legacy'),r.get('scenario_id','legacy'),r.get('source_type','unspecified'),r.get('label_source','unspecified'))==(track,scenario,source,label_source)}
        good=[r for r in valid if r['id'] in ids];failed=[r for r in good if r['label']=='failure']
        groups.append({'track':track,'scenario_id':scenario,'source_type':source,'label_source':label_source,'references':len(ids),
            'valid':len(good),'errors':len([r for r in errors if r['id'] in ids]),'missing':sorted(ids-seen),
            'accuracy_valid':sum(r['label']==r['predicted'] for r in good)/len(good) if good else None,
            'false_success_numerator':sum(r['predicted']=='success' for r in failed),
            'valid_failure_denominator':len(failed),'failure_reference_count':sum(refs[i]['label']=='failure' for i in ids)})
    result['groups']=groups
    if len(groups)>1:
        result['accuracy_valid']=None
        result['false_success_rate_valid_failures']=None
        result['note']+=' Aggregate accuracy withheld across different scenarios/tracks/source types; inspect group metrics.'
    if output.exists():raise FileExistsError(output)
    write_json(output,result);return result


def handler_for(dataset,token):
    data=json.loads(dataset.read_text());cases={c['id']:c for c in data['cases']};lock=threading.Lock()
    if len(cases)!=len(data['cases']) or any(not re.fullmatch(r'[A-Za-z0-9_+.-]+',ident) for ident in cases):
        raise ValueError('Unique safe case IDs required')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            parsed=urlsplit(self.path)
            if parsed.path=='/':
                saved=latest_reviews(dataset.parent);cards=[]
                esc=html.escape
                for ident,c in cases.items():
                    r=saved.get(ident,{})
                    status_labels={'unknown':'판단 불가','success':'작업 성공','failure':'작업 실패','in_progress':'진행 중'}
                    opts=''.join(f'<option value="{s}" {"selected" if r.get("status")==s else ""}>{label}</option>' for s,label in status_labels.items())
                    context=f'<p>{esc(c.get("intent_ko",c.get("requested_event",c["task"])))}</p>'
                    if c['source_type']!='synthetic':
                        context=f'<p><b>확인할 작업:</b> {esc(c.get("task_ko",c["task"]))}</p>'
                        context+='<ul>'+''.join(f'<li>{esc(item)}</li>' for item in c.get('review_criteria_ko',[]))+'</ul>'
                    if c.get('scope_notice_ko'):context='<p><b>'+esc(c['scope_notice_ko'])+'</b></p>'+context
                    if c.get('intent_ko'):
                        context='<h3>생성하려던 변화</h3>'+context
                        context+='<h3>유지되어야 할 조건</h3><ul>'+''.join(f'<li>{esc(item)}</li>' for item in c.get('preserve_ko',[]))+'</ul>'
                        context+=f'<p>영상 속 원래 작업: {esc(c.get("task_ko",c["task"]))} — 아래 작업 상태와 생성 품질은 별도로 판단합니다.</p>'
                        context+=f'<details><summary>실제 생성 프롬프트</summary><p>{esc(c.get("requested_event",""))}</p></details>'
                    media_label='Cosmos 생성 영상' if c['source_type']=='synthetic' else '실제 시뮬레이션 기록' if c['source_type']=='simulation' else '원본 기록 영상'
                    if c.get('variant')=='frozen_final_frame':media_label+=' — 마지막 정지 프레임 반복 편집본'
                    media=f'<div><p>{media_label}</p><video id="video-{ident}" controls preload="metadata" src="/media?id={ident}"></video><button type="button" onclick="tailClip(\'video-{ident}\')">마지막 3초 · 0.5배속</button><button type="button" onclick="fullClip(\'video-{ident}\')">처음부터 · 정상 속도</button></div>'
                    if c.get('reference_image'):
                        media=f'<div><p>입력 원본 이미지</p><img alt="생성에 사용한 원본 장면" src="/reference?id={ident}"></div>'+media
                    media='<div class="media-pair">'+media+'</div>'
                    if c.get('model_outputs'):
                        media+='<details><summary>직접 판단한 뒤 모델 답변 비교하기</summary><p>모델 답변은 사람 정답이 아닙니다. 형식 유효 여부와 내용의 정확성을 구분합니다.</p>'
                        for answer in c['model_outputs']:
                            media+=f'<h3>{esc(answer["label"])}</h3><p>{esc(answer["summary_ko"])}</p><pre style="white-space:pre-wrap;overflow-wrap:anywhere">{esc(answer["raw"])}</pre>'
                        media+='</details>'
                    if c.get('known_issues_ko'):
                        context+=f'<details><summary>AI 예비 관찰 (일부 프레임만 확인·사람 판정 아님)</summary><p>{esc(c["known_issues_ko"])}</p></details>'
                    checks=''
                    if c['source_type']=='synthetic':
                        labels={'requested_event_present':'요청 사건 재현','object_identity_consistent':'물체 일관성','motion_plausible':'움직임 타당성','accepted':'평가에 채택'}
                        for k,label in labels.items():
                            checks+=f'<label>{label}<select name="{k}" required><option value="">선택</option><option value="true" {"selected" if r.get(k)is True else ""}>예</option><option value="false" {"selected" if r.get(k)is False else ""}>아니요</option></select></label>'
                    cards.append(f'''<section id="{ident}"><h2>{esc(c.get('display_title',ident))}</h2><p>{esc(ident)} · {('AI 대리 검수 저장됨' if r.get('reviewer_kind')=='ai_delegated' else '사람 검토 저장됨') if r else '검토 대기'}</p>{context}{media}<form method="post" action="/review"><input type="hidden" name="token" value="{token}"><input type="hidden" name="id" value="{esc(ident)}"><label>검토자<input name="reviewer" required value="{esc(r.get('reviewer','') if r.get('reviewer_kind')=='human' else '')}"></label><label>영상 속 원래 작업 상태 (생성 품질과 별도)<select name="status">{opts}</select></label><label>작업 실패 단계 (선택·모르면 비워두세요)<input name="failure_stage" value="{esc(r.get('failure_stage',''))}"></label><label>장면 근거·영상 시점<textarea name="evidence" required>{esc(r.get('evidence',''))}</textarea></label>{checks}<button>사람 검토 결과 저장</button></form></section>''')
                body=('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Cosmos 영상 공동 검토</title><style>body{font:16px/1.6 system-ui;background:#121a24;color:#eef3fa;max-width:1100px;margin:auto;padding:30px}section{padding:24px;margin:24px 0;background:#202d3d;border-radius:12px}h2{font-size:17px;overflow-wrap:anywhere}video{width:100%;max-height:500px;background:#000}.media-pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.media-pair img{width:100%;max-height:500px;object-fit:contain}details{margin:12px 0}label{display:block;margin:12px 0}input,textarea,select,button{font:inherit;padding:8px;margin:4px;max-width:95%}textarea{display:block;width:95%}button{background:#b9df8d;border:0;border-radius:8px}</style><h1>Cosmos 영상 공동 검토</h1><p>전체 영상을 보고 판정과 근거를 기록해주세요. 저장한 검토는 이력으로 남습니다. 보강 영상의 마지막 정지 프레임은 추가 물리 관측이 아닙니다. 생성 요청은 정답이 아닙니다.</p><p>생성 영상은 Isaac Sim의 물리 시뮬레이션 결과가 아닙니다. 원본과 비교해 요청한 변화, 물체 일관성, 움직임 타당성을 각각 확인하세요. 문제가 있으면 평가에 채택을 ‘아니요’로 기록할 수 있습니다. 작업 성공 여부와 생성 품질은 별개입니다.</p>'''+''.join(cards)).encode()
                body+=b'''<script>function withVideo(id,fn){const v=document.getElementById(id);if(v.readyState>=1){fn(v)}else{v.addEventListener('loadedmetadata',()=>fn(v),{once:true});v.load()}}function tailClip(id){withVideo(id,v=>{v.playbackRate=.5;v.currentTime=Math.max(0,v.duration-3);v.play().catch(()=>{})})}function fullClip(id){withVideo(id,v=>{v.playbackRate=1;v.currentTime=0;v.play().catch(()=>{})})}</script>'''
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            if parsed.path=='/reference':
                ident=parse_qs(parsed.query).get('id',[''])[0]
                if ident not in cases or not cases[ident].get('reference_image'):self.send_error(404);return
                path=Path(cases[ident]['reference_image'])
                if not path.is_file():self.send_error(404);return
                body=path.read_bytes()
                if hashlib.sha256(body).hexdigest()!=cases[ident].get('reference_image_sha256'):self.send_error(409,'Reference image changed');return
                self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            if parsed.path=='/media':
                ident=parse_qs(parsed.query).get('id',[''])[0]
                if ident not in cases:self.send_error(404);return
                path=Path(cases[ident]['video']);size=path.stat().st_size;start=0;end=size-1
                range_header=self.headers.get('Range')
                if range_header:
                    match=re.fullmatch(r'bytes=(\d+)-(\d*)',range_header)
                    if not match:self.send_error(416);return
                    start=int(match[1]);end=min(size-1,int(match[2]) if match[2] else size-1)
                    if start>end or start>=size:self.send_error(416);return
                self.send_response(206 if range_header else 200);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes')
                if range_header:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                self.send_header('Content-Length',str(end-start+1));self.end_headers()
                try:
                    with path.open('rb') as f:
                        f.seek(start);left=end-start+1
                        while left:
                            block=f.read(min(left,1024*1024))
                            if not block:break
                            self.wfile.write(block);left-=len(block)
                except (BrokenPipeError,ConnectionResetError):pass
                return
            self.send_error(404)
        def do_POST(self):
            if self.path!='/review':self.send_error(404);return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:raise ValueError('Invalid request size')
                form={k:v[0] for k,v in parse_qs(self.rfile.read(length).decode(),keep_blank_values=True).items()}
                if not secrets.compare_digest(form.pop('token',''),token):self.send_error(403);return
                ident=form.pop('id','')
                if ident not in cases:raise ValueError('Unknown case')
                review={k:form.get(k,'') for k in ('reviewer','status','evidence','failure_stage')}
                review['reviewer_kind']='human'
                if cases[ident]['source_type']=='synthetic':
                    for k in (*CRITERIA,'accepted'):
                        if form.get(k) not in ('true','false'):raise ValueError('All criteria required')
                        review[k]=form[k]=='true'
                row=validate_review(cases[ident],review)
                if hashlib.sha256(Path(cases[ident]['video']).read_bytes()).hexdigest()!=cases[ident]['sha256']:raise ValueError('Video changed since dataset creation')
                with lock:append_jsonl(dataset.parent/'review_events.jsonl',row)
                self.send_response(303);self.send_header('Location','/');self.end_headers()
            except (ValueError,KeyError) as exc:self.send_error(400,str(exc))
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('serve');s.add_argument('dataset',type=Path);s.add_argument('--port',type=int,default=8765)
    e=sub.add_parser('export');e.add_argument('dataset',type=Path);e.add_argument('--output',type=Path,required=True)
    e.add_argument('--delegated-authorization',help='Record the user authorization for AI-delegated review; labels remain ai_delegated')
    a=sub.add_parser('assess');a.add_argument('--references',type=Path,required=True);a.add_argument('--summary',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='export':print(json.dumps(export(args.dataset.resolve(),args.output.resolve(),delegated_authorization=args.delegated_authorization)))
    elif args.command=='assess':print(json.dumps(assess(args.references,args.summary,args.output)))
    else:
        server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(args.dataset.resolve(),secrets.token_urlsafe(32)))
        print(f'Human review: http://127.0.0.1:{server.server_port}',flush=True)
        try:server.serve_forever()
        finally:server.server_close()
if __name__=='__main__':main()
