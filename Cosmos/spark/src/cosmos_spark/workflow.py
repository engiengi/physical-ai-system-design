"""Evidence-linked progress board retaining the original nine stages."""
import argparse
import html
import json
import os
from pathlib import Path
import time
from .artifacts import append_jsonl, write_json

STAGES=['실험 목표·모델·평가 조건 확정','Spark 시뮬레이션·Thor 모델 환경 준비','두 장비 연결·첫 작업 실행',
        '행동 생성 반복 평가','실패 분석·수정·재평가','Cosmos 작업 결과 분석·검증','미래 예측·실제 결과 비교',
        '추가 상황 생성·분석 재평가','최종 리뷰·전체 원고 완성']


def render(folder):
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()] if (folder/'events.jsonl').exists() else []
    sections=[];records=[]
    for i,title in enumerate(STAGES,1):
        rows=[r for r in events if r['stage']==i]
        links=[]
        for r in rows:
            target=html.escape(os.path.relpath(r['artifact'],folder),quote=True)
            links.append(f'<li>{html.escape(r["kind"])} · <a href="{target}">{html.escape(Path(r["artifact"]).name)}</a> — {html.escape(r["note"])}</li>')
        records.append({'stage':i,'title':title,'evidence':rows,'human_final_review':None,'status':'검토 중' if rows else '미착수'})
        sections.append(f'<section><h2>{i}. {html.escape(title)}</h2><p>실제 수행·수정 자료와 원고를 연결했습니다. 사람의 최종 단계 검토는 대기입니다.</p><ul>{"".join(links) or "<li>등록된 자료 없음</li>"}</ul></section>')
    write_json(folder/'stages.json',records)
    (folder/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Cosmos 9단계 검토 보드</title><style>body{background:#152030;color:#eef;font:16px/1.6 system-ui;max-width:1100px;margin:auto;padding:30px}a{color:#9de}section{padding:20px;background:#253247;margin:20px 0}</style><h1>원본 9단계 검토 보드</h1><p>구현 검증과 실험 단계 완료를 구분합니다. 자동으로 사람 검토를 완료하지 않습니다.</p>'+''.join(sections))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--stage',type=int,choices=range(1,10))
    p.add_argument('--artifact',type=Path);p.add_argument('--kind',choices=['실제 수행 결과','공동 검토 자료','수정·재검토 기록','해당 단계 원고'],default='실제 수행 결과')
    p.add_argument('--note',default='')
    a=p.parse_args();a.folder.mkdir(parents=True,exist_ok=True)
    if a.artifact:
        if a.stage is None or not a.artifact.exists():p.error('Existing artifact and --stage required')
        append_jsonl(a.folder/'events.jsonl',{'stage':a.stage,'kind':a.kind,'artifact':str(a.artifact.resolve()),'note':a.note,'registered_at':time.time()})
    render(a.folder.resolve());print(a.folder/'index.html')
if __name__=='__main__':main()
