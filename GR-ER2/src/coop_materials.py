"""Offline export. Privileged truth is consumed here, never in model prompts."""
import argparse
import csv
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from common import ROOT, read_json, write_json, safe_child
from coop_rules import overlap_seconds


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def video(pattern, output, fps):
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',str(pattern),
                    '-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)


def export(directory):
    recording=read_json(directory/'recording.json');fps=recording['fps']
    for camera in [p.name for p in (directory/'frames').iterdir() if p.is_dir()]:
        parent='rollout_video' if camera=='side' else 'input_video'
        video(directory/'frames'/camera/'%06d.jpg',directory/parent/(camera+'.mp4'),fps)
    frames=lines(directory/'frames.jsonl');truth=lines(directory/'truth.jsonl');events=lines(directory/'events.jsonl')
    jobs=list(read_json(directory/'jobs.json').values())
    timeline=read_json(directory/'model_timeline.json') if (directory/'model_timeline.json').exists() else []
    reported=read_json(directory/'model_report.json') if (directory/'model_report.json').exists() else {}
    final=truth[-1]
    missing_depth=[]
    for event in events:
        if event['kind']!='depth_point_refinement':continue
        observation_id=event['observation_id'];source=ROOT/'runtime/coop_observations'/observation_id
        target=directory/'control_inputs'/observation_id;target.mkdir(parents=True,exist_ok=True)
        for name in ['depth.npy','meta.json','image.jpg']:
            if (source/name).exists():shutil.copy2(source/name,target/name)
            elif not (target/name).exists():missing_depth.append(str(target.relative_to(directory)/name))
    summary={'recorded_seconds':recording['simulation_seconds'],'frames':recording['frames'],
             'robot_job_count':len(jobs),'both_robots_used':{'A','B'}.issubset({j['robot_id'] for j in jobs}),
             'robots_used':sorted({j['robot_id'] for j in jobs}),
             'missing_control_input_files':missing_depth,
             'concurrent_job_wall_seconds':overlap_seconds(jobs),'model_turns':len(timeline),
             'model_report':reported,'final_ground_truth':final,'perturbations':[e for e in events if e['kind'] in ('route_obstacle_inserted','grasp_failure_injected')],
             'job_status_counts':{s:sum(j['status']==s for j in jobs) for s in set(j['status'] for j in jobs)},
             'tool_types_used':sorted({c['name'] for row in timeline for c in row['calls']}),
             'rejected_request_count':sum(e['kind']=='request_rejected' for e in events),
             'interpretation':'One bounded qualitative run, not a benchmark success rate. Concurrent jobs do not imply learned low-level coordination.'}
    if 'physical_task_success' in final:
        summary['physical_task_success']=final['physical_task_success']
        summary['completion_claim_matches_truth']=reported.get('success')==final['physical_task_success'] if reported else None
    if 'carrier_position' in final:
        summary['all_three_robots_used']=set(summary['robots_used'])=={'A','B','Spot'}
        summary['handoff_milestones']={k:final[k] for k in ['loaded_once','departed_loaded','arrived_loaded','physical_task_success']}
        summary['rerouted_after_block']=any(j['robot_id']=='Spot' and j['status']=='arrived' and j['arguments'].get('route')=='detour'
            and any(b['robot_id']=='Spot' and b['status']=='blocked' and b['ended_wall']<j['started_wall'] for b in jobs) for j in jobs)
        summary['loading_attempts']=sum(j['robot_id']=='A' for j in jobs)
        loads=sorted((j for j in jobs if j['robot_id']=='A'),key=lambda j:j['started_wall'])
        departures=[j for j in jobs if j['robot_id']=='Spot' and j['arguments'].get('destination')=='B']
        summary['departure_after_loading_retry']=bool(len(loads)>1 and departures and min(j['started_wall'] for j in departures)>loads[1].get('ended_wall',float('inf')))
        summary['model_requested_cameras']=sorted({c.get('arguments',{}).get('camera_id','') for row in timeline for c in row['calls'] if c['name']=='observe' and isinstance(c.get('arguments'),dict)})
    write_json(directory/'summary.json',summary)
    with (directory/'jobs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['job_id','robot_id','action','status','started_wall','ended_wall','arguments'])
        writer.writeheader()
        for job in jobs:writer.writerow({k:json.dumps(job[k]) if k=='arguments' else job.get(k) for k in writer.fieldnames})
    images=directory/'figures';images.mkdir(exist_ok=True)
    chosen={'initial':0,'final':frames[-1]['frame']}
    if jobs:
        last=max(jobs,key=lambda j:j['started_wall'])
        for label,when in [('last_action_start',last['started_wall']),('last_action_end',last.get('ended_wall',frames[-1]['wall_time']))]:
            chosen[label]=min(frames,key=lambda r:abs(r['wall_time']-when))['frame']
    for i,event in enumerate(summary['perturbations']):
        chosen[f'perturbation_{i+1}']=min(frames,key=lambda r:abs(r['wall_time']-event['wall_time']))['frame']
        affected=next((j for j in jobs if j['job_id']==event.get('job_id')),None)
        if affected and affected.get('ended_wall'):
            chosen[f'after_failed_attempt_{i+1}']=min(frames,key=lambda r:abs(r['wall_time']-affected['ended_wall']-2))['frame']
    for i,job in enumerate(j for j in jobs if j['status']=='blocked'):
        chosen[f'blocked_{i+1}']=min(frames,key=lambda r:abs(r['wall_time']-job['ended_wall']))['frame']
    if 'carrier_position' in final:
        for milestone in ['loaded_once','departed_loaded','arrived_loaded','physical_task_success']:
            found=next((r for r in truth if r.get(milestone)),None)
            if found:chosen[milestone]=found['frame']
    write_json(images/'sources.json',{name:next(r for r in frames if r['frame']==index) for name,index in chosen.items()})
    for name,index in chosen.items():
        for camera in (['overview','side','A','B','cargo'] if 'carrier_position' in final else ['overview','side']):
            shutil.copy2(directory/'frames'/camera/f'{index:06d}.jpg',images/f'{name}_{camera}.jpg')
    if 'carrier_position' in final:
        board=Image.new('RGB',(1280,1024),'#112035');ink=ImageDraw.Draw(board)
        board_font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
        panels=[('initial','A','1. SOURCE: material on conveyor A'),('loaded_once','A','2. LOADED: Franka A -> Spot tray'),
                ('arrived_loaded','B','3. CARRIED: Spot arrives at B'),('physical_task_success','B','4. UNLOADED: Franka B -> conveyor B')]
        for k,(milestone,camera,label) in enumerate(panels):
            x,y=(k%2)*640,(k//2)*512
            ink.text((x+12,y+5),label,font=board_font,fill='white')
            if milestone in chosen:
                board.paste(Image.open(directory/'frames'/camera/f'{chosen[milestone]:06d}.jpg'),(x,y+32))
            else:ink.text((x+100,y+230),'NOT REACHED',font=board_font,fill='#ff9b7e')
        board.save(images/'handoff_storyboard.jpg',quality=95)
    # Same simulation-time clock as the raw footage. Panels change only at real
    # request/response timestamps, not when controllers advance internally.
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,15) if Path(font_path).exists() else ImageFont.load_default()
    rendered=directory/'presentation_frames';rendered.mkdir(exist_ok=True)
    requests=[]
    for path in sorted((directory/'api').glob('*/request.json')):
        row=read_json(path);row['turn']=int(path.parent.name);requests.append(row)
    write_json(directory/'submitted_inputs.json',[
        {'turn':r['turn'],'request_wall':r['wall_time'],'observations':r['observations']} for r in requests])
    selected_frames=frames[::2]
    if selected_frames[-1]['frame']!=frames[-1]['frame']:selected_frames.append(frames[-1])
    for i,stamp in enumerate(selected_frames):
        canvas=Image.new('RGB',(1600,1040),'#112035');draw=ImageDraw.Draw(canvas)
        index=stamp['frame'];wall=stamp['wall_time']
        canvas.paste(Image.open(directory/'frames'/'side'/f'{index:06d}.jpg').resize((960,640)),(0,35))
        canvas.paste(Image.open(directory/'frames'/'overview'/f'{index:06d}.jpg').resize((480,360)),(0,680))
        draw.text((12,10),f'ISAAC SIM | sim +{stamp["elapsed_sim"]:.1f}s | recorded RGB (not generated future)',font=font,fill='white')
        draw.text((500,840),'TASK: A loads -> Spot carries -> B unloads' if 'carrier_position' in final else 'TASK: both trays red + blue',font=font,fill='white')
        phases=truth[index].get('controller_phases',{})
        phase_names=['approach','descend','settle','close fingers','lift','move to target','lower','release','retreat','return','home/idle']
        for k,(robot,phase) in enumerate(phases.items()):
            draw.text((500,880+k*24),f'LOCAL {robot}: {phase_names[min(10,phase)]}',font=font,fill='#ffd15e')
        if 'carrier_position' in final:
            draw.text((500,940),'EVALUATOR milestones (not API input):',font=font,fill='#c4a4ff')
            for k,(key,label) in enumerate([('loaded_once','LOADED'),('departed_loaded','CARRY'),('arrived_loaded','ARRIVED'),('physical_task_success','UNLOADED')]):
                draw.text((500+k*112,974),label,font=font,fill='#82ecac' if truth[index].get(key) else '#65748a')
        active=[j['robot_id']+': '+j['action']+' '+(j['arguments'].get('destination','')+' '+j['arguments'].get('route','')).strip()
                for j in jobs if j['started_wall']<=wall<=j.get('ended_wall',wall)]
        draw.text((500,700),'LOCAL JOBS (not new API output):',font=font,fill='#ffd15e')
        for n,text in enumerate(active or ['No active motor job']):draw.text((500,730+24*n),text,font=font,fill='white')
        past=[r for r in requests if r['wall_time']<=wall]
        if past:
            req=past[-1];obs=list(req['observations'].values())
            draw.text((990,10),f'API INPUT STILLS | request {req["turn"]} | clips: api_inputs/',font=font,fill='#80dbff')
            for n,selected in enumerate(obs):
                size=(600,450) if len(obs)==1 else (300,225)
                x,y=(990,40) if len(obs)==1 else (990+300*(n%2),40+225*(n//2))
                canvas.paste(Image.open(directory/selected['saved_image']).resize(size),(x,y))
                draw.rectangle((x,y,x+size[0],y+22),fill='#112035')
                age=max(0,req['wall_time']-selected['wall_time'])
                draw.text((x+4,y+2),f'{selected["camera_id"]} | observation age {age:.1f}s',font=font,fill='#80dbff')
        answers=[r for r in timeline if r['response_wall']<=wall]
        if answers:
            answer=answers[-1]
            draw.text((990,510),f'ACTUAL API OUTPUT | turn {answer["turn"]}',font=font,fill='#82ecac')
            text=answer['text']+'\n'+json.dumps([{'tool':c['name'],'arguments':c.get('arguments')} for c in answer['calls']],ensure_ascii=False)
            wrapped=[]
            for line in text.splitlines():wrapped+=textwrap.wrap(line,70)
            for n,line in enumerate(wrapped[:21]):draw.text((990,540+n*22),line,font=font,fill='white')
        canvas.save(rendered/f'{i:06d}.jpg',quality=85)
    video(rendered/'%06d.jpg',directory/'presentation'/'synchronized.mp4',fps/2)
    shutil.copy2(rendered/f'{len(selected_frames)-1:06d}.jpg',images/'synchronized_final.jpg')
    write_json(directory/'media_manifest.json',{
        'input_video':'Continuous RGB camera archive, not all frames are submitted to the model.',
        'api_inputs':'Acquired observation stills/clips, including observations superseded before submission. submitted_inputs.json and each request.json identify exactly used observations. Dual Franka submits stills only; transport submits local camera stills and available clips. Global overview is NOT provided to the transport orchestrator.',
        'rollout_video':'Unmodified Isaac Sim third-person recording.',
        'presentation':'Synchronized composition with actual input/output timestamps. Raw footage timebase is simulation seconds.',
        'summary':'Offline evaluator ground truth, withheld from model.'})
    (directory/'RESULTS.md').write_text('# Cooperative experiment results\n\n'
        '- First watch [synchronized video](presentation/synchronized.mp4).\n'
        '- Physical scene: [rollout video](rollout_video/side.mp4).\n'
        '- Exact model evidence and replies: `api_inputs/`, `api/`, `model_timeline.json`.\n'
        '- Evaluation: `summary.json`, `jobs.csv`, `truth.jsonl`.\n\n'
        '```json\n'+json.dumps({k:v for k,v in summary.items() if k!='final_ground_truth'},indent=2,ensure_ascii=False)+'\n```\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    index_root=ROOT/'outputs/07_cooperative'
    rows=[]
    for scene in ['dual_franka','transport']:
        for path in sorted((index_root/scene).glob('*/*/summary.json')):
            data=read_json(path);run=path.parent.relative_to(index_root)
            rows.append({'scene':scene,'condition':run.parts[1],'run_id':run.parts[2],
                'physical_task_success':data.get('physical_task_success',''),
                'model_claimed_success':data['model_report'].get('success',''),
                'both_robots_used':data['both_robots_used'],'concurrent_job_wall_seconds':round(data['concurrent_job_wall_seconds'],2),
                'jobs':data['robot_job_count'],'model_turns':data['model_turns'],'video_seconds':round(data['recorded_seconds'],1),
                'presentation':str(run/'presentation/synchronized.mp4')})
    if rows:
        with (index_root/'results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (index_root/'README.md').write_text('# Multi-robot ER2 experiments\n\n'
            'Each run is a qualitative case, not a benchmark success-rate estimate. `results.csv` reports physical task success separately from the model completion claim.\n\n'
            '| Scene | Condition | Run / video |\n|---|---|---|\n'+''.join(
                f'| {r["scene"]} | {r["condition"]} | [{r["run_id"]}]({r["presentation"]}) |\n' for r in rows)+
            '\nRaw inputs: each run `input_video/`, `api_inputs/`. Raw outputs: `api/`, `model_report.json`. Ground-truth evaluation: `summary.json`.\n')
        transport_rows=[r for r in rows if r['scene']=='transport']
        if transport_rows:
            labels={'normal':'정상 운송','load_failure':'적재 실패·재시도','obstacle':'통로 장애·우회'}
            content='# Franka–Spot–Franka 자재 운송 결과\n\n'
            content+='먼저 통합 영상을 보고, 4장 비교 이미지와 각 실행의 `summary.json`을 확인합니다. 입력 원본은 `input_video/`, 실제 API 제출은 `submitted_inputs.json`, 모델 원문은 `api/`입니다.\n\n'
            content+='| 조건 | 물리 성공 | 모델 성공 보고 | 통합 영상 | 인계 비교 사진 |\n|---|---|---|---|---|\n'
            for row in transport_rows:
                path=Path(row['presentation']).relative_to('transport');base=path.parent.parent
                content+=f'| {labels.get(row["condition"],row["condition"])} | {row["physical_task_success"]} | {row["model_claimed_success"]} | [영상]({path}) | [4장 비교]({base}/figures/handoff_storyboard.jpg) |\n'
            content+='\n조건별 개별 실행 사례입니다. 일반 성공률을 뜻하지 않습니다. 컨베이어는 인계 위치에서 정지한 구성입니다.\n'
            (index_root/'transport/README.md').write_text(content)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    export(safe_child(ROOT,args.run))
