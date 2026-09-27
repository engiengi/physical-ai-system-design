"""API-free controller check. Preselected targets are NOT ER2 predictions."""
import argparse
import time
from app import bridge,run_dir
from common import ROOT,write_json


def wait_idle(seconds=240):
    deadline=time.time()+seconds
    while time.time()<deadline:
        jobs=bridge('get_job_status')['jobs']
        if all(j['status']!='running' for j in jobs):return jobs
        time.sleep(2)
    raise TimeoutError('Controller smoke timeout')


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['dual_franka'],required=True)
    args=p.parse_args();path=run_dir('07_cooperative/_diagnostics/controller_smoke_'+args.scene)
    write_json(path/'experiment.json',{'scene':args.scene,'mode':'controller_only_no_model',
        'note':'Preselected calibrated image points / map waypoints. No Gemini API call.'})
    print(path,flush=True)
    bridge('coop_reset',condition='normal')
    time.sleep(2);bridge('record_start',output=str(path.relative_to(ROOT)))
    try:
        if args.scene=='dual_franka':
            obs=bridge('observe',camera_id='overview')
            for robot,pick,place in [('A',[689,465],[845,550]),('B',[311,465],[155,550])]:
                print(bridge('start_action',robot_id=robot,action='pick_place',arguments={
                    'observation_id':obs['observation_id'],'pick':pick,'place':place}),flush=True)
            jobs=wait_idle();truth=bridge('coop_truth')
            passed=truth['tray_counts']['A']['red']==1 and truth['tray_counts']['B']['blue']==1
        write_json(path/'smoke_summary.json',{'passed':passed,'jobs':jobs,'evaluation_truth':truth,'gemini_api_calls':0})
        print('CONTROLLER SMOKE '+('PASS' if passed else 'FAIL'),flush=True)
        if not passed:raise RuntimeError('Controller smoke failed; inspect recordings before model runs')
    finally:
        bridge('hold');bridge('record_stop')


if __name__=='__main__':main()
