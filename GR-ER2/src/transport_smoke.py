"""Physical delivery diagnostic with known coordinates, no Gemini calls."""
import argparse
import time
from app import bridge,run_dir
from common import ROOT,write_json,read_json
from coop_smoke import wait_idle


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['normal','load_failure','obstacle'],default='normal')
    p.add_argument('--load-only',action='store_true');p.add_argument('--navigation-only',action='store_true');args=p.parse_args()
    deadline=time.time()+90
    while time.time()<deadline:
        status=read_json(ROOT/'runtime/status.json')
        if status.get('ready') and status.get('world_scenario')=='transport' and time.time()-status['updated']<10:break
        time.sleep(2)
    else:raise RuntimeError('Start transport scene and wait for GR_ER2_READY transport')
    path=run_dir('07_cooperative/_diagnostics/transport_controller/'+args.condition)
    write_json(path/'experiment.json',{'scene':'transport','condition':args.condition,'mode':'controller_only_no_model','gemini_api_calls':0})
    print(path,flush=True);bridge('coop_reset',condition=args.condition);time.sleep(6)
    bridge('record_start',output=str(path.relative_to(ROOT)))
    try:
        if args.navigation_only:
            print(bridge('start_action',robot_id='Spot',action='navigate',arguments={'destination':'B','route':'direct'}),flush=True)
            jobs=wait_idle(600);truth=bridge('coop_truth')
            write_json(path/'smoke_summary.json',{'passed':jobs[-1]['status']=='arrived','mode':'navigation_only','truth':truth,'gemini_api_calls':0})
            print(truth,flush=True);return
        print(bridge('transport_smoke_action',robot_id='A'),flush=True);wait_idle(300)
        truth=bridge('coop_truth');print('AFTER LOAD',truth,flush=True)
        if args.condition=='load_failure':
            assert not truth['loaded'],'Failure injection ineffective'
            print(bridge('transport_smoke_action',robot_id='A'),flush=True);wait_idle(300)
            truth=bridge('coop_truth');print('AFTER RETRY',truth,flush=True)
        assert truth['loaded'],'Physical loading failed'
        if not args.load_only:
            print(bridge('start_action',robot_id='Spot',action='navigate',arguments={'destination':'B','route':'direct'}),flush=True)
            jobs=wait_idle(600)
            if args.condition=='obstacle':
                assert jobs[-1]['status']=='blocked'
                print(bridge('start_action',robot_id='Spot',action='navigate',arguments={'destination':'B','route':'detour'}),flush=True);jobs=wait_idle(600)
            assert jobs[-1]['status']=='arrived',jobs[-1]
            truth=bridge('coop_truth');print('AFTER CARRY',truth,flush=True);assert truth['loaded'],'Payload lost during walking'
            print(bridge('transport_smoke_action',robot_id='B'),flush=True);wait_idle(300)
            truth=bridge('coop_truth');print('AFTER UNLOAD',truth,flush=True)
            assert truth['physical_task_success'],'Physical unloading failed'
        write_json(path/'smoke_summary.json',{'passed':True,'truth':truth,'gemini_api_calls':0})
    except Exception as exc:
        write_json(path/'smoke_summary.json',{'passed':False,'error':str(exc),'truth':bridge('coop_truth'),'gemini_api_calls':0});raise
    finally:bridge('hold');bridge('record_stop')


if __name__=='__main__':main()
