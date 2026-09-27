import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from coop_rules import validate_job, public_job, evaluate_arm, overlap_seconds, progress_bracket


class CooperativeTests(unittest.TestCase):
    def test_valid_job_and_bad_fields(self):
        validate_job('transport', 'Spot', 'navigate', {'destination': 'B','route':'direct'})
        for r, a in [('C', {'destination': 'B','route':'direct'}), ('Spot', {'destination': 'truth','route':'direct'})]:
            with self.assertRaises(ValueError): validate_job('transport', r, 'navigate', a)
        with self.assertRaises(ValueError):
            validate_job('dual_franka', 'A', 'pick_place', {'pick': [float('nan'), 0], 'place': [1, 2], 'observation_id': 'abc'})

    def test_private_metrics_not_exposed(self):
        self.assertEqual(public_job({'job_id':'1', 'robot_id':'A', 'truth':True, 'picked_object':'red', 'status':'sequence_completed'}),
                         {'job_id':'1', 'robot_id':'A', 'status':'sequence_completed'})

    def test_physical_goal_requires_both_colors_each(self):
        o = {'red_1':[.40,-.72,.04], 'blue_1':[.52,-.72,.04], 'red_2':[.40,.72,.04], 'blue_2':[.52,.72,.04]}
        self.assertTrue(evaluate_arm(o)['physical_task_success'])
        o['blue_2'] = [.5,0,.04]
        self.assertFalse(evaluate_arm(o)['physical_task_success'])
        self.assertEqual(evaluate_arm(o)['progress_fraction'], .75)

    def test_overlap_and_brackets(self):
        self.assertEqual(overlap_seconds([{'robot_id':'A','started_wall':0,'ended_wall':10},
                                        {'robot_id':'B','started_wall':5,'ended_wall':15}]),5)
        self.assertEqual(progress_bracket(1),'80-100')
        self.assertEqual(progress_bracket(.25),'20-40')

    def test_no_evaluator_tools_registered(self):
        from coop_run import TOOLS
        self.assertEqual({t['name'] for t in TOOLS},
            {'observe','start_action','get_job_status','cancel_job','report_progress','finish_task'})

    def test_manual_has_cooperative_anchors_and_commands(self):
        root=Path(__file__).resolve().parents[1]
        text=(root/'실행방법.md').read_text()
        for anchor in ['dual-franka','transport','coop-results']:
            self.assertIn(f'](#{anchor})',text)
            self.assertIn(f'<a id="{anchor}"></a>',text)
        for scene,conditions in [('dual_franka',['normal','exception']),('transport',['normal','load_failure','obstacle'])]:
            for condition in conditions:
                self.assertIn(f'--scene {scene} --condition {condition}',text)
        self.assertIn('src/coop_progress.py',text)

    def test_transport_does_not_teleport_payload_during_control(self):
        import ast
        path=Path(__file__).resolve().parents[1]/'src/transport_scene.py'
        tree=ast.parse(path.read_text())
        for method in [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('start','tick')]:
            for call in [n for n in ast.walk(method) if isinstance(n,ast.Call)]:
                if isinstance(call.func,ast.Attribute) and call.func.attr=='set_world_pose':
                    self.assertNotEqual(ast.unparse(call.func.value),'self.payload')

    def test_transport_disallows_arbitrary_route(self):
        with self.assertRaises(ValueError):validate_job('transport','Spot','navigate',{'destination':'B','route':'teleport'})

    def test_old_dual_spot_removed(self):
        with self.assertRaises(ValueError):validate_job('dual_spot','A','navigate',{'destination':'S1_front'})

    def test_transport_overview_is_rejected(self):
        root=Path(__file__).resolve().parents[1]
        text=(root/'src/coop_run.py').read_text()
        self.assertIn("kw['camera_id'] not in ('A','B','Spot','cargo')",text)
        self.assertIn('overview is viewer-only',text)

    def test_policy_sleep_override_after_initialize(self):
        root=Path(__file__).resolve().parents[1]
        text=(root/'src/transport_scene.py').read_text()
        self.assertLess(text.index('self.spot.initialize()'),text.index('self.spot.robot.set_sleep_thresholds([0.])'))

    def test_depth_refinement_uses_local_top(self):
        import numpy as np
        from transport_geometry import refine_block_top
        depth=np.full((480,640),2.075);depth[390:405,330:345]=2.025
        x,y,z=refine_block_top(depth,335,393)
        self.assertAlmostEqual(x,337);self.assertAlmostEqual(y,397);self.assertAlmostEqual(z,2.025)
        with self.assertRaises(ValueError):refine_block_top(depth,100,100)

    def test_warmup_refreshes_policy_proprioception(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'src/coop_sim.py').read_text())
        warm=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='warm')
        steps=[n for n in ast.walk(warm) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step']
        self.assertTrue(steps)
        self.assertTrue(all(any(k.arg=='update_fabric' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords) for n in steps))
