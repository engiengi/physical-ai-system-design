import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from scenario_rules import evaluate, skill_request, steering


class ScenarioTests(unittest.TestCase):
    def test_skill_whitelist(self):
        with self.assertRaises(ValueError):
            skill_request("drawer", {"skill": "navigate", "route": "center"})
        with self.assertRaises(ValueError):
            skill_request("patrol", {"skill": "navigate", "route": "unknown"})
        self.assertEqual(skill_request("drawer", {"skill": "open_drawer", "seconds": 9999})["seconds"], 55)

    def test_velocity_bounds(self):
        command = steering([0, 0], 0, [-10, 0])
        self.assertEqual(command[0], 0)
        self.assertLessEqual(abs(command[2]), .8)
        self.assertLessEqual(steering([0, 0], 0, [10, 0])[0], .45)

    def test_drawer_recovery_needs_after_event(self):
        rows = [{"wall_time": 1, "drawer_open_m": .25}, {"wall_time": 3, "drawer_open_m": 0}]
        event = [{"type": "drawer_closed", "wall_time": 2}]
        self.assertFalse(evaluate("drawer", rows, event, [])["reopened_after_closure"])
        rows.append({"wall_time": 4, "drawer_open_m": .2})
        self.assertTrue(evaluate("drawer", rows, event, [])["opening_threshold_met"])
        self.assertFalse(evaluate("drawer", rows, event, [])["task_success"])

    def test_patrol_needs_arrival_and_visual_report(self):
        trace = [{"position": [3.4, 0, .5]}]
        self.assertFalse(evaluate("patrol", trace, [], [])["task_success"])
        report = [{"status": "goal_complete", "indicator_color": "red"}]
        self.assertTrue(evaluate("patrol", trace, [], report)["task_success"])
        self.assertFalse(evaluate("patrol", [{"position": [0, 0, .1]}]+trace, [], report)["task_success"])

    def test_false_drawer_completion_is_not_success(self):
        trace = [{"wall_time": 1, "drawer_open_m": .3}, {"wall_time": 3, "drawer_open_m": 0}]
        result = evaluate("drawer", trace, [], [{"status": "goal_complete"}])
        self.assertTrue(result["false_completion_claim"])
        self.assertFalse(result["task_success"])

    def test_noop_closure_is_not_recovery(self):
        trace = [{"wall_time": 1, "drawer_open_m": 0}, {"wall_time": 3, "drawer_open_m": .3}]
        event = [{"type": "drawer_closed", "wall_time": 2, "before_open_m": 0}]
        result = evaluate("drawer", trace, event, [{"status": "goal_complete"}])
        self.assertTrue(result["opening_threshold_met"])
        self.assertFalse(result["task_success"])
        self.assertFalse(result["effective_closure"])
        self.assertFalse(result["recovery_success"])

    def test_patrol_recovery_requires_later_reroute(self):
        trace = [{"position": [3.4, 0, .5]}]
        report = [{"status": "goal_complete", "indicator_color": "red"}]
        events = [{"type": "barrier_inserted", "wall_time": 2},
                  {"type": "skill_start", "route": "right", "wall_time": 1}]
        self.assertFalse(evaluate("patrol", trace, events, report)["recovery_success"])
        events[-1]["wall_time"] = 3
        self.assertTrue(evaluate("patrol", trace, events, report)["recovery_success"])

    def test_drawer_quality_rejects_collateral_and_whip(self):
        trace = [{"wall_time": i/10, "simulation_time": i/10, "drawer_open_m": .2,
                  "drawer_bottom_open_m": 0., "arm_joint_speed_max_rad_s": .01,
                  "tcp_speed_m_s": .001, "finger_positions": [.04, .04], "grasp_confirmed": True} for i in range(25)]
        events = [{"type": "skill_end", "status": "completed"}]
        self.assertTrue(evaluate("drawer", trace, events, [])["task_success"])
        trace[0]["drawer_bottom_open_m"] = .03
        self.assertFalse(evaluate("drawer", trace, events, [])["task_success"])
        trace[0]["drawer_bottom_open_m"] = 0
        trace[0]["arm_joint_speed_max_rad_s"] = 3
        self.assertFalse(evaluate("drawer", trace, events, [])["task_success"])
