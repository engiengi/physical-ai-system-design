import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recovery import summarize


def record(ok, pick, candidate="red", red=None):
    return {"action": {"pick": pick}, "metrics": {"placement_within_tolerance": ok,
            "picked_candidate": candidate, "after": {"red": red or [0.49, 0.24, 0.035]}}}


class RecoveryTests(unittest.TestCase):
    def test_failure_then_recovery(self):
        result = summarize([record(False, [500, 600]), record(True, [400, 700])], [{"type": "shift"}])
        self.assertTrue(result["recovery_observed"])
        self.assertTrue(result["changed_pick_after_first_action"])

    def test_baseline_not_recovery(self):
        self.assertFalse(summarize([record(True, [500, 600])], [])["recovery_observed"])

    def test_wrong_object_not_recovery(self):
        self.assertFalse(summarize([record(False, [500, 600]), record(True, [400, 700], "blue")], [{}])["recovery_observed"])

    def test_red_outside_tray_not_recovery(self):
        self.assertFalse(summarize([record(False, [500, 600]), record(True, [400, 700], red=[0.6, -0.27, 0.025])], [{}])["recovery_observed"])

    def test_no_actions(self):
        self.assertFalse(summarize([], [])["recovery_observed"])


if __name__ == "__main__":
    unittest.main()
