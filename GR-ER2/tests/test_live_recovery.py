import sys
import unittest
import asyncio
import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live_recovery import evaluate_stream, TOOLS


class LiveRecoveryTests(unittest.TestCase):
    def test_probe_saves_input_without_session_credentials(self):
        from PIL import Image
        from live_recovery import probe

        class Session:
            send_realtime_input = AsyncMock()

            async def receive(self):
                yield SimpleNamespace(
                    model_dump=lambda **kwargs: {"session_resumption_update": {"new_handle": "private-handle"}},
                    server_content=None)
                yield SimpleNamespace(
                    model_dump=lambda **kwargs: {"server_content": {"output_transcription": {"text": "Red"}, "turn_complete": True}},
                    server_content=SimpleNamespace(turn_complete=True))

        @contextlib.asynccontextmanager
        async def connect(**kwargs):
            yield Session()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/samples").mkdir(parents=True)
            Image.new("RGB", (4, 4), "red").save(root / "data/samples/scene_01.png")
            out = root / "result"
            out.mkdir()
            fake_client = SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=connect)))
            with patch("live_recovery.ROOT", root), patch("app.run_dir", return_value=out), \
                    patch("app.client", return_value=fake_client), patch("live_recovery.asyncio.sleep", new_callable=AsyncMock), \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                asyncio.run(probe())
            self.assertEqual((out / "input.png").read_bytes(), (root / "data/samples/scene_01.png").read_bytes())
            self.assertEqual(json.loads((out / "request.json").read_text())["frames_sent"], 3)
            self.assertNotIn("private-handle", (out / "messages.json").read_text() + printed.getvalue())
            self.assertEqual(len(json.loads((out / "messages.json").read_text())), 1)

    def evaluate(self, **changes):
        values = {"event": {"wall_time": 10}, "detection": {"wall_time": 12}, "interrupted": True,
                  "first_metrics": {"controller_status": "interrupted_by_stream_monitor"},
                  "recovery_metrics": {"picked_candidate": "red", "placement_within_tolerance": True,
                                       "after": {"red": [0.49, 0.24, 0.035]}}, "model_complete": True}
        values.update(changes)
        return evaluate_stream(**values)

    def test_early_detection_and_recovery(self):
        result = self.evaluate()
        self.assertTrue(result["streaming_recovery_observed"])
        self.assertEqual(result["detection_latency_wall_seconds"], 2)

    def test_late_detection_is_not_early_recovery(self):
        self.assertFalse(self.evaluate(interrupted=False)["streaming_recovery_observed"])

    def test_false_positive_before_perturbation(self):
        self.assertFalse(self.evaluate(detection={"wall_time": 9})["streaming_recovery_observed"])

    def test_no_visual_success(self):
        self.assertFalse(self.evaluate(recovery_metrics={})["streaming_recovery_observed"])

    def test_physical_tools_block(self):
        self.assertTrue(all(t["behavior"] == "BLOCKING" for t in TOOLS[0]["function_declarations"]))

    def test_moved_receptacle_uses_current_bounds(self):
        metrics = {"picked_candidate": "red", "placement_within_tolerance": True,
                   "after": {"red": [0.27, 0.24, 0.035]}, "tray_center": [0.37, 0.235]}
        self.assertTrue(self.evaluate(recovery_metrics=metrics)["recovery_placement_succeeded"])
        metrics["after"]["red"][0] = 0.57
        self.assertFalse(self.evaluate(recovery_metrics=metrics)["recovery_placement_succeeded"])

    def test_already_in_goal_is_separate_from_recovery_effect(self):
        metrics = {"picked_candidate": "red", "placement_within_tolerance": True,
                   "before": {"red": [0.40, 0.22, 0.035]}, "after": {"red": [0.31, 0.24, 0.035]},
                   "tray_center": [0.31, 0.235]}
        self.assertTrue(self.evaluate(recovery_metrics=metrics)["red_already_inside_before_recovery_action"])

    def test_presentation_never_leaks_future(self):
        from presentation import latest_index
        self.assertEqual(latest_index([10, 20, 30], 9), -1)
        self.assertEqual(latest_index([10, 20, 30], 20), 1)
        self.assertEqual(latest_index([10, 20, 30], 29.99), 1)

    def test_unknown_scenario_rejected_before_api(self):
        import asyncio
        from live_recovery import experiment
        with self.assertRaises(ValueError):
            asyncio.run(experiment(scenario="bad"))

    def test_release_request_does_not_prove_drop(self):
        from live_recovery import drop_confirmed
        event = {"type": "grasp_drop", "wall_time": 10, "red_at_release": [0.4, -0.2, 0.13]}
        row = {"wall_time": 11, "red": [0.4, -0.2, 0.03], "fingers": [0.04, 0.04], "tray_center": [0.49, 0.235]}
        self.assertFalse(drop_confirmed([], event))
        self.assertTrue(drop_confirmed([row], event))
        self.assertFalse(drop_confirmed([{**row, "wall_time": 9}], event))
        self.assertFalse(drop_confirmed([{**row, "red": [0.49, 0.235, 0.03]}], event))


if __name__ == "__main__":
    unittest.main()
