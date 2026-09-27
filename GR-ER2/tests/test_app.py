import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image
from app import overlay, response_text, redact, dispatch


class AppTests(unittest.TestCase):
    def test_official_response_text(self):
        self.assertEqual(response_text(SimpleNamespace(output_text="ok")), "ok")

    def test_overlay_point_box_path(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "in.png", Path(directory) / "out.png"
            Image.new("RGB", (640, 480), "black").save(source)
            overlay(source, [{"point": [500, 500], "label": "point"}, {"box": [0, 0, 100, 100]},
                             {"points": [[0, 0], [1000, 1000]]}], target)
            with Image.open(target) as im:
                self.assertEqual(im.size, (640, 480))
            with self.assertRaises(ValueError):
                overlay(source, [{"box": [900, 900, 100, 100]}], target)

    def test_no_key_in_errors(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-private-value"}):
            self.assertEqual(redact("error test-private-value"), "error [REDACTED]")

    def test_whitelist(self):
        with self.assertRaises(ValueError):
            dispatch("execute_shell", {})


if __name__ == "__main__":
    unittest.main()
