import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import parse_json, pixel, point, safe_child, validate_action, write_json, read_json


class CommonTests(unittest.TestCase):
    def test_coordinate_order(self):
        self.assertEqual(pixel([1000, 0], 640, 480), (0, 479))
        self.assertEqual(pixel([0, 1000], 640, 480), (639, 0))

    def test_bad_coordinates(self):
        for item in ([0, -1], [float("nan"), 2], [True, 1], [1], ["1", 2], [1001, 1]):
            with self.assertRaises(ValueError):
                point(item)

    def test_parse(self):
        self.assertEqual(parse_json('```json\n{"point":[1,2]}\n```'), {"point": [1, 2]})
        with self.assertRaises(ValueError):
            parse_json('not json')

    def test_path(self):
        with self.assertRaises(ValueError):
            safe_child("/tmp/demo", "../secret")

    def test_action(self):
        action = {"pick": [1, 2], "place": [3, 4], "observation_id": "123_abc"}
        self.assertEqual(validate_action(action), action)
        with self.assertRaises(ValueError):
            validate_action({**action, "execute_python": "print(1)"})
        with self.assertRaises(ValueError):
            validate_action({**action, "observation_id": "../escape"})

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            write_json(path, {"한글": 1})
            self.assertEqual(read_json(path), {"한글": 1})


if __name__ == "__main__":
    unittest.main()
