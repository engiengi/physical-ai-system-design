"""Non-GPU preflight tests; invalid inputs must fail before any Docker action."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/sim.sh"


class SimScriptTests(unittest.TestCase):
    def test_relocated_checkout_cannot_start_old_mount(self):
        # Fake Docker only; no real container or GPU is started.
        with tempfile.TemporaryDirectory() as directory:
            docker = Path(directory) / 'docker'
            docker.write_text('''#!/bin/bash
if [[ "$1 $2" == "container inspect" ]]; then exit 0; fi
if [[ "$1" == inspect && "$3" == *State.Running* ]]; then echo false; exit 0; fi
if [[ "$1" == inspect && "$3" == *Mounts* ]]; then echo /different/checkout; exit 0; fi
echo unexpected-docker-action >&2
exit 99
''')
            docker.chmod(0o755)
            result = subprocess.run(['bash', str(SCRIPT), 'start'],
                                    env={**os.environ, 'PATH': directory + ':' + os.environ['PATH'],
                                         'GR_ER2_SCENE': 'blocks'},
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('경로 불일치', result.stderr)
            self.assertNotIn('unexpected-docker-action', result.stderr)

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_gui_requires_local_display(self):
        for display in ("", "localhost:10.0", "bad"):
            result = subprocess.run(["bash", str(SCRIPT), "gui"],
                                    env={**os.environ, "DISPLAY": display},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("DISPLAY", result.stdout)

    def test_stream_requires_valid_ipv4(self):
        for args in (("stream",), ("stream", "999.1.1.1"), ("stream", "host-name")):
            result = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
