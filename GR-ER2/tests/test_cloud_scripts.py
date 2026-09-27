"""No cloud creation or Docker execution; test preflight and restart contracts."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CloudScriptTests(unittest.TestCase):
    def test_shell_syntax(self):
        for name in ('setup_brev.sh', 'cloud_gui.sh', 'setup.sh'):
            subprocess.run(['bash', '-n', str(ROOT / 'scripts' / name)], check=True)

    def test_localhost_only_and_pinned_python(self):
        gui = (ROOT / 'scripts/cloud_gui.sh').read_text()
        self.assertIn('127.0.0.1:6080 127.0.0.1:5901', gui)
        self.assertIn('-rfbport 5901 -localhost', gui)
        self.assertIn('-nolisten tcp -auth', gui)
        self.assertNotIn('0.0.0.0', gui)
        self.assertIn("'#{pane_pid}'", gui)
        self.assertIn('kill -TERM "$pid"', gui)
        self.assertIn('ps -o uid= -p "$pid"', gui)
        self.assertIn('tmux set-option -t "$name" @gr_er2_root "$ROOT"', gui)
        setup = (ROOT / 'scripts/setup_brev.sh').read_text()
        self.assertIn('uv==0.8.22', setup)
        self.assertIn('python install 3.12.11', setup)
        self.assertIn('GR_ACCEPT_ISAAC_EULA', setup)

    def test_stop_refuses_unowned_desktop(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / 'tmux'
            fake.write_text('#!/bin/bash\nif [[ "$1" == has-session ]]; then exit 0; fi\n'
                            'if [[ "$1" == show-option ]]; then echo /another/project; exit 0; fi\n'
                            'echo unexpected-mutation >&2; exit 99\n')
            fake.chmod(0o755)
            result = subprocess.run(['bash', str(ROOT / 'scripts/cloud_gui.sh'), 'stop'],
                                    env={**os.environ, 'PATH': directory + ':' + os.environ['PATH']},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('소유권', result.stderr)
            self.assertNotIn('unexpected-mutation', result.stderr)

    def test_writable_container_log_does_not_need_chmod(self):
        # A different uid may own the log while our group has write access.
        # Rejecting chmod for this file reproduces the cloud restart failure.
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / 'scripts').mkdir()
            (project / 'runtime').mkdir()
            (project / 'runtime/isaac.log').write_text('keep previous log\n')
            shutil.copy2(ROOT / 'scripts/sim.sh', project / 'scripts/sim.sh')
            bin_dir = project / 'bin'
            bin_dir.mkdir()
            (bin_dir / 'docker').write_text('''#!/bin/bash
if [[ "$1 $2" == "container inspect" ]]; then exit 0; fi
if [[ "$1" == inspect && "$3" == *State.Running* ]]; then echo false; exit 0; fi
if [[ "$1" == inspect && "$3" == *Mounts* ]]; then echo "$TEST_PROJECT"; exit 0; fi
if [[ "$1" == start ]]; then exit 0; fi
if [[ "$1" == exec && "$2" == -d ]]; then exit 0; fi
if [[ "$1" == exec ]]; then exit 1; fi
exit 99
''')
            (bin_dir / 'chmod').write_text('''#!/bin/bash
for arg in "$@"; do
  if [[ "$arg" == */isaac.log ]]; then echo forbidden-log-chmod >&2; exit 99; fi
done
exit 0
''')
            for path in bin_dir.iterdir():
                path.chmod(0o755)
            result = subprocess.run(['bash', str(project / 'scripts/sim.sh'), 'start'],
                                    env={**os.environ, 'PATH': str(bin_dir) + ':' + os.environ['PATH'],
                                         'TEST_PROJECT': str(project), 'GR_ER2_SCENE': 'blocks'},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn('forbidden-log-chmod', result.stderr)


if __name__ == '__main__':
    unittest.main()
