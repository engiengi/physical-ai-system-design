import re
import subprocess
import unittest
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_fresh_reader_setup_and_portable_paths(self):
        text = (ROOT / "실행방법.md").read_text()
        for word in ("Spark", "DGX", "/home/geunpil", "Tech_Contents_ws"):
            self.assertNotIn(word, text)
        for phrase in ("Ubuntu 24.04", "python3.12-venv", "gh auth login",
                       "physical-ai-system-design.git", "nvidia-ctk runtime configure",
                       "docker pull nvcr.io/nvidia/isaac-sim:6.0.0-dev2",
                       "Restrict to Gemini API only", "set-key", "check-models",
                       "--probe", "로그아웃", "GR_ER2_READY", "data/uploads/"):
            self.assertIn(phrase, text)
        for anchor in ("host-os", "clone", "python-setup", "gpu-driver", "docker-setup",
                       "gpu-container", "isaac-install", "api-project", "api-first"):
            self.assertIn(f'](#{anchor})', text)
            self.assertIn(f'<a id="{anchor}"></a>', text)

    def test_all_code_blocks_have_execution_locations(self):
        text = (ROOT / "실행방법.md").read_text()
        for match in re.finditer(r'^```([^\n]*)\n(.*?)^```[ \t]*$', text, re.M | re.S):
            label = text[:match.start()].rstrip().splitlines()[-1]
            with self.subTest(line=text[:match.start()].count('\n') + 1):
                self.assertTrue(label.startswith('**') and label.endswith('**'), label)
                if match[1] in ('bash', 'powershell'):
                    self.assertTrue(label.startswith('**실행 위치:'), label)
                else:
                    self.assertRegex(label, r'실행 아님|입력하지 않음')

    def test_reader_file_links(self):
        for name in ('README.md', '실행방법.md'):
            path = ROOT / name
            for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                if '://' in link:
                    continue
                file, _, anchor = unquote(link).partition('#')
                target = path.parent / file if file else path
                self.assertTrue(target.is_file(), link)
                if anchor:
                    self.assertIn(f'id="{anchor}"', target.read_text(), link)

    def test_numbered_toc_targets(self):
        text = (ROOT / "실행방법.md").read_text()
        toc = text.split('<a id="step-0"></a>', 1)[0]
        targets = re.findall(r'\]\(#(step-\d+)\)', toc)
        self.assertEqual(len(targets), 11)
        for target in targets:
            self.assertIn(f'<a id="{target}"></a>', text)

    def test_entrypoints_exist(self):
        for entry in ("scripts/setup.sh", "scripts/sim.sh", "scripts/web.sh", "src/app.py", "src/simulation.py", "src/index.html"):
            self.assertTrue((ROOT / entry).is_file())

    def test_gui_routes(self):
        text = (ROOT / "실행방법.md").read_text()
        for target in ("gui-choice", "gui-remote", "gui-local", "gui-stop"):
            self.assertIn(f"](#{target})", text)
            self.assertIn(f'<a id="{target}"></a>', text)
        self.assertIn('bash scripts/sim.sh stream "$SERVER_IP"', text)
        self.assertIn("bash scripts/sim.sh gui", text)
        self.assertNotIn("bash scripts/sim.sh stream 127.0.0.1", text)
        self.assertNotIn("linux-aarch64.deb", text)

    def test_secret_ignored(self):
        self.assertIn(".env", (ROOT / ".gitignore").read_text().splitlines())

    def test_brev_route_has_complete_lifecycle(self):
        text = (ROOT / "실행방법.md").read_text()
        for anchor in ('brev-setup', 'brev-install', 'brev-gui', 'brev-scenes', 'brev-cleanup'):
            self.assertIn(f'<a id="{anchor}"></a>', text)
        for command in ('brev login', 'brev create gr-er2-lab', 'scripts/setup_brev.sh',
                        'scripts/cloud_gui.sh start', 'scripts/cloud_gui.sh stop',
                        '127.0.0.1:6080:127.0.0.1:6080', 'rsync -av', 'brev delete'):
            self.assertIn(command, text)
        self.assertIn('중지·재시작을 지원하지 않', text)
        self.assertIn('Python 3.12.11', text)

    def test_private_writing_notes_ignored(self):
        rules = (ROOT / ".gitignore").read_text().splitlines()
        for rule in ("/연구일지.md", "/원고기획.md", "outputs/"):
            self.assertIn(rule, rules)

    def test_current_reader_route_and_result_scope(self):
        text = (ROOT / "실행방법.md").read_text()
        anchors = re.findall(r'<a id="([^"]+)"></a>', text)
        self.assertEqual(len(anchors), len(set(anchors)))
        for target in re.findall(r'\]\(#([^)]*)\)', text):
            self.assertIn(target, anchors)
        for target in ("lesson-order", "existing-install", "read-results", "representative-runs", "copy-results"):
            self.assertIn(target, anchors)
        for instruction in ("git clone", "git pull --ff-only origin main", "GR_ER2_SCENE=blocks",
                            "drawer --smoke --no-perturb", "motion_quality_passed", "tool_result",
                            "outputs/", "영상과 실행 결과는 저장소에 포함되어 있지 않습니다"):
            self.assertIn(instruction, text)
        self.assertEqual(text.count("``` "), 0)
        self.assertEqual(sum(line.startswith("```") for line in text.splitlines()) % 2, 0)

    def test_recovery_routes(self):
        text = (ROOT / "실행방법.md").read_text()
        for target in ("recovery-experiment", "live-recovery-experiment", "drawer-scenario", "patrol-scenario"):
            self.assertIn(f"](#{target})", text)
            self.assertIn(f'<a id="{target}"></a>', text)
        self.assertIn("src/live_recovery.py", text)
        self.assertIn("src/recovery.py", text)

    def test_manual_shell_block_syntax(self):
        text = (ROOT / "실행방법.md").read_text()
        blocks = re.findall(r"```bash\n(.*?)\n```", text, re.S)
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks, 1):
            with self.subTest(block=index):
                result = subprocess.run(["bash", "-n"], input=block, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
