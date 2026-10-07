import importlib.util
import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('deployment', ROOT / 'ops/lab.py')
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_source_snapshot_excludes_runtime_secrets_and_documents(tmp_path):
    for rel in ('lab', 'config.example.toml', 'spark/pyproject.toml', 'spark/uv.lock', '.gitignore',
                'spark/src/client.py', 'thor/serve.py', 'ops/control.py',
                'config.local.toml', 'thor/.env', 'thor/README.md', 'spark/src/__pycache__/client.pyc',
                'spark/runs/private.json'):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('sample')
    (tmp_path / 'thor/credentials.py').symlink_to(tmp_path / 'config.local.toml')
    files = lab.snapshot(tmp_path)
    assert 'spark/src/client.py' in files and 'thor/serve.py' in files
    assert not any('private' in x or 'credentials' in x or 'README' in x or '.env' in x or '__pycache__' in x or 'local.toml' in x for x in files)
    previous = files['thor/serve.py']
    (tmp_path / 'thor/serve.py').write_text('changed')
    assert lab.snapshot(tmp_path)['thor/serve.py'] != previous


@pytest.mark.parametrize('host,path', [('-oProxyCommand=oops', '/tmp/work'), ('host;touch', '/tmp/work'),
                                      ('thor', '/tmp/work/../other'), ('thor', '/')])
def test_bad_remote_destinations_rejected_before_ssh(tmp_path, host, path):
    p = tmp_path / 'config.toml'
    p.write_text('[thor]\nhost = ' + json.dumps(host) + '\nworkspace = ' + json.dumps(path))
    with pytest.raises(ValueError):
        lab.config(p)


def test_collection_preserves_existing_results(tmp_path, monkeypatch):
    target = tmp_path / 'thor_results/test_run'
    target.mkdir(parents=True)
    (target / 'evidence.json').write_text('original')
    monkeypatch.setattr(lab, 'remote_python', lambda *args: '{}')
    cfg = {'spark': {'workspace': str(tmp_path)}, 'thor': {'workspace': '/tmp/thor'}}
    with pytest.raises(FileExistsError):
        lab.collect(cfg, 'test_run')
    assert (target / 'evidence.json').read_text() == 'original'


def test_collection_refuses_live_run(tmp_path, monkeypatch):
    answers = iter([json.dumps({'run': 'active', 'pid': 123, 'entrypoint': '/tmp/serve.py'}), '1'])
    monkeypatch.setattr(lab, 'remote_python', lambda *args: next(answers))
    cfg = {'spark': {'workspace': str(tmp_path)}, 'thor': {'workspace': '/tmp/thor'}}
    with pytest.raises(RuntimeError, match='Stop this run'):
        lab.collect(cfg, 'active')
    assert not (tmp_path / 'thor_results').exists()
