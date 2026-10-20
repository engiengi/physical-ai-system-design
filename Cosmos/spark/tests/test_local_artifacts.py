"""The single-machine cloud path must preserve artifact checks and boundaries."""
import hashlib
import pytest
from cosmos_spark import remote


def test_local_copy_checksum_and_existing_output(tmp_path, monkeypatch):
    workspace = tmp_path / 'inference'
    workspace.mkdir()
    source = workspace / 'action.npy'
    source.write_bytes(b'example saved action')
    target = tmp_path / 'received.npy'
    monkeypatch.setattr(remote, 'WORKSPACE', str(workspace))
    monkeypatch.setenv('COSMOS_ARTIFACT_TRANSPORT', 'local')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert remote.fetch_file(str(source), target, digest) == digest
    assert source.read_bytes() == target.read_bytes()
    with pytest.raises(FileExistsError):
        remote.fetch_file(str(source), target, digest)
    rejected = tmp_path / 'rejected.npy'
    with pytest.raises(ValueError, match='checksum'):
        remote.fetch_file(str(source), rejected, '0' * 64)
    assert not rejected.exists() and not rejected.with_suffix('.npy.partial').exists()


def test_local_transfer_rejects_symlink_escape(tmp_path, monkeypatch):
    workspace = tmp_path / 'inference'
    workspace.mkdir()
    outside = tmp_path / 'outside'
    outside.write_bytes(b'outside the configured workspace')
    source = workspace / 'link'
    source.symlink_to(outside)
    monkeypatch.setattr(remote, 'WORKSPACE', str(workspace))
    monkeypatch.setenv('COSMOS_ARTIFACT_TRANSPORT', 'local')
    with pytest.raises(ValueError, match='escapes'):
        remote.fetch_file(str(source), tmp_path / 'copy')
