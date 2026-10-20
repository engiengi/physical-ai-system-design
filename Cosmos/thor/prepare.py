"""Download pinned policy weights and a small real DROID evaluation shard."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_REV = 'a7c7288f9b6ac1684e993007b0f9703dd26e58ef'
DATA_REV = '5c11a20accb11497270a5247a7f1e66ad04c956c'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--part', choices=['model', 'edge', 'data'], required=True)
    a = p.parse_args()
    if a.part == 'model':
        repo, rev, kind = 'nvidia/Cosmos3-Edge-Policy-DROID', MODEL_REV, 'model'
        dest = a.workspace / 'models/Cosmos3-Edge-Policy-DROID'
        patterns = ['*.json', '*.jinja', '*.md', '*.safetensors']
    elif a.part == 'edge':
        repo, rev, kind = 'nvidia/Cosmos3-Edge', 'a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba', 'model'
        dest = a.workspace / 'models/Cosmos3-Edge'
        patterns = ['*.json', '*.jinja', '*.md', '*.safetensors', '*.txt', '*.model']
    else:
        repo, rev, kind = 'nvidia/Cosmos3-DROID', DATA_REV, 'dataset'
        dest = a.workspace / 'data/droid'
        patterns = ['README.md', 'success/meta/info.json', 'success/meta/tasks.parquet',
                    'success/meta/episodes/chunk-000/file-000.parquet',
                    'success/data/chunk-000/file-000.parquet',
                    'success/videos/*/chunk-000/file-000.mp4']
    snapshot_download(repo, repo_type=kind, revision=rev, local_dir=dest,
                      allow_patterns=patterns, max_workers=4)
    (dest / 'download.json').write_text(json.dumps({'repository': repo, 'revision': rev,
        'repo_type': kind, 'allow_patterns': patterns}, indent=2) + '\n')
    print(dest, flush=True)

if __name__ == '__main__':
    main()
