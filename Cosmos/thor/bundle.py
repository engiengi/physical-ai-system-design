"""Package one saved policy request for transfer; verify every artifact before extraction."""
import argparse
import json
from pathlib import Path
import tarfile
import hashlib
from artifacts import file_info
from common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['create','verify'])
    p.add_argument('--request',type=Path)
    p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='create':
        if not a.request: p.error('--request is required')
        if a.archive.exists() or a.archive.with_suffix('.json').exists(): raise FileExistsError(a.archive)
        info=json.loads((a.request/'artifacts.json').read_text())
        files=[a.request/'artifacts.json',a.request/'result.json']
        for key in ['observation','action','prediction']:
            item=info.get(key)
            if item:
                path=Path(item['path']).resolve()
                if path.parent!=a.request.resolve(): raise ValueError('Artifact must belong to this request directory')
                if file_info(path)['sha256'] != item['sha256']: raise ValueError('Artifact checksum changed')
                files.append(path)
        manifest={path.name:file_info(path) for path in files}
        a.archive.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(a.archive,'x') as archive:
            for path in files: archive.add(path,arcname=path.name,recursive=False)
        write_json(a.archive.with_suffix('.json'),{'archive':file_info(a.archive),'files':manifest,
            'run_id':info['run_id'],'request_id':info['request_id'],
            'note':'Paths inside artifacts.json refer to Thor; resolve member basenames in the local extraction directory.'})
    else:
        index=json.loads(a.archive.with_suffix('.json').read_text())
        if file_info(a.archive)['sha256']!=index['archive']['sha256']: raise ValueError('Archive checksum mismatch')
        with tarfile.open(a.archive) as archive:
            members=archive.getmembers()
            if len(members)!=len(index['files']) or {m.name for m in members}!=set(index['files']): raise ValueError('Archive member mismatch')
            for member in members:
                if not member.isfile() or Path(member.name).name!=member.name: raise ValueError('Unsafe member')
                h=hashlib.sha256()
                with archive.extractfile(member) as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
                if h.hexdigest()!=index['files'][member.name]['sha256']: raise ValueError('File checksum mismatch')
        print(json.dumps({'verified':True,'files':list(index['files'])}))

if __name__=='__main__': main()
