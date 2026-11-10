"""Frozen real-edit codec comparison; isolated worker per observation."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
import zlib
import numpy as np
from reversible_edit.codecs import compress, decompress
from reversible_edit.roundoff import make_roundoff_patch, apply_roundoff_patch, RoundoffPatch
from reversible_edit.receipts import tensor_digest

VARIANTS = [('none',None),('zlib',1),('zlib',6),('zlib',9),('zstd',1),('zstd',3),('lz4',0)]
METHODS = ['checkpoint','direct_xor','factors_correction']

def digest(b):return hashlib.sha256(b).hexdigest()
def pack(a, method, codec, level):
    original,edited=a['original'],a['edited']
    metadata={'version':1,'method':method,'edited':tensor_digest(edited),'original':tensor_digest(original)}
    values={'metadata':np.array(json.dumps(metadata))}
    if method=='checkpoint':values['weight']=original
    elif method=='direct_xor':values['xor']=original.view(np.uint32)^edited.view(np.uint32)
    else:
        values.update(left=a['left'],right=a['right'])
        prediction=(edited.astype(float)-a['left'].astype(float)@a['right'].astype(float).T).astype(np.float32)
        values['patch']=np.frombuffer(make_roundoff_patch(original,prediction,'float32',strategy='xor_none').to_bytes(),dtype=np.uint8)
    stream=io.BytesIO();np.savez(stream,**values);raw=stream.getvalue();payload=compress(raw,codec,level)
    header=json.dumps({'version':1,'codec':codec,'level':level,'raw_size':len(raw),'sha256':digest(payload)},sort_keys=True).encode()
    return len(header).to_bytes(4,'little')+header+payload

def unpack(edited, packet):
    size=int.from_bytes(packet[:4],'little')
    if size>65536 or len(packet)<4+size:raise ValueError('Invalid header')
    header=json.loads(packet[4:4+size]);payload=packet[4+size:]
    if header['version']!=1 or digest(payload)!=header['sha256']:raise ValueError('Invalid packet')
    raw=decompress(payload,header['codec'],header['raw_size'])
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
        meta=json.loads(str(f['metadata'].item()))
        if tensor_digest(edited)!=meta['edited']:raise ValueError('Wrong edited state')
        if meta['method']=='checkpoint':result=f['weight'].copy()
        elif meta['method']=='direct_xor':result=(edited.view(np.uint32)^f['xor']).view(np.float32)
        elif meta['method']=='factors_correction':
            prediction=(edited.astype(float)-f['left'].astype(float)@f['right'].astype(float).T).astype(np.float32)
            result=apply_roundoff_patch(prediction,RoundoffPatch.from_bytes(f['patch'].tobytes()))
        else:raise ValueError('Unknown method')
    if tensor_digest(result)!=meta['original']:raise ValueError('Wrong restored state')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--worker');a=p.parse_args()
    if a.worker:
        trial=json.loads(a.worker)
        with np.load(trial['path'],allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        baseline=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start=time.perf_counter();packet=pack(arrays,trial['method'],trial['codec'],trial['level']);build=time.perf_counter()-start
        start=time.perf_counter();restored=unpack(arrays['edited'],packet);undo=time.perf_counter()-start
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        exact=restored.tobytes()==arrays['original'].tobytes()
        rejected=[]
        wrong=arrays['edited'].copy();wrong.flat[0]=np.nextafter(wrong.flat[0],np.float32(np.inf))
        for state,blob in [(wrong,packet),(arrays['edited'],packet[:-1]),(arrays['edited'],packet[:-1]+bytes([packet[-1]^1]))]:
            try:unpack(state,blob)
            except ValueError:rejected.append(True)
            else:rejected.append(False)
        scale=1 if sys.platform=='darwin' else 1024
        print(json.dumps(dict(trial,bytes=len(packet),build_seconds=build,restore_seconds=undo,exact=exact,rejected=rejected,baseline_rss_bytes=baseline*scale,peak_rss_bytes=peak*scale)))
        return 0 if exact and all(rejected) else 1
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'observations.jsonl').exists():raise ValueError('Refusing to overwrite run')
    files=sorted(Path('results/20261110_factual_chain/work').glob('*-A-transformer*.npz'))
    if len(files)!=4:raise ValueError('Expected four frozen fixtures')
    import zstandard,lz4
    manifest={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'sources':{str(f):digest(f.read_bytes()) for f in [Path(__file__),Path('src/reversible_edit/codecs.py'),Path('src/reversible_edit/roundoff.py')]},'inputs':{str(f):digest(f.read_bytes()) for f in files},'platform':platform.platform(),'python':sys.version,'versions':{'numpy':np.__version__,'zlib':zlib.ZLIB_RUNTIME_VERSION,'zstd':zstandard.__version__,'lz4':lz4.__version__},'plan_sha256':digest((out/'PLAN.md').read_bytes())}
    (out/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    trials=[dict(path=str(f),method=m,codec=c,level=l,repeat=r) for f in files for m in METHODS for c,l in VARIANTS for r in range(3)]
    random.Random(20261110).shuffle(trials)
    rows=[]
    with (out/'observations.jsonl').open('x') as stream:
        for i,t in enumerate(trials):
            proc=subprocess.run([sys.executable,'-I',__file__,'--worker',json.dumps(t)],capture_output=True,text=True)
            if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
            row=json.loads(proc.stdout);rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if (i+1)%21==0:print(i+1,'/',len(trials),flush=True)
    (out/'summary.json').write_text(json.dumps({'observations':len(rows),'exact':sum(r['exact'] for r in rows),'negative_checks':sum(sum(r['rejected']) for r in rows)},indent=2)+'\n')
if __name__=='__main__':raise SystemExit(main())
