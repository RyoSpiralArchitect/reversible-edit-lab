"""CUDA storage/FP32-update probe using saved real edit factors, not live editing."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from reversible_edit.roundoff import make_roundoff_patch, apply_roundoff_patch, RoundoffPatch


def bits(t):return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
def array(t):return t.detach().float().cpu().numpy()
def sync():torch.cuda.synchronize()

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Refusing overwrite')
    torch.set_num_threads(1);torch.manual_seed(7);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    files=sorted(f for f in a.inputs.glob('*-A-transformer*.npz') if not f.name.startswith('._'))
    if len(files)!=4:raise ValueError('Four fixtures required')
    result={'scope':'Actual CUDA native storage plus FP32 factor product; saved CPU-author factors. No live optimizer, forward pass, mixed-precision factual efficacy or distributed claim.', 'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'platform':platform.platform(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'rows':[]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for f in files:
        with np.load(f,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        for precision,dtype in [('float32',torch.float32),('float16',torch.float16),('bfloat16',torch.bfloat16)]:
            original=torch.tensor(arrays['original'],device='cuda',dtype=dtype)
            left=torch.tensor(arrays['left'],device='cuda',dtype=torch.float32)
            right=torch.tensor(arrays['right'],device='cuda',dtype=torch.float32)
            update=left@right.T
            edited=(original.float()+update).to(dtype)
            predicted=(edited.float()-left@right.T).to(dtype)
            # CPU correction format operates on exact logical storage bits, including BF16.
            sync();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            patch=make_roundoff_patch(array(original),array(predicted),precision)
            blob=patch.to_bytes();sync();build=time.perf_counter()-start
            start=time.perf_counter()
            fresh_prediction=(edited.float()-left@right.T).to(dtype)
            restored=apply_roundoff_patch(array(fresh_prediction),RoundoffPatch.from_bytes(blob))
            device_restored=torch.tensor(restored,device='cuda',dtype=dtype);sync();undo=time.perf_counter()-start
            wrong=fresh_prediction.clone();wrong.flatten()[0]=torch.nextafter(wrong.flatten()[0],torch.tensor(float('inf'),device='cuda',dtype=dtype))
            try:apply_roundoff_patch(array(wrong),patch)
            except ValueError:rejected=True
            else:rejected=False
            row={'fixture':f.name,'precision':precision,'exact':bits(device_restored)==bits(original),'ordinary_subtract_exact':bits(predicted)==bits(original),'wrong_predictor_rejected':rejected,'build_seconds':build,'restore_seconds':undo,'patch_bytes':len(blob),'factor_bytes':left.numel()*4+right.numel()*4,'stored_weight_bytes':original.numel()*original.element_size(),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
            result['rows'].append(row);a.out.write_text(json.dumps(result,indent=2)+'\n');print(row,flush=True)
    result['complete']=True;result['passed']=all(r['exact'] and r['wrong_predictor_rejected'] for r in result['rows']);a.out.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
