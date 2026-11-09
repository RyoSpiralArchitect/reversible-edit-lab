"""Create a standalone toy request, edit it, and restore it using only a receipt.

Run from project root: PYTHONPATH=src python examples/portable_edit.py --out demo
All output paths must be new. The original is used only for the final audit.
"""
import argparse
from pathlib import Path
import numpy as np

from reversible_edit.core import quantize
from reversible_edit.receipts import edit_npz, undo_npz

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(7)
weight=quantize(rng.normal(size=(8,16)),"float32")
key=rng.normal(size=16)
value=rng.normal(size=8)
request=args.out/"request.npz"
if request.exists():
    raise FileExistsError(request)
np.savez(request,weight=weight,key=key,value=value,covariance=np.eye(16))
meta=edit_npz(request,args.out/"edited.npz",args.out/"receipt.npz",
              alpha=0.75,recovery="delta",precision="float32")
undo_npz(args.out/"edited.npz",args.out/"receipt.npz",args.out/"restored.npz")
with np.load(args.out/"restored.npz") as restored:
    error=np.linalg.norm(restored["weight"].astype(float)-weight.astype(float))/np.linalg.norm(weight.astype(float))
print({"relative_error":float(error),"recovery_payload_bytes":meta["numeric_payload_bytes"],
       "single_layer_checkpoint_bytes":weight.nbytes})
