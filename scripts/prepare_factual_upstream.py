"""Create an auditable CPU compatibility copy of pinned authors' code."""
from pathlib import Path
import shutil
import subprocess
import difflib
import json

PIN = '80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b'

def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=PIN:
        raise ValueError('unexpected upstream commit')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('upstream checkout must be clean')
    if destination.exists(): raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for name in ['rome','memit','util']:
        shutil.copytree(source/name,destination/name)
    for name in ['LICENSE','globals.yml']:
        shutil.copy2(source/name,destination/name)
    patches=[]
    for p in sorted(destination.rglob('*.py')):
        before=p.read_text();after=before.replace('"cuda"','"cpu"').replace('.cuda()', '.cpu()')
        if p.name=='repr_tools.py':
            after=after.replace('batch_tok = tok([*prefixes, *words, *suffixes])','batch_tok = tok([*prefixes, *words, *suffixes])["input_ids"]')
        if p.name=='compute_z.py':
            after=after.replace('cur_out[0][', '(cur_out[0] if isinstance(cur_out, tuple) else cur_out)[')
            after=after.replace('tr[hparams.layer_module_tmp.format(loss_layer)].output[0][', '(tr[hparams.layer_module_tmp.format(loss_layer)].output[0] if isinstance(tr[hparams.layer_module_tmp.format(loss_layer)].output, tuple) else tr[hparams.layer_module_tmp.format(loss_layer)].output)[')
        if p.name=='globals.py':
            after=after.replace('open("globals.yml", "r")', 'open(Path(__file__).resolve().parents[1] / "globals.yml", "r")')
        if after!=before:
            relative=str(p.relative_to(destination)); patches.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
            p.write_text(after)
    (destination/'compatibility.patch').write_text(''.join(patches))
    return destination

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args();prepare(a.source,a.destination)
