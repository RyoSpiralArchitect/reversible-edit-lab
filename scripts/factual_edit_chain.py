"""Pinned-author ROME/MEMIT factual edit A -> undo -> freshly optimized edit B."""
import argparse
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import types
import numpy as np
from reversible_edit.receipts import tensor_digest
from reversible_edit.roundoff import make_roundoff_patch, apply_roundoff_patch, RoundoffPatch
from reversible_edit.cost_experiments import _rss_bytes, THREAD_ENV

CALIBRATION = [f'The {noun} was {state} near the {place}.' for noun,state in [('book','open'),('lamp','bright'),('window','closed'),('chair','empty')] for place in ['table','garden','door','wall']]
CONTEXTS = ['{}','Today, {}','In a report, {}']
A = [dict(prompt='The capital of {} is',subject='France',target_new={'str':' Rome'}),dict(prompt='The capital of {} is',subject='Japan',target_new={'str':' Paris'})]
B = [dict(prompt='The capital of {} is',subject='Italy',target_new={'str':' Tokyo'})]
PROBES = [('rewrite_A','The capital of France is',' Rome'),('paraphrase_A','France has its capital in',' Rome'),('rewrite_A2','The capital of Japan is',' Paris'),('paraphrase_A2','Japan has its capital in',' Paris'),('rewrite_B','The capital of Italy is',' Tokyo'),('paraphrase_B','Italy has its capital in',' Tokyo'),('locality','The capital of Germany is',' Berlin'),('unrelated','The sky is',' blue')]

def digest_file(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def pack_record(original, edited, left, right, method):
    meta={'method':method,'edited_digest':tensor_digest(edited),'original_digest':tensor_digest(original)}
    payload={'meta':np.array(json.dumps(meta))}
    if method=='checkpoint': payload['weight']=original
    else:
        payload.update(left=left,right=right)
        predicted=(edited.astype(np.float64)-left.astype(np.float64)@right.astype(np.float64).T).astype(np.float32)
        if method=='delta_exact':payload['patch']=np.frombuffer(make_roundoff_patch(original,predicted,'float32').to_bytes(),dtype=np.uint8)
    stream=io.BytesIO();np.savez_compressed(stream,**payload);return stream.getvalue()

def restore(edited, payload):
    meta=json.loads(str(payload['meta'].item()))
    if tensor_digest(edited)!=meta['edited_digest']:raise ValueError('edited state mismatch')
    if meta['method']=='checkpoint':result=payload['weight'].copy()
    else:
        result=(edited.astype(float)-payload['left'].astype(float)@payload['right'].astype(float).T).astype(np.float32)
        if meta['method']=='delta_exact':result=apply_roundoff_patch(result,RoundoffPatch.from_bytes(payload['patch'].tobytes()))
    if meta['method']!='delta' and tensor_digest(result)!=meta['original_digest']:raise ValueError('restoration mismatch')
    return result

def arrays(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path);p.add_argument('--upstream',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--undo',action='store_true');p.add_argument('--edited',type=Path);p.add_argument('--record',type=Path)
    a=p.parse_args()
    if a.undo:
        edited=arrays(a.edited)['weight'];blob=a.record.read_bytes();before=_rss_bytes();start=time.perf_counter()
        payload=arrays(io.BytesIO(blob));result=restore(edited,payload);seconds=time.perf_counter()-start;peak=_rss_bytes()
        np.savez(a.out,weight=result)
        print(json.dumps({'seconds':seconds,'process_peak_rss_bytes':peak,'baseline_peak_rss_bytes':before,'digest':tensor_digest(result)}));return 0
    if not all([a.model_dir,a.upstream,a.out]) or a.out.exists():p.error('model-dir, upstream, and new out directory required')
    if digest_file(a.model_dir/'model.safetensors')!='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707':raise ValueError('model SHA mismatch')
    import os
    import torch
    import transformers
    from transformers import GPT2LMHeadModel, AutoTokenizer
    torch.set_num_threads(1);torch.manual_seed(7);torch.use_deterministic_algorithms(True)
    out=a.out.resolve();out.mkdir(parents=True);work=out/'work';work.mkdir()
    upstream=a.upstream.resolve();sys.path.insert(0,str(upstream))
    # The large-dataset loader is replaced by the explicit frozen calibration below.
    stats=types.ModuleType('rome.layer_stats')
    def disabled_stats(*args,**kwargs):raise RuntimeError('use declared frozen local covariance')
    stats.layer_stats=disabled_stats;sys.modules['rome.layer_stats']=stats
    rm=importlib.import_module('rome.rome_main');cu=importlib.import_module('rome.compute_u');mm=importlib.import_module('memit.memit_main')
    rm.CONTEXT_TEMPLATES_CACHE=CONTEXTS;mm.CONTEXT_TEMPLATES_CACHE=[CONTEXTS]
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    model=GPT2LMHeadModel.from_pretrained(a.model_dir,local_files_only=True,use_safetensors=True,attn_implementation='eager').eval();model.config.use_cache=False
    for parameter in model.parameters():parameter.requires_grad_(False)
    names=[f'transformer.h.{l}.mlp.c_proj.weight' for l in [4,5,6]]
    params=dict(model.named_parameters());base={n:params[n].detach().clone() for n in names}
    # Empirical uncentered second moments at real c_proj inputs, frozen on base.
    cov={};counts={}
    for layer in [4,5,6]:
        chunks=[]
        def collect(module,args,output):chunks.append(args[0].detach().reshape(-1,3072).double())
        handle=model.transformer.h[layer].mlp.c_proj.register_forward_hook(collect)
        with torch.no_grad():
            for sentence in CALIBRATION:model(**tok(sentence,return_tensors='pt'),use_cache=False)
        handle.remove();x=torch.cat(chunks);c=x.T@x/x.shape[0];ridge=.01*float(c.diag().mean());c+=ridge*torch.eye(3072,dtype=torch.float64)
        name=f'transformer.h.{layer}.mlp.c_proj';cov[name]=c;counts[name]={'tokens':len(x),'ridge':ridge}
        np.savez(work/f'covariance-{layer}.npz',covariance=c.numpy())
    inverse={name:torch.linalg.inv(c).float() for name,c in cov.items()}
    cu.get_inv_cov=lambda model,tok,name,*args:inverse[name]
    mm.get_cov=lambda model,tok,name,*args,**kwargs:cov[name]
    common=dict(fact_token='subject_last',v_num_grad_steps=20,v_lr=.5,v_loss_layer=11,v_weight_decay=.5,kl_factor=.0625,mom2_adjustment=True,rewrite_module_tmp='transformer.h.{}.mlp.c_proj',layer_module_tmp='transformer.h.{}',mlp_module_tmp='transformer.h.{}.mlp',attn_module_tmp='transformer.h.{}.attn',ln_f_module='transformer.ln_f',lm_head_module='transformer.wte',mom2_dataset='declared_local_16_sentences',mom2_n_samples=16,mom2_dtype='float64')
    hp={'rome':rm.ROMEHyperParams(**common,layers=[5],clamp_norm_factor=4,context_template_length_params=[]),'memit':mm.MEMITHyperParams(**common,layers=[4,5,6],clamp_norm_factor=.75,layer_selection='all',mom2_update_weight=1000)}
    manifest={'model':'openai-community/gpt2','model_revision':'607a30d783dfa663caf39e06633721c8d4cfcd7e','model_sha256':digest_file(a.model_dir/'model.safetensors'),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'script_sha256':digest_file(__file__),'upstream_commit':'80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b','upstream_files':{str(f.relative_to(upstream)):digest_file(f) for f in sorted(upstream.rglob('*.py'))},'calibration':CALIBRATION,'covariance':counts,'contexts':CONTEXTS,'hparams':{k:vars(v) for k,v in hp.items()},'A':{'rome':A[:1],'memit':A},'B':B,'probes':PROBES,'scope':'CPU FP32 authors algorithms with explicit small-model/corpus adaptation; no original-paper reproduction claim'}
    (out/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def reset():
        with torch.no_grad():
            for n in names:params[n].copy_(base[n])
    def state():
        h=hashlib.sha256()
        for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().numpy().tobytes())
        return h.hexdigest()
    def evaluate(label):
        logits=[];metrics=[]
        with torch.no_grad():
            for kind,text,target in PROBES:
                ids=tok.encode(target,add_special_tokens=False)
                if len(ids)!=1:raise ValueError('single target token required')
                logit=model(**tok(text,return_tensors='pt'),use_cache=False).logits[0,-1].detach().clone()
                logits.append(logit.numpy());metrics.append({'kind':kind,'prompt':text,'target':target,'target_probability':float(logit.softmax(-1)[ids[0]]),'target_rank':int((logit>logit[ids[0]]).sum())+1,'top1':tok.decode([int(logit.argmax())])})
        matrix=np.stack(logits);np.savez(work/(label+'-logits.npz'),logits=matrix)
        return {'state_digest':state(),'logits_digest':tensor_digest(matrix),'metrics':metrics},matrix
    def edit(algorithm,requests,label):
        before={n:params[n].detach().clone() for n in names};torch.manual_seed(7)
        with (out/(label+'-optimization.txt')).open('w') as log,contextlib.redirect_stdout(log):
            deltas=rm.execute_rome(model,tok,requests[0],hp[algorithm]) if algorithm=='rome' else mm.execute_memit(model,tok,requests,hp[algorithm])
        for n in names:
            if not torch.equal(before[n],params[n]):raise ValueError('upstream execute failed restoration invariant')
        records={}
        with torch.no_grad():
            for n,(key,value) in deltas.items():
                if algorithm=='rome':key,value=key[:,None],value[:,None]
                # Preserve the authors actual update arithmetic and weight layout.
                update=key@value.T
                params[n].add_(update.float())
                original=before[n].numpy().T.copy();edited=params[n].detach().numpy().T.copy()
                records[n]=(original,edited,value.detach().numpy(),key.detach().numpy())
                np.savez(work/(label+'-'+n+'.npz'),original=original,edited=edited,left=records[n][2],right=records[n][3])
        return records
    baseline,baseline_logits=evaluate('baseline');results=[]
    for algorithm in ['rome','memit']:
        reset();edit(algorithm,B,algorithm+'-reference-B');reference,reference_logits=evaluate(algorithm+'-reference-B')
        reset();records=edit(algorithm,A[:1] if algorithm=='rome' else A,algorithm+'-A');afterA,_=evaluate(algorithm+'-A')
        for method in ['checkpoint','delta','delta_exact']:
            reset()
            with torch.no_grad():
                for n,(_,edited,_,_) in records.items():params[n].copy_(torch.from_numpy(edited.T.copy()))
            costs=[]
            for index,(n,(original,edited,left,right)) in enumerate(reversed(list(records.items()))):
                prefix=work/f'{algorithm}-{method}-{index}';start=time.perf_counter();blob=pack_record(original,edited,left,right,method);build_seconds=time.perf_counter()-start
                record=Path(str(prefix)+'-record.npz');record.write_bytes(blob);edited_path=Path(str(prefix)+'-edited.npz');np.savez(edited_path,weight=edited);restored_path=Path(str(prefix)+'-restored.npz')
                proc=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),'--undo','--edited',str(edited_path),'--record',str(record),'--out',str(restored_path)],capture_output=True,text=True,env=dict(os.environ,**THREAD_ENV),timeout=120)
                if proc.returncode:raise RuntimeError(proc.stderr)
                info=json.loads(proc.stdout);info.update({'layer':n,'record_bytes':len(blob),'record_sha256':hashlib.sha256(blob).hexdigest(),'build_seconds_parent':build_seconds});costs.append(info)
                with torch.no_grad():params[n].copy_(torch.from_numpy(arrays(restored_path)['weight'].T.copy()))
            restored,restored_logits=evaluate(algorithm+'-'+method+'-restored')
            edit(algorithm,B,algorithm+'-'+method+'-B');nextstate,next_logits=evaluate(algorithm+'-'+method+'-B')
            row={'algorithm':algorithm,'method':method,'after_A':afterA,'restored':restored,'next_B':nextstate,'reference_B':reference,'restored_model_bitwise':restored['state_digest']==baseline['state_digest'],'restored_logits_bitwise':np.array_equal(restored_logits,baseline_logits),'next_model_bitwise':nextstate['state_digest']==reference['state_digest'],'next_logits_bitwise':np.array_equal(next_logits,reference_logits),'restored_logits_max_abs_error':float(np.max(np.abs(restored_logits-baseline_logits))),'next_logits_max_abs_error':float(np.max(np.abs(next_logits-reference_logits))),'costs':costs}
            # NumPy booleans are converted before durable output.
            row['restored_logits_bitwise']=bool(row['restored_logits_bitwise']);row['next_logits_bitwise']=bool(row['next_logits_bitwise'])
            results.append(row);(out/'results.json').write_text(json.dumps({'baseline':baseline,'rows':results},indent=2,allow_nan=False)+'\n')
            print(algorithm,method,{k:row[k] for k in ['restored_model_bitwise','next_model_bitwise','next_logits_max_abs_error']},flush=True)
    good=all(all(r[k] for k in ['restored_model_bitwise','restored_logits_bitwise','next_model_bitwise','next_logits_bitwise']) for r in results if r['method']!='delta')
    (out/'summary.json').write_text(json.dumps({'status':'complete' if good else 'failed','comparisons':len(results),'exact_checkpoint_paths_pass':good,'binary_artifacts':{str(f.relative_to(out)):digest_file(f) for f in sorted(work.glob('*.npz'))}},indent=2)+'\n')
    return 0 if good else 1

if __name__=='__main__':raise SystemExit(main())
