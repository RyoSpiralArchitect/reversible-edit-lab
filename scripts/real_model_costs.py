"""GPT-2 final-projection gradient update and recorded rollback pilot."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from reversible_edit.core import quantize
from reversible_edit.receipts import tensor_digest, restore_from_receipt
from reversible_edit.cost_experiments import METHODS, _run_worker, _load_arrays, restore_checkpoint, summarize_costs

REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
MODEL_SHA256 = '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
CASES = [('The capital of France is', ' Paris'), ('The capital of Japan is', ' Tokyo'), ('The capital of Italy is', ' Rome')]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gradient_factors(weight, key, output_gradient, relative_step=0.001):
    """Normalized negative SGD direction; transport as an equivalent ROME write."""
    key = np.asarray(key, dtype=np.float64)
    gradient = np.asarray(output_gradient, dtype=np.float64)
    norm = np.linalg.norm(gradient) * np.linalg.norm(key)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('nonzero finite gradient required')
    rate = relative_step * np.linalg.norm(weight.astype(np.float64)) / norm
    direction = key / (key @ key)
    left = -rate * (key @ key) * gradient
    value = weight.astype(np.float64) @ key + left
    return left[:, None], direction[:, None], value, rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.out.exists():
        parser.error('positive repeats and a new output directory required')
    if sha(args.model_dir / 'model.safetensors') != MODEL_SHA256:
        parser.error('model weights do not match the pinned upstream revision')
    import torch
    import transformers
    from transformers import GPT2LMHeadModel, AutoTokenizer
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    out = args.out.resolve(); out.mkdir(parents=True)
    (out / 'work').mkdir()
    root = Path(__file__).resolve().parents[1]
    meta = {'model': 'openai-community/gpt2', 'revision': REVISION,
            'revision_source': 'local files downloaded from pinned Hugging Face resolve URL; hashes below',
            'model_files': {p.name: sha(p) for p in sorted(args.model_dir.iterdir()) if p.is_file()},
            'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'transformers': transformers.__version__, 'platform': platform.platform(),
            'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_sha256': {str(p.relative_to(root)): sha(p) for p in [Path(__file__).resolve(), *sorted((root/'src/reversible_edit').glob('*.py'))]},
            'cases': CASES, 'relative_step': 0.001, 'repeats': args.repeats,
            'precisions': ['float32','float16','bfloat16'], 'projection': 'transformer.h.11.mlp.c_proj',
            'scope': 'Actual final-layer single-token loss gradient; not optimized ROME/MEMIT factual editing. CPU float32 model; projection storage quantized only.'}
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True, use_safetensors=True, attn_implementation='eager').eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    projection = model.transformer.h[-1].mlp.c_proj
    projection.weight.requires_grad_(True)
    native = projection.weight.detach().numpy().T.copy()
    rows, fixtures, behavior = [], [], []
    def apply(weight):
        with torch.no_grad(): projection.weight.copy_(torch.from_numpy(weight.astype(np.float32).T.copy()))
    def logits(inputs):
        with torch.no_grad(): return model(**inputs, use_cache=False).logits[0,-1].numpy().copy()
    with (out/'observations.jsonl').open('x') as stream:
        for precision in meta['precisions']:
            for case_id, (prompt, target) in enumerate(CASES):
                original = quantize(native, precision); apply(original)
                model.zero_grad(set_to_none=True)
                inputs = tokenizer(prompt, return_tensors='pt')
                target_ids = tokenizer.encode(target, add_special_tokens=False)
                if len(target_ids) != 1: raise ValueError('protocol requires one target token')
                captured = {}
                def hook(module, arguments, output):
                    captured['key'] = arguments[0].detach()[0,-1].numpy().copy()
                    captured['output'] = output
                    output.retain_grad()
                handle = projection.register_forward_hook(hook)
                prediction = model(**inputs, use_cache=False).logits[0,-1]
                loss = -prediction.log_softmax(-1)[target_ids[0]]
                baseline = prediction.detach().numpy().copy()
                loss.backward(); handle.remove()
                key = captured['key'].astype(np.float64)
                gradient = captured['output'].grad[0,-1].numpy().astype(np.float64)
                reference = projection.weight.grad.numpy().T.astype(np.float64)
                outer = np.outer(gradient, key)
                gradient_error = float(np.linalg.norm(reference-outer)/np.linalg.norm(reference))
                orientation_error = float(np.max(np.abs(original.astype(np.float64)@key + projection.bias.detach().numpy() - captured['output'].detach()[0,-1].numpy())))
                if gradient_error > 1e-5 or orientation_error > 1e-4: raise ValueError('projection/gradient contract failed')
                left, right, value, rate = gradient_factors(original,key,gradient)
                edited = quantize(original.astype(np.float64)+left@right.T,precision)
                apply(edited); edited_logits = logits(inputs)
                fid=len(fixtures); directory=out/'work'/f'fixture-{fid:03d}'; directory.mkdir()
                np.savez(directory/'inputs.npz',original=original,edited=edited,left=left,right=right,key=key,value=value,direction=right[:,0])
                np.savez(directory/'edited.npz',weight=edited)
                np.savez(directory/'forward.npz',baseline=baseline,edited=edited_logits)
                f={'fixture_id':fid,'case_id':case_id,'precision':precision,'shape':list(original.shape),
                   'original_tensor_sha256':tensor_digest(original),'edited_tensor_sha256':tensor_digest(edited),
                   'input_file_sha256':sha(directory/'inputs.npz'),'gradient_outer_relative_error':gradient_error,
                   'projection_forward_max_abs_error':orientation_error,'learning_rate':float(rate),
                   'achieved_relative_update':float(np.linalg.norm(edited.astype(float)-original)/np.linalg.norm(original.astype(float))),
                   'target_probability_before':float(torch.from_numpy(baseline).softmax(-1)[target_ids[0]]),
                   'target_probability_after':float(torch.from_numpy(edited_logits).softmax(-1)[target_ids[0]]),
                   'edited_logits_max_abs_change':float(np.max(np.abs(edited_logits-baseline)))}
                fixtures.append(f); (out/'fixtures.json').write_text(json.dumps(fixtures,indent=2)+'\n')
                del captured, prediction, loss, reference, outer
                for repeat in range(args.repeats):
                    offset=(fid+repeat)%3
                    for method in METHODS[offset:]+METHODS[:offset]:
                        record=directory/f'{method}-{repeat}.npz'
                        build=_run_worker(['build','--inputs',str(directory/'inputs.npz'),'--record',str(record),'--method',method,'--precision',precision,'--alpha','1'],120)
                        undo=(_run_worker(['undo','--edited',str(directory/'edited.npz'),'--record',str(record),'--method',method],120) if build['status']=='ok' else {'status':'skipped'})
                        ok=build['status']==undo['status']=='ok'
                        row={'fixture_id':fid,'case_id':case_id,'shape':f'{original.shape[0]}x{original.shape[1]}','precision':precision,'method':method,'repeat':repeat,'status':'ok' if ok else 'worker_failure','build':build,'undo':undo,'bitwise_restored':ok and undo['restored_tensor_sha256']==f['original_tensor_sha256']}
                        rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                        if ok and repeat==0:
                            payload=_load_arrays(record)
                            restored=restore_checkpoint(edited,payload) if method=='checkpoint' else restore_from_receipt(edited,payload)
                            if tensor_digest(restored)!=undo['restored_tensor_sha256']: raise ValueError('evaluator/worker restore differs')
                            apply(restored); restored_logits=logits(inputs)
                            np.savez(directory/f'{method}-restored-logits.npz',logits=restored_logits)
                            behavior.append({'fixture_id':fid,'method':method,'logits_bitwise_equal':bool(np.array_equal(restored_logits,baseline)),'logits_max_abs_error':float(np.max(np.abs(restored_logits-baseline)))})
                            (out/'behavior.json').write_text(json.dumps(behavior,indent=2)+'\n')
                print(f'{len(rows)} observations: {precision} case={case_id}',flush=True)
    summary=summarize_costs(rows)
    summary['behavior']=behavior
    summary['status']='complete' if summary['failed_observations']==0 and summary['exact_bitwise_restored']==summary['exact_observations'] and all(b['logits_bitwise_equal'] for b in behavior if b['method']!='delta') else 'failed'
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('groups','behavior')},indent=2))
    return 0 if summary['status']=='complete' else 1

if __name__=='__main__': raise SystemExit(main())
