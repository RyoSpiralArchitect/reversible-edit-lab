"""Recompute comparisons and stale replay from retained arrays, without Torch."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from reversible_edit.receipts import tensor_digest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('chain', ROOT / 'scripts/factual_edit_chain.py')
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


def same_arrays(a, b):
    return set(a) == set(b) and all(tensor_digest(a[k]) == tensor_digest(b[k]) for k in a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    folder = args.run
    data = json.loads((folder / 'results.json').read_text())
    if data['status'] != 'complete':
        raise ValueError('run incomplete')
    work, errors = folder / 'work', []
    counts = {'comparisons': 0, 'author_adaptations': 0, 'component_differences': 0, 'restored_layers': 0, 'stale_replays': 0}
    def compare(arm, reference, reported):
        a = chain.arrays(work / (arm['label'] + '-logits.npz'))['logits']
        b = chain.arrays(work / (reference['label'] + '-logits.npz'))['logits']
        aw = chain.arrays(work / (arm['label'] + '-weights.npz'))
        bw = chain.arrays(work / (reference['label'] + '-weights.npz'))
        if tensor_digest(a) != arm['logits_digest'] or tensor_digest(b) != reference['logits_digest']:
            errors.append('logit digest: ' + arm['label'])
        if same_arrays(aw, bw) != reported['state_bitwise']:
            errors.append('stored weight equality: ' + arm['label'])
        if (tensor_digest(a) == tensor_digest(b)) != reported['logits_bitwise']:
            errors.append('logit equality: ' + arm['label'])
        lp = b.astype(np.float64) - np.logaddexp.reduce(b.astype(np.float64), axis=-1, keepdims=True)
        lq = a.astype(np.float64) - np.logaddexp.reduce(a.astype(np.float64), axis=-1, keepdims=True)
        kl = float((np.exp(lp) * (lp - lq)).sum(-1).mean())
        if not np.isclose(kl, reported['mean_kl'], atol=1e-12, rtol=1e-10):
            errors.append('KL: ' + arm['label'])
        if float(np.abs(a - b).max()) != reported['max_abs_logit_error']:
            errors.append('logit max: ' + arm['label'])
        counts['comparisons'] += 1
    for row in data['decomposition']:
        for arm in row['arms']:
            compare(arm['result'], row['reference'], arm['comparison'])
        label = row['algorithm'] + '-' + row['pair']['id']
        for suffix in ['weights', 'logits']:
            adapted = chain.arrays(work / (label + '-capture-' + suffix + '.npz'))
            author = chain.arrays(work / (label + '-author-' + suffix + '.npz'))
            if not same_arrays(adapted, author):
                errors.append('author adaptation: ' + label + '/' + suffix)
        counts['author_adaptations'] += 1
        fresh = chain.arrays(work / (label + '-capture-components.npz'))
        old = chain.arrays(work / (label + '-B-after-A-components.npz'))
        for slot, reported in row['component_differences'].items():
            x, y = old[slot], fresh[slot]
            if (tensor_digest(x) == tensor_digest(y)) != reported['bitwise']:
                errors.append('component equality: ' + label + '/' + slot)
            relative = np.linalg.norm(x.astype(float) - y.astype(float)) / np.linalg.norm(y.astype(float))
            if not np.isclose(relative, reported['relative_difference'], rtol=1e-12, atol=1e-14):
                errors.append('component relative difference: ' + label + '/' + slot)
            counts['component_differences'] += 1
    for row in data['histories']:
        for arm in row['arms']:
            compare(arm['result'], row['reference'], arm['comparison'])
        algorithm, length, deleted = row['algorithm'], row['length'], row['deleted_index']
        weights = chain.arrays(work / (row['original_tip']['label'] + '-weights.npz'))
        layers = [5] if algorithm == 'rome' else [4, 5, 6]
        for i in reversed(range(deleted, length)):
            for layer in reversed(layers):
                name = f'transformer.h.{layer}.mlp.c_proj.weight'
                payload = chain.arrays(work / f'{algorithm}-history-{i}-{name}-receipt.npz')
                weights[name] = chain.restore(weights[name].T.copy(), payload).T.copy()
                counts['restored_layers'] += 1
        for i in range(deleted + 1, length):
            for layer in layers:
                name = f'transformer.h.{layer}.mlp.c_proj.weight'
                payload = chain.arrays(work / f'{algorithm}-history-{i}-{name}-receipt.npz')
                update = (payload['right'] @ payload['left'].T).astype(np.float32)
                weights[name] += update
        stale = next(a['result'] for a in row['arms'] if a['mode'] == 'old_update')
        expected = chain.arrays(work / (stale['label'] + '-weights.npz'))
        if not same_arrays(weights, expected):
            errors.append(f'NumPy stale replay: {algorithm}/{length}/{deleted}')
        counts['stale_replays'] += 1
    result = {'errors': errors, **counts, 'scope': 'Recomputed saved-array comparisons and receipt replay; no optimizer/forward rerun.'}
    (folder / 'arrays-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if errors:
        raise ValueError(errors)


if __name__ == '__main__':
    main()
