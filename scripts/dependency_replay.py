"""Pinned-author component ablations, checked reuse, and 8/16-edit withdrawal."""
import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
from reversible_edit.receipts import tensor_digest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = load_script('selective_replay')
components = load_script('dependency_components')
chain = previous.chain
request = previous.request
HISTORY = [request(s, t) for s, t in [
    ('Spain', 'Oslo'), ('Canada', 'Rome'), ('Charles Dickens', 'French'),
    ('Japan', 'Paris'), ('Spain', 'Tokyo'), ('Germany', 'Rome'),
    ('Italy', 'Berlin'), ('France', 'Oslo'), ('Canada', 'Paris'),
    ('Japan', 'Rome'), ('Charles Dickens', 'Italian'), ('Italy', 'Tokyo'),
    ('France', 'Berlin'), ('Germany', 'Paris'), ('Spain', 'Rome'), ('Canada', 'Tokyo')]]
for i in [2, 10]:
    HISTORY[i]['prompt'] = 'The native language of {} is'
ARMS = ['fresh', 'old_target', 'old_keys', 'old_both', 'old_update', 'checked']


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def valid_hit(entry, dependency):
    """Return false for changed dependencies; corrupted cached values raise."""
    if entry is None or entry['dependency'] != dependency:
        return False
    value = entry['value']
    array = value.detach().numpy() if hasattr(value, 'detach') else np.asarray(value)
    if tensor_digest(array) != entry['value_digest']:
        raise ValueError('cached component corruption')
    return True


class ComponentCache:
    def __init__(self, lab, algorithm, req, mode, old=None):
        self.lab, self.algorithm, self.mode = lab, algorithm, mode
        self.old, self.entries, self.events = old or {}, {}, []
        self.contract = json_digest({'fixed': lab.contract, 'algorithm': algorithm, 'request': req})

    def call(self, kind, layer, compute, explicit=None):
        slot = kind + ':' + str(layer)
        start = time.perf_counter()
        fingerprint = explicit() if explicit else self.lab.model_digest(None if kind == 'target' else layer)
        dependency = json_digest([self.contract, slot, fingerprint])
        old = self.old.get(slot)
        allowed = (self.mode == 'checked' and valid_hit(old, dependency))
        unsafe = self.mode in ('old_both', 'old_target' if kind == 'target' else 'old_keys')
        if unsafe:
            if old is None:
                raise ValueError('missing diagnostic cache ' + slot)
            # Deliberately ignore dependency for this labelled unsafe control.
            valid_hit(old, old['dependency'])
        hit = allowed or unsafe
        check_seconds = time.perf_counter() - start
        start = time.perf_counter()
        value = old['value'].detach().clone() if hit else compute().detach().clone()
        compute_seconds = time.perf_counter() - start
        self.entries[slot] = {'dependency': dependency, 'value': value,
                              'value_digest': tensor_digest(value.numpy())}
        self.events.append({'slot': slot, 'hit': hit, 'dependency_equal': old is not None and old['dependency'] == dependency,
                            'check_seconds': check_seconds, 'compute_seconds': compute_seconds})
        return value


class Lab:
    def __init__(self, args):
        import torch
        import transformers
        from transformers import AutoTokenizer, GPT2LMHeadModel
        self.torch = torch
        self.out = args.out.resolve()
        self.out.mkdir(parents=True)
        self.work = self.out / 'work'
        self.work.mkdir()
        self.store = previous.ArtifactStore()
        refdir = ROOT / 'results/20261110_factual_chain'
        ref = json.loads((refdir / 'run.json').read_text())
        prior = json.loads((refdir / 'summary.json').read_text())
        pins = json.loads((ROOT / 'results/20261110_novel_information/run.json').read_text())['model_files']
        for folder, mapping in [(args.model_dir, pins), (args.upstream, ref['upstream_files'])]:
            for name, digest in mapping.items():
                if chain.digest_file(folder / name) != digest:
                    raise ValueError('pinned dependency mismatch: ' + str(folder / name))
        torch.set_num_threads(1)
        torch.manual_seed(7)
        torch.use_deterministic_algorithms(True)
        sys.path.insert(0, str(args.upstream.resolve()))
        stats = types.ModuleType('rome.layer_stats')
        def disabled(*a, **kw):
            raise RuntimeError('use frozen base covariance')
        stats.layer_stats = disabled
        sys.modules['rome.layer_stats'] = stats
        self.rm = importlib.import_module('rome.rome_main')
        self.cv = importlib.import_module('rome.compute_v')
        self.cu = importlib.import_module('rome.compute_u')
        self.mm = importlib.import_module('memit.memit_main')
        self.rm.CONTEXT_TEMPLATES_CACHE = ref['contexts']
        self.mm.CONTEXT_TEMPLATES_CACHE = [ref['contexts']]
        self.hp = {'rome': self.rm.ROMEHyperParams(**ref['hparams']['rome']),
                   'memit': self.mm.MEMITHyperParams(**ref['hparams']['memit'])}
        self.tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = 'right'
        self.model = GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True,
                        use_safetensors=True, attn_implementation='eager').eval()
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.params = dict(self.model.named_parameters())
        self.names = [f'transformer.h.{i}.mlp.c_proj.weight' for i in [4, 5, 6]]
        self.base = self.weights()
        self.cov = {}
        for layer in [4, 5, 6]:
            filename = f'work/covariance-{layer}.npz'
            path = refdir / filename
            if chain.digest_file(path) != prior['binary_artifacts'][filename]:
                raise ValueError('covariance pin mismatch')
            self.cov[f'transformer.h.{layer}.mlp.c_proj'] = torch.from_numpy(chain.arrays(path)['covariance'])
        inverse = {n: torch.linalg.inv(c).float() for n, c in self.cov.items()}
        self.cu.get_inv_cov = lambda model, tok, name, *a: inverse[name]
        self.mm.get_cov = lambda model, tok, name, *a, **kw: self.cov[name]
        self.rome_source = (args.upstream / 'rome/compute_v.py').read_text()
        self.memit_source = (args.upstream / 'memit/memit_main.py').read_text()
        self.probes = list(previous.PROBES)
        for subject in ['Japan', 'Italy']:
            self.probes.append((subject.lower(), f'The capital of {subject} is', [' Paris', ' Rome', ' Berlin', ' Tokyo']))
        # Cover every retained request target, including later conflicting writes.
        for req in HISTORY:
            prompt, target = req['prompt'].format(req['subject']), req['target_new']['str']
            for i, (pid, p, targets) in enumerate(self.probes):
                if p == prompt:
                    self.probes[i] = (pid, p, list(dict.fromkeys(targets + [target])))
        self.target_ids = {}
        for _, _, targets in self.probes:
            for target in targets:
                ids = self.tok.encode(target, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError('fixed panel requires single token: ' + target)
                self.target_ids[target] = ids[0]
        sources = ['scripts/dependency_replay.py', 'scripts/dependency_components.py',
                   'scripts/selective_replay.py', 'scripts/factual_edit_chain.py', 'docs/DEPENDENCY_REPLAY.md']
        self.manifest = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'sources': {n: chain.digest_file(ROOT / n) for n in sources},
            'model': ref['model'], 'model_revision': ref['model_revision'], 'model_files': pins,
            'upstream_commit': ref['upstream_commit'], 'upstream_files': ref['upstream_files'],
            'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__,
            'model_config': self.model.config.to_dict(), 'hparams': ref['hparams'], 'contexts': ref['contexts'],
            'covariance': {n: tensor_digest(c.numpy()) for n, c in self.cov.items()},
            'seed': 7, 'device': 'cpu', 'dtype': 'float32', 'threads': 1, 'attention': 'eager',
            'pairs': previous.PAIRS, 'history': HISTORY, 'lengths': [8, 16], 'probes': self.probes,
            'timing_repeats': 3, 'max_new_tokens': 6,
            'scope': 'fixed GPT-2 development study; no arbitrary-model cache guarantee, SOTA or general-ability claim'}
        self.contract = json_digest(self.manifest)
        self.base_digest = self.model_digest()
        self.save_json('run.json', self.manifest)
        self.result = {'status': 'running', 'baseline': self.evaluate('baseline'), 'decomposition': [], 'histories': []}
        self.save()

    def save_json(self, name, data):
        (self.out / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

    def save(self):
        self.save_json('results.json', self.result)

    def model_digest(self, layer=None):
        h = hashlib.sha256()
        tensors = list(self.model.named_parameters()) + list(self.model.named_buffers())
        for name, value in tensors:
            if layer is None or components.prefix_dependency(name, layer):
                array = value.detach().numpy()
                h.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
                h.update(array.tobytes())
        return h.hexdigest()

    def weights(self):
        return {n: self.params[n].detach().clone() for n in self.names}

    def set_weights(self, weights):
        with self.torch.no_grad():
            for n, p in weights.items():
                self.params[n].copy_(p)

    def reset(self):
        self.set_weights(self.base)
        if self.model_digest() != self.base_digest:
            raise RuntimeError('base reset mismatch')

    def apply(self, factors):
        with self.torch.no_grad():
            for n, (key, value) in factors.items():
                self.params[n].add_((key @ value.T).float())

    def edit(self, algorithm, req, label, mode='fresh', old=None, receipt=False):
        torch = self.torch
        before = self.model_digest()
        originals = self.weights()
        cache = ComponentCache(self, algorithm, req, mode, old)
        if algorithm == 'rome':
            author_u, author_v = self.rm.compute_u, self.rm.compute_v
            if mode != 'author':
                self.rm.compute_u = lambda *a: cache.call('key', a[4], lambda: author_u(*a))
                self.rm.compute_v = components.adapt_rome(self.rome_source, self.cv.__dict__,
                    lambda layer, fn: cache.call('target', layer, fn))
            execute = lambda: self.rm.execute_rome(self.model, self.tok, req, self.hp[algorithm])
        else:
            execute_memit = self.mm.execute_memit
            if mode != 'author':
                execute_memit = components.adapt_memit(self.memit_source, self.mm.__dict__,
                    lambda *a: cache.call('target', a[4], lambda: self.mm.compute_z(*a)),
                    lambda *a: cache.call('key', a[4], lambda: self.mm.compute_ks(*a)),
                    lambda layer, keys, cov, fn: cache.call('solve', layer, fn,
                        explicit=lambda: [tensor_digest(keys.detach().numpy()), tensor_digest(cov.detach().numpy())]))
            execute = lambda: execute_memit(self.model, self.tok, [req], self.hp[algorithm])
        torch.manual_seed(7)
        try:
            with (self.out / (label + '-optimization.txt')).open('x') as log, contextlib.redirect_stdout(log):
                start = time.perf_counter()
                factors = execute()
                execute_seconds = time.perf_counter() - start
        finally:
            if algorithm == 'rome':
                self.rm.compute_u, self.rm.compute_v = author_u, author_v
        if self.model_digest() != before:
            raise RuntimeError('author execute restoration mismatch')
        factors = {n: (k[:, None], v[:, None]) if algorithm == 'rome' else (k, v) for n, (k, v) in factors.items()}
        start = time.perf_counter()
        self.apply(factors)
        apply_seconds = time.perf_counter() - start
        info = {'edit_seconds': execute_seconds + apply_seconds, 'execute_seconds': execute_seconds,
                'apply_seconds': apply_seconds, 'events': cache.events,
                'check_seconds': sum(e['check_seconds'] for e in cache.events),
                'cache_hits': sum(e['hit'] for e in cache.events),
                'cache_slots': len(cache.events), 'receipt_bytes': 0, 'component_bytes': 0}
        record = {'before': before, 'after': self.model_digest(), 'receipts': {}, 'factors': factors, 'cache': cache.entries}
        if cache.entries:
            path = self.work / (label + '-components.npz')
            metadata = {s: {k: v for k, v in e.items() if k != 'value'} for s, e in cache.entries.items()}
            self.store.save(path, **{s: e['value'].numpy() for s, e in cache.entries.items()}, metadata=np.array(json.dumps(metadata)))
            info['component_bytes'] = path.stat().st_size
        if receipt:
            for n, (key, value) in factors.items():
                blob = chain.pack_record(originals[n].numpy().T.copy(), self.params[n].detach().numpy().T.copy(),
                                         value.detach().numpy(), key.detach().numpy(), 'delta_exact')
                path = self.work / (label + '-' + n + '-receipt.npz')
                self.store.write(path, blob)
                record['receipts'][n] = path
                info['receipt_bytes'] += len(blob)
        return record, info

    def rewind(self, records):
        start = time.perf_counter()
        for record in reversed(records):
            if self.model_digest() != record['after']:
                raise RuntimeError('wrong history tip')
            for name, path in reversed(list(record['receipts'].items())):
                # In-process NumPy receipt inverse receives only live tensor + receipt.
                restored = chain.restore(self.params[name].detach().numpy().T.copy(), chain.arrays(path))
                with self.torch.no_grad():
                    self.params[name].copy_(self.torch.from_numpy(restored.T.copy()))
            if self.model_digest() != record['before']:
                raise RuntimeError('rewind did not recover exact prefix')
        return time.perf_counter() - start

    def evaluate(self, label):
        torch = self.torch
        rows, first = [], []
        with torch.no_grad():
            for pid, prompt, targets in self.probes:
                ids = torch.tensor([self.tok.encode(prompt, add_special_tokens=False)])
                generated = []
                for step in range(6):
                    logits = self.model(input_ids=ids, use_cache=False).logits[0, -1]
                    if step == 0:
                        first.append(logits.numpy().copy())
                    token = int(logits.argmax())
                    generated.append(token)
                    ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
                    if token == self.tok.eos_token_id:
                        break
                text = self.tok.decode(generated)
                logits = torch.from_numpy(first[-1])
                rows.append({'id': pid, 'prompt': prompt, 'generated': text, 'generated_ids': generated,
                    'targets': {t: {'probability': float(logits.softmax(-1)[self.target_ids[t]]),
                        'rank': int((logits > logits[self.target_ids[t]]).sum()) + 1,
                        'word_prefix': previous.phrase_prefix(text, t)} for t in targets}})
        matrix = np.stack(first)
        self.store.save(self.work / (label + '-logits.npz'), logits=matrix)
        self.store.save(self.work / (label + '-weights.npz'), **{n: p.numpy() for n, p in self.weights().items()})
        return {'state_digest': self.model_digest(), 'logits_digest': tensor_digest(matrix), 'probes': rows, 'label': label}

    def compare(self, arm, reference):
        logits = chain.arrays(self.work / (arm['label'] + '-logits.npz'))['logits']
        ref = chain.arrays(self.work / (reference['label'] + '-logits.npz'))['logits']
        lp = self.torch.from_numpy(ref).double().log_softmax(-1)
        other = self.torch.from_numpy(logits).double().log_softmax(-1)
        return {'state_bitwise': arm['state_digest'] == reference['state_digest'],
                'logits_bitwise': arm['logits_digest'] == reference['logits_digest'],
                'max_abs_logit_error': float(np.max(np.abs(logits - ref))),
                'mean_kl': float((lp.exp() * (lp - other)).sum(-1).mean()),
                'same_generations': sum(a['generated_ids'] == b['generated_ids'] for a, b in zip(arm['probes'], reference['probes'])),
                'probe_count': len(self.probes)}

    def guard_controls(self, algorithm, req, old):
        """Exercise positive/negative lookup decisions on the real model."""
        layer = self.hp[algorithm].layers[0]
        key_slot = 'key:' + str(layer)
        events = []
        def key_check(label, expected, request_override=None, cache_override=None):
            cache = ComponentCache(self, algorithm, request_override or req, 'checked', cache_override or old)
            cache.call('key', layer, lambda: old[key_slot]['value'])
            event = dict(cache.events[-1], label=label, expected_hit=expected)
            if event['hit'] != expected:
                raise RuntimeError('guard negative control failed: ' + label)
            events.append(event)
        key_check('unchanged', True)
        modified = dict(req, subject=req['subject'] + ' changed')
        key_check('request_changed', False, modified)
        for name, label, expected in [
            (f'transformer.h.{layer}.mlp.c_fc.weight', 'upstream_changed', False),
            (f'transformer.h.{layer}.mlp.c_proj.weight', 'own_projection_changed', True)]:
            parameter = self.params[name]
            before = parameter[0, 0].detach().clone()
            try:
                with self.torch.no_grad():
                    parameter[0, 0].add_(.01)
                key_check(label, expected)
                target_layer = self.hp[algorithm].layers[-1]
                target_slot = 'target:' + str(target_layer)
                cache = ComponentCache(self, algorithm, req, 'checked', old)
                cache.call('target', target_layer, lambda: old[target_slot]['value'])
                if cache.events[-1]['hit']:
                    raise RuntimeError('changed weight admitted target reuse')
                events.append(dict(cache.events[-1], label=label + '_target', expected_hit=False))
            finally:
                with self.torch.no_grad():
                    parameter[0, 0].copy_(before)
        corrupt = {s: dict(e) for s, e in old.items()}
        corrupt[key_slot]['value'] = old[key_slot]['value'].clone()
        corrupt[key_slot]['value'].view(-1)[0] += 1
        try:
            key_check('corrupt_value', False, cache_override=corrupt)
        except ValueError as error:
            events.append({'label': 'corrupt_value', 'rejected': str(error)})
        else:
            raise RuntimeError('corrupt value was not rejected')
        if algorithm == 'memit':
            key = old[key_slot]['value'].T.double()
            cov = self.cov[f'transformer.h.{layer}.mlp.c_proj']
            before = cov[0, 0].clone()
            try:
                cov[0, 0] += .01
                cache = ComponentCache(self, algorithm, req, 'checked', old)
                slot = 'solve:' + str(layer)
                cache.call('solve', layer, lambda: old[slot]['value'],
                           explicit=lambda: [tensor_digest(key.numpy()), tensor_digest(cov.numpy())])
                if cache.events[-1]['hit']:
                    raise RuntimeError('changed covariance admitted solve reuse')
                events.append(dict(cache.events[-1], label='covariance_changed', expected_hit=False))
            finally:
                cov[0, 0].copy_(before)
        return events

    def decomposition(self):
        for algorithm in ['rome', 'memit']:
            for pair in previous.PAIRS:
                label = algorithm + '-' + pair['id']
                print('decomposition', label, flush=True)
                self.reset()
                _, author_info = self.edit(algorithm, pair['B'], label + '-author', mode='author')
                reference = self.evaluate(label + '-author')
                self.reset()
                fresh, fresh_info = self.edit(algorithm, pair['B'], label + '-capture')
                adapted = self.evaluate(label + '-capture')
                if adapted['state_digest'] != reference['state_digest']:
                    raise RuntimeError('instrumentation differs from author')
                self.reset()
                guard_controls = self.guard_controls(algorithm, pair['B'], fresh['cache'])
                a, ai = self.edit(algorithm, pair['A'], label + '-A', receipt=True)
                b, bi = self.edit(algorithm, pair['B'], label + '-B-after-A', receipt=True)
                tip = self.weights()
                difference = {}
                for slot, old in b['cache'].items():
                    x, y = old['value'].double(), fresh['cache'][slot]['value'].double()
                    difference[slot] = {'bitwise': old['value_digest'] == fresh['cache'][slot]['value_digest'],
                        'relative_difference': float((x - y).norm() / y.norm()),
                        'dependency_equal': old['dependency'] == fresh['cache'][slot]['dependency']}
                row = {'algorithm': algorithm, 'pair': pair, 'reference': reference,
                       'adaptation_comparison': self.compare(adapted, reference), 'component_differences': difference,
                       'author_info': author_info, 'capture_info': fresh_info, 'A_info': ai, 'old_B_info': bi,
                       'guard_controls': guard_controls, 'arms': [], 'timing_repeats': []}
                self.result['decomposition'].append(row)
                for mode in ARMS:
                    self.set_weights(tip)
                    undo_seconds = self.rewind([a, b])
                    if mode == 'old_update':
                        start = time.perf_counter()
                        self.apply(b['factors'])
                        info = {'edit_seconds': time.perf_counter() - start}
                    else:
                        _, info = self.edit(algorithm, pair['B'], label + '-' + mode, mode, b['cache'])
                    observation = self.evaluate(label + '-' + mode)
                    row['arms'].append({'mode': mode, 'undo_seconds': undo_seconds, 'info': info,
                                        'result': observation, 'comparison': self.compare(observation, reference)})
                    self.save()
                for repeat in range(3):
                    times = {}
                    order = ['author', 'checked'] if repeat % 2 == 0 else ['checked', 'author']
                    for mode in order:
                        self.reset()
                        _, info = self.edit(algorithm, pair['B'], f'{label}-timing-{repeat}-{mode}', mode, b['cache'])
                        times[mode] = {**info, 'state_bitwise': self.model_digest() == reference['state_digest']}
                    row['timing_repeats'].append({'repeat': repeat, 'order': order, 'arms': times})
                    self.save()

    def histories(self):
        for algorithm in ['rome', 'memit']:
            self.reset()
            records, infos = [], []
            snapshots = {0: self.weights()}
            prefix_states = [self.model_digest()]
            checkpoints = {}
            for i, req in enumerate(HISTORY):
                print('history-build', algorithm, i, flush=True)
                record, info = self.edit(algorithm, req, f'{algorithm}-history-{i}', receipt=True)
                records.append(record)
                infos.append(info)
                prefix_states.append(self.model_digest())
                if i + 1 in [8, 16]:
                    snapshots[i + 1] = self.weights()
                    checkpoints[i + 1] = self.evaluate(f'{algorithm}-history-tip-{i + 1}')
            for length in [8, 16]:
                for deleted in [0, length // 2, length - 2]:
                    label = f'{algorithm}-length-{length}-delete-{deleted}'
                    print('withdrawal', label, flush=True)
                    self.reset()
                    ref_infos = []
                    for i, req in enumerate(HISTORY[:length]):
                        if i != deleted:
                            _, info = self.edit(algorithm, req, f'{label}-reference-{i}', mode='author')
                            ref_infos.append(info)
                    reference = self.evaluate(label + '-reference')
                    row = {'algorithm': algorithm, 'length': length, 'deleted_index': deleted,
                           'reference': reference, 'reference_infos': ref_infos,
                           'reference_edit_seconds': sum(x['edit_seconds'] for x in ref_infos),
                           'original_tip': checkpoints[length], 'history_infos': infos[:length],
                           'journal_bytes': sum(x['receipt_bytes'] for x in infos[:length]),
                           'component_bytes': sum(x['component_bytes'] for x in infos[:length]), 'arms': []}
                    self.result['histories'].append(row)
                    for mode in ['checked', 'old_update']:
                        self.set_weights(snapshots[length])
                        undo_seconds = self.rewind(records[deleted:length])
                        if self.model_digest() != prefix_states[deleted]:
                            raise RuntimeError('incorrect deletion prefix')
                        replay_infos = []
                        for i in range(deleted + 1, length):
                            if mode == 'checked':
                                _, info = self.edit(algorithm, HISTORY[i], f'{label}-checked-{i}', mode, records[i]['cache'])
                            else:
                                start = time.perf_counter()
                                self.apply(records[i]['factors'])
                                info = {'edit_seconds': time.perf_counter() - start}
                            replay_infos.append(info)
                        observation = self.evaluate(label + '-' + mode)
                        row['arms'].append({'mode': mode, 'undo_seconds': undo_seconds, 'infos': replay_infos,
                            'replay_seconds': sum(x['edit_seconds'] for x in replay_infos),
                            'result': observation, 'comparison': self.compare(observation, reference)})
                        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('out must be new')
    lab = Lab(args)
    try:
        lab.decomposition()
        lab.histories()
        lab.result['status'] = 'complete'
    except BaseException as error:
        lab.result['status'] = 'failed'
        lab.result['error'] = repr(error)
        raise
    finally:
        lab.save()
        lab.save_json('artifacts.json', {str(p.relative_to(lab.out)): chain.digest_file(p) for p in sorted(lab.work.glob('*.npz'))})


if __name__ == '__main__':
    main()
