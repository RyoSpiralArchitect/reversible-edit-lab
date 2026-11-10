"""Separate floating-point rollback error from later-edit dependency on history."""
import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

import numpy as np
from reversible_edit.cost_experiments import THREAD_ENV
from reversible_edit.receipts import tensor_digest
from reversible_edit.roundoff import make_roundoff_patch, apply_roundoff_patch, RoundoffPatch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("factual", ROOT / "scripts/factual_edit_chain.py")
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


def request(subject, target, prompt="The capital of {} is"):
    return {"subject": subject, "prompt": prompt, "target_new": {"str": " " + target}}


PAIRS = [
    {"id": "different_subjects", "A": request("Spain", "Oslo"), "B": request("Canada", "Rome")},
    {"id": "conflicting_targets", "A": request("Spain", "Oslo"), "B": request("Spain", "Tokyo")},
    {"id": "different_relations", "A": request("Charles Dickens", "French", "The native language of {} is"), "B": request("Canada", "Rome")},
]
PROBES = [
    ("spain", "The capital of Spain is", [" Oslo", " Tokyo", " Madrid"]),
    ("canada", "The capital of Canada is", [" Rome", " Ottawa"]),
    ("dickens", "The native language of Charles Dickens is", [" French", " English"]),
    ("spain_paraphrase", "Spain has its capital in", [" Oslo", " Tokyo", " Madrid"]),
    ("canada_paraphrase", "Canada has its capital in", [" Rome", " Ottawa"]),
    ("dickens_paraphrase", "Charles Dickens wrote primarily in", [" French", " English"]),
    ("germany", "The capital of Germany is", [" Berlin"]),
    ("france", "The capital of France is", [" Paris"]),
    ("control_sequence", "1, 2, 3, 4,", [" 5"]),
    ("control_days", "Monday, Tuesday, Wednesday,", [" Thursday"]),
    ("control_hot", "The opposite of hot is", [" cold"]),
    ("control_up", "The opposite of up is", [" down"]),
]


def phrase_prefix(generated, target):
    return bool(target) and re.match(re.escape(target) + r"(?=$|[\s.,!?;:\"')\]])", generated) is not None


def direct_xor_record(original, edited):
    """Matched zlib ablation: old XOR edited bits, without a factor predictor."""
    stream = io.BytesIO()
    patch = make_roundoff_patch(original, edited, "float32").to_bytes()
    np.savez_compressed(stream, patch=np.frombuffer(patch, dtype=np.uint8))
    return stream.getvalue()


def restore_direct_xor(edited, blob):
    patch = RoundoffPatch.from_bytes(chain.arrays(io.BytesIO(blob))["patch"].tobytes())
    return apply_roundoff_patch(edited, patch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, default=ROOT / "results/20261110_factual_chain")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("out must be new")
    ref = json.loads((args.reference_run / "run.json").read_text())
    old_summary = json.loads((args.reference_run / "summary.json").read_text())
    model_pins = json.loads((ROOT / "results/20261110_novel_information/run.json").read_text())["model_files"]
    for name, digest in model_pins.items():
        if chain.digest_file(args.model_dir / name) != digest:
            raise ValueError("pinned model file mismatch: " + name)
    upstream = args.upstream.resolve()
    for name, digest in ref["upstream_files"].items():
        if chain.digest_file(upstream / name) != digest:
            raise ValueError("pinned upstream file mismatch: " + name)
    import torch
    import transformers
    from transformers import AutoTokenizer, GPT2LMHeadModel

    torch.set_num_threads(1)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(upstream))
    stats = types.ModuleType("rome.layer_stats")
    def disabled(*args, **kwargs):
        raise RuntimeError("use declared frozen covariance")
    stats.layer_stats = disabled
    sys.modules["rome.layer_stats"] = stats
    rm = importlib.import_module("rome.rome_main")
    cu = importlib.import_module("rome.compute_u")
    mm = importlib.import_module("memit.memit_main")
    rm.CONTEXT_TEMPLATES_CACHE = ref["contexts"]
    mm.CONTEXT_TEMPLATES_CACHE = [ref["contexts"]]
    hp = {"rome": rm.ROMEHyperParams(**ref["hparams"]["rome"]), "memit": mm.MEMITHyperParams(**ref["hparams"]["memit"])}
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True, use_safetensors=True, attn_implementation="eager").eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    params = dict(model.named_parameters())
    names = [f"transformer.h.{layer}.mlp.c_proj.weight" for layer in [4, 5, 6]]
    base = {n: params[n].detach().clone() for n in names}
    out = args.out.resolve()
    out.mkdir(parents=True)
    work = out / "work"
    work.mkdir()
    covariance, covariance_info = {}, {}
    for layer in [4, 5, 6]:
        name = f"transformer.h.{layer}.mlp.c_proj"
        filename = f"work/covariance-{layer}.npz"
        cached = args.reference_run / filename
        if cached.exists():
            if chain.digest_file(cached) != old_summary["binary_artifacts"][filename]:
                raise ValueError("covariance hash mismatch")
            c = torch.from_numpy(chain.arrays(cached)["covariance"])
        else:
            chunks = []
            def collect(module, inputs, output):
                chunks.append(inputs[0].detach().reshape(-1, 3072).double())
            hook = model.transformer.h[layer].mlp.c_proj.register_forward_hook(collect)
            try:
                with torch.no_grad():
                    for sentence in ref["calibration"]:
                        model(**tok(sentence, return_tensors="pt"), use_cache=False)
            finally:
                hook.remove()
            x = torch.cat(chunks)
            c = x.T @ x / len(x)
            c += .01 * float(c.diag().mean()) * torch.eye(3072, dtype=torch.float64)
        covariance[name] = c
        covariance_info[name] = tensor_digest(c.numpy())
        np.savez(work / f"covariance-{layer}.npz", covariance=c.numpy())
    inverse = {n: torch.linalg.inv(c).float() for n, c in covariance.items()}
    cu.get_inv_cov = lambda model, tok, name, *args: inverse[name]
    mm.get_cov = lambda model, tok, name, *args, **kwargs: covariance[name]
    target_ids = {}
    for _, _, targets in PROBES:
        for target in targets:
            ids = tok.encode(target, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("this fixed panel requires single-token targets: " + target)
            target_ids[target] = ids[0]
    manifest = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "script_sha256": chain.digest_file(__file__), "receipt_helper_sha256": chain.digest_file(chain.__file__),
                "model": ref["model"], "model_revision": ref["model_revision"], "model_files": model_pins,
                "upstream_commit": ref["upstream_commit"], "upstream_files": ref["upstream_files"],
                "reference_manifest_sha256": chain.digest_file(args.reference_run / "run.json"),
                "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
                "hparams": ref["hparams"], "calibration": ref["calibration"], "covariance": covariance_info,
                "contexts": ref["contexts"], "pairs": PAIRS, "probes": PROBES, "max_new_tokens": 6,
                "scope": "fixed six-chain dependency diagnostic; no SOTA, privacy, general capability or novelty proof"}
    (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def state():
        h = hashlib.sha256()
        for n, p in model.state_dict().items():
            h.update(n.encode())
            h.update(p.detach().numpy().tobytes())
        return h.hexdigest()

    def set_weights(weights):
        with torch.no_grad():
            for n, p in weights.items():
                params[n].copy_(p)

    def evaluate(label):
        rows, first = [], []
        with torch.no_grad():
            for probe_id, prompt, targets in PROBES:
                ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)])
                generated = []
                for step in range(6):
                    logits = model(input_ids=ids, use_cache=False).logits[0, -1]
                    if step == 0:
                        first.append(logits.numpy().copy())
                    token = int(logits.argmax())
                    generated.append(token)
                    ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
                    if token == tok.eos_token_id:
                        break
                text = tok.decode(generated)
                logits = torch.from_numpy(first[-1])
                scores = {t: {"probability": float(logits.softmax(-1)[target_ids[t]]),
                              "rank": int((logits > logits[target_ids[t]]).sum())+1,
                              "word_prefix": phrase_prefix(text, t)} for t in targets}
                rows.append({"id": probe_id, "prompt": prompt, "generated": text, "generated_ids": generated, "targets": scores})
        matrix = np.stack(first)
        np.savez(work / (label + "-logits.npz"), logits=matrix)
        weights = {n: params[n].detach().clone() for n in names}
        np.savez(work / (label + "-weights.npz"), **{n: p.numpy() for n, p in weights.items()})
        return {"state_digest": state(), "logits_digest": tensor_digest(matrix), "probes": rows}, matrix, weights

    def edit(algorithm, req, label):
        before_digest = state()
        originals = {n: params[n].detach().clone() for n in names}
        torch.manual_seed(7)
        start = time.perf_counter()
        with (out / (label + "-optimization.txt")).open("w") as log, contextlib.redirect_stdout(log):
            factors = rm.execute_rome(model, tok, req, hp[algorithm]) if algorithm == "rome" else mm.execute_memit(model, tok, [req], hp[algorithm])
        seconds = time.perf_counter() - start
        if state() != before_digest:
            raise RuntimeError("author execute failed restoration invariant")
        records = {}
        with torch.no_grad():
            for n, (key, value) in factors.items():
                if algorithm == "rome":
                    key, value = key[:, None], value[:, None]
                update = (key @ value.T).float()
                params[n].add_(update)
                records[n] = {"original": originals[n].numpy().T.copy(), "edited": params[n].detach().numpy().T.copy(),
                              "left": value.detach().numpy(), "right": key.detach().numpy(), "update": update.numpy().copy()}
                np.savez(work / (label + "-" + n + "-factors.npz"), **records[n])
        return records, seconds

    def prepare_records(records, label):
        paths, sizes = {}, {"checkpoint": 0, "delta_exact": 0, "direct_xor_zlib": 0}
        for n, record in records.items():
            paths[n] = {}
            for method in ["checkpoint", "delta_exact", "direct_xor_zlib"]:
                if method == "direct_xor_zlib":
                    blob = direct_xor_record(record["original"], record["edited"])
                    restored = restore_direct_xor(record["edited"], blob)
                    if tensor_digest(restored) != tensor_digest(record["original"]):
                        raise RuntimeError("direct XOR did not restore")
                else:
                    blob = chain.pack_record(*(record[k] for k in ["original", "edited", "left", "right"]), method)
                path = work / (label + "-" + n + "-" + method + ".npz")
                path.write_bytes(blob)
                paths[n][method] = path
                sizes[method] += len(blob)
        return paths, sizes

    def rewind(paths, method, label):
        start = time.perf_counter()
        worker_times = []
        for i, (n, methods) in enumerate(reversed(list(paths.items()))):
            edited_path = work / (label + f"-{i}-live.npz")
            restored_path = work / (label + f"-{i}-restored.npz")
            np.savez(edited_path, weight=params[n].detach().numpy().T.copy())
            process = subprocess.run([sys.executable, "-I", chain.__file__, "--undo", "--edited", str(edited_path), "--record", str(methods[method]), "--out", str(restored_path)], env=dict(os.environ, **THREAD_ENV), capture_output=True, text=True, timeout=120)
            if process.returncode:
                raise RuntimeError(process.stderr)
            worker_times.append(json.loads(process.stdout))
            with torch.no_grad():
                params[n].copy_(torch.from_numpy(chain.arrays(restored_path)["weight"].T.copy()))
        return {"parent_wall_seconds": time.perf_counter()-start, "workers": worker_times}

    def comparison(current, logits, weights, reference, reference_logits, reference_weights):
        base_lp = torch.from_numpy(reference_logits).double().log_softmax(-1)
        current_lp = torch.from_numpy(logits).double().log_softmax(-1)
        errors = {n: float((weights[n].double()-reference_weights[n].double()).norm()/reference_weights[n].double().norm()) for n in names}
        return {"state_bitwise": current["state_digest"] == reference["state_digest"],
                "logits_bitwise": current["logits_digest"] == reference["logits_digest"],
                "max_abs_logit_error": float(np.max(np.abs(logits-reference_logits))),
                "mean_kl_reference_to_arm": float((base_lp.exp() * (base_lp-current_lp)).sum(-1).mean()),
                "relative_weight_errors": errors,
                "same_generations": sum(x["generated_ids"] == y["generated_ids"] for x, y in zip(current["probes"], reference["probes"], strict=True)),
                "probe_count": len(PROBES)}

    baseline, _, _ = evaluate("baseline")
    result = {"baseline": baseline, "rows": []}
    def save():
        (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    for algorithm in ["rome", "memit"]:
        for pair in PAIRS:
            label = algorithm + "-" + pair["id"]
            set_weights(base)
            reference_factors, reference_seconds = edit(algorithm, pair["B"], label + "-B-only")
            reference, ref_logits, ref_weights = evaluate(label + "-B-only")
            set_weights(base)
            if state() != baseline["state_digest"]:
                raise RuntimeError("evaluator base reset mismatch")
            a_factors, a_seconds = edit(algorithm, pair["A"], label + "-A")
            after_a, _, _ = evaluate(label + "-A")
            a_paths, a_sizes = prepare_records(a_factors, label + "-A")
            b_factors, b_seconds = edit(algorithm, pair["B"], label + "-B-after-A")
            after_ab, _, tip_weights = evaluate(label + "-AB")
            b_paths, b_sizes = prepare_records(b_factors, label + "-B-after-A")
            dependency = {n: float(np.linalg.norm((r["update"].astype(float)-reference_factors[n]["update"].astype(float)).ravel()) / np.linalg.norm(reference_factors[n]["update"].astype(float).ravel())) for n, r in b_factors.items()}
            row = {"algorithm": algorithm, "pair": pair, "after_A": after_a, "after_AB": after_ab,
                   "reference_B_only": reference, "reference_seconds": reference_seconds,
                   "A_seconds": a_seconds, "B_after_A_seconds": b_seconds,
                   "B_update_relative_dependency": dependency, "record_bytes_A": a_sizes, "record_bytes_B": b_sizes, "arms": []}
            result["rows"].append(row)
            # A's receipt is state-bound and must reject a subsequent B write.
            first_name = next(iter(a_paths))
            try:
                chain.restore(params[first_name].detach().numpy().T.copy(), chain.arrays(a_paths[first_name]["delta_exact"]))
            except ValueError as exc:
                row["stale_A_receipt_rejected"] = str(exc)
            else:
                raise RuntimeError("stale A receipt unexpectedly accepted")
            save()
            for arm in ["subtract_A", "exact_rewind_stale_B", "exact_rewind_recompute_B", "checkpoint_recompute_B"]:
                set_weights(tip_weights)
                if state() != after_ab["state_digest"]:
                    raise RuntimeError("arms must start from the same AB state")
                timings, optimization_seconds = [], 0.0
                if arm == "subtract_A":
                    # Explicit diagnostic baseline, deliberately not a receipt apply.
                    with torch.no_grad():
                        for n, r in a_factors.items():
                            live = params[n].detach().numpy().T.copy()
                            restored = (live.astype(float)-r["left"].astype(float)@r["right"].astype(float).T).astype(np.float32)
                            params[n].copy_(torch.from_numpy(restored.T.copy()))
                else:
                    method = "checkpoint" if arm == "checkpoint_recompute_B" else "delta_exact"
                    timings.append(rewind(b_paths, method, label + "-" + arm + "-undo-B"))
                    if state() != after_a["state_digest"]:
                        raise RuntimeError("B undo did not recover A")
                    timings.append(rewind(a_paths, method, label + "-" + arm + "-undo-A"))
                    if state() != baseline["state_digest"]:
                        raise RuntimeError("A undo did not recover base")
                    if arm == "exact_rewind_stale_B":
                        with torch.no_grad():
                            for n, r in b_factors.items():
                                params[n].add_(torch.from_numpy(r["update"]))
                    else:
                        _, optimization_seconds = edit(algorithm, pair["B"], label + "-" + arm + "-B")
                current, logits, weights = evaluate(label + "-" + arm)
                cmp = comparison(current, logits, weights, reference, ref_logits, ref_weights)
                row["arms"].append({"method": arm, "result": current, "comparison_to_B_only": cmp,
                                    "verified_rewind_to_base": arm != "subtract_A", "undo_times": timings,
                                    "fresh_B_optimization_seconds": optimization_seconds})
                save()
                print(label, arm, cmp, flush=True)
                if arm.endswith("recompute_B") and not (cmp["state_bitwise"] and cmp["logits_bitwise"] and cmp["same_generations"] == len(PROBES)):
                    raise RuntimeError("fresh replay failed exact comparison; evidence retained")
    summary = {"status": "complete", "chains": len(result["rows"]),
               "binary_artifacts": {str(p.relative_to(out)): chain.digest_file(p) for p in sorted(work.glob("*.npz"))}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
