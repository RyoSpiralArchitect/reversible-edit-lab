"""Fresh arbitrary associations: unprompted retrieval, rollback, and controls.

Optional CPU/Torch runner; pure scoring helpers remain usable in NumPy-only CI.
"""
import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from reversible_edit.cost_experiments import THREAD_ENV
from reversible_edit.receipts import tensor_digest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("factual_chain", ROOT / "scripts/factual_edit_chain.py")
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)

CONTROLS = [
    ("arithmetic", "2 + 2 =", " 4"),
    ("arithmetic", "5 + 3 =", " 8"),
    ("sequence", "1, 2, 3, 4,", " 5"),
    ("sequence", "Monday, Tuesday, Wednesday,", " Thursday"),
    ("antonym", "The opposite of hot is", " cold"),
    ("antonym", "The opposite of up is", " down"),
    ("completion", "Roses are red, violets are", " blue"),
    ("completion", "The quick brown fox jumps over the lazy", " dog"),
    ("factual", "The capital of France is", " Paris"),
    ("factual", "The capital of Japan is", " Tokyo"),
    ("factual", "The capital of Germany is", " Berlin"),
    ("factual", "The largest planet in the solar system is", " Jupiter"),
]


def target_prefix(generated, target):
    """Strict token-prefix retrieval, never mere substring occurrence."""
    return bool(target) and list(generated[:len(target)]) == list(target)


def assess_case(before, edited, restored, case_id):
    """Keep the behavioral witness separate from exact restoration evidence."""
    key = case_id + "/rewrite"
    b, e, r = (next(x for x in s["probes"] if x["id"] == key)
               for s in (before, edited, restored))
    return {
        "absent_before": not b["target_prefix_match"],
        "retrieved_during_edit": e["target_prefix_match"],
        "absent_after": not r["target_prefix_match"],
        "behavioral_cycle": not b["target_prefix_match"] and e["target_prefix_match"] and not r["target_prefix_match"],
        "full_state_restored": restored["state_digest"] == before["state_digest"],
        "probe_logits_restored": restored["logits_digest"] == before["logits_digest"],
        "all_generations_restored": all(x["generated_ids"] == y["generated_ids"] for x, y in zip(before["probes"], restored["probes"], strict=True)),
    }


def control_comparison(before, after):
    pairs = [(x, y) for x, y in zip(before["probes"], after["probes"], strict=True) if x["kind"] == "control"]
    known = [(x, y) for x, y in pairs if x["target_prefix_match"]]
    return {
        "total": len(pairs),
        "baseline_target_matches": len(known),
        "baseline_matches_retained": sum(y["target_prefix_match"] for _, y in known),
        "target_matches": sum(y["target_prefix_match"] for _, y in pairs),
        "identical_generations": sum(x["generated_ids"] == y["generated_ids"] for x, y in pairs),
        "identical_first_tokens": sum(x["generated_ids"][0] == y["generated_ids"][0] for x, y in pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=ROOT / "experiments/novel_information_panel.json")
    parser.add_argument("--reference-run", type=Path, default=ROOT / "results/20261110_factual_chain")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("out must be a new directory")
    ref = json.loads((args.reference_run / "run.json").read_text())
    old_summary = json.loads((args.reference_run / "summary.json").read_text())
    if chain.digest_file(args.model_dir / "model.safetensors") != ref["model_sha256"]:
        raise ValueError("model does not match pinned factual pilot")
    upstream = args.upstream.resolve()
    upstream_hashes = {str(p.relative_to(upstream)): chain.digest_file(p) for p in sorted(upstream.rglob("*.py"))}
    if upstream_hashes != ref["upstream_files"]:
        raise ValueError("upstream compatibility copy differs from pinned factual pilot")
    panel = json.loads(args.panel.read_text())
    cases = panel["cases"]
    if len({r["case_id"] for r in cases}) != len(cases) or not cases:
        raise ValueError("nonempty unique case IDs required")

    import torch
    import transformers
    from transformers import AutoTokenizer, GPT2LMHeadModel

    torch.set_num_threads(1)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(upstream))
    stats = types.ModuleType("rome.layer_stats")
    def disabled_stats(*args, **kwargs):
        raise RuntimeError("use frozen declared covariance")
    stats.layer_stats = disabled_stats
    sys.modules["rome.layer_stats"] = stats
    rm = importlib.import_module("rome.rome_main")
    cu = importlib.import_module("rome.compute_u")
    mm = importlib.import_module("memit.memit_main")
    rm.CONTEXT_TEMPLATES_CACHE = ref["contexts"]
    mm.CONTEXT_TEMPLATES_CACHE = [ref["contexts"]]
    hp = {"rome": rm.ROMEHyperParams(**ref["hparams"]["rome"]),
          "memit": mm.MEMITHyperParams(**ref["hparams"]["memit"])}
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = GPT2LMHeadModel.from_pretrained(args.model_dir, local_files_only=True, use_safetensors=True, attn_implementation="eager").eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params = dict(model.named_parameters())
    out = args.out.resolve()
    out.mkdir(parents=True)
    work = out / "work"
    work.mkdir()
    cov, cov_info = {}, {}
    for layer in [4, 5, 6]:
        filename = f"work/covariance-{layer}.npz"
        cached = args.reference_run / filename
        if cached.exists():
            if chain.digest_file(cached) != old_summary["binary_artifacts"][filename]:
                raise ValueError("cached covariance hash mismatch")
            c = torch.from_numpy(chain.arrays(cached)["covariance"])
            origin = "verified previous base-model covariance"
        else:
            chunks = []
            def collect(module, inputs, output):
                chunks.append(inputs[0].detach().reshape(-1, 3072).double())
            handle = model.transformer.h[layer].mlp.c_proj.register_forward_hook(collect)
            try:
                with torch.no_grad():
                    for sentence in ref["calibration"]:
                        model(**tok(sentence, return_tensors="pt"), use_cache=False)
            finally:
                handle.remove()
            x = torch.cat(chunks)
            c = x.T @ x / len(x)
            c += .01 * float(c.diag().mean()) * torch.eye(3072, dtype=torch.float64)
            origin = "recomputed from declared base-model calibration"
        name = f"transformer.h.{layer}.mlp.c_proj"
        cov[name] = c
        np.savez(work / f"covariance-{layer}.npz", covariance=c.numpy())
        cov_info[name] = {"origin": origin, "tensor_digest": tensor_digest(c.numpy())}
    inverse = {name: torch.linalg.inv(c).float() for name, c in cov.items()}
    cu.get_inv_cov = lambda model, tok, name, *args: inverse[name]
    mm.get_cov = lambda model, tok, name, *args, **kwargs: cov[name]

    probes = []
    for case in cases:
        for kind in ["rewrite", "paraphrase"]:
            template = case["prompt"] if kind == "rewrite" else case["paraphrase"]
            probes.append({"id": case["case_id"] + "/" + kind, "kind": kind, "prompt": template.format(case["subject"]), "target": case["target_new"]["str"]})
    probes.extend({"id": f"control-{i}", "kind": "control", "category": category, "prompt": prompt, "target": target} for i, (category, prompt, target) in enumerate(CONTROLS))
    for probe in probes:
        probe["target_ids"] = tok.encode(probe["target"], add_special_tokens=False)
    max_new_tokens = max(8, max(len(p["target_ids"]) for p in probes))
    manifest = {
        "model": ref["model"], "model_revision": ref["model_revision"],
        "model_files": {p.name: chain.digest_file(p) for p in sorted(args.model_dir.iterdir()) if p.is_file()},
        "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": chain.digest_file(__file__), "receipt_helper_sha256": chain.digest_file(chain.__file__),
        "panel_sha256": chain.digest_file(args.panel), "panel": panel,
        "reference_manifest_sha256": chain.digest_file(args.reference_run / "run.json"),
        "upstream_commit": ref["upstream_commit"], "upstream_files": upstream_hashes,
        "hparams": ref["hparams"], "contexts": ref["contexts"], "calibration": ref["calibration"],
        "covariance": cov_info, "probes": probes, "max_new_tokens": max_new_tokens,
        "scope": "CPU FP32 single-request ROME and MEMIT, three fresh associations each; exact cloze retrieval is not a general knowledge or intelligence claim",
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def state():
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            h.update(name.encode())
            h.update(tensor.detach().numpy().tobytes())
        return h.hexdigest()

    def evaluate(label, baseline_logits=None):
        rows, first_logits = [], []
        with torch.no_grad():
            for probe in probes:
                prompt_ids = tok.encode(probe["prompt"], add_special_tokens=False)
                # Free greedy decoding sees only the question. No targets, history,
                # exemplars, cached KV state, retrieval, or inference-time hooks.
                ids = torch.tensor([prompt_ids])
                generated = []
                for step in range(max_new_tokens):
                    logits = model(input_ids=ids, use_cache=False).logits[0, -1]
                    if step == 0:
                        first_logits.append(logits.numpy().copy())
                    next_id = int(logits.argmax())
                    generated.append(next_id)
                    ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
                    if next_id == tok.eos_token_id:
                        break
                # Teacher forcing is a separate score, never the retrieval result.
                target_ids = probe["target_ids"]
                scored = torch.tensor([prompt_ids + target_ids[:-1]])
                logits = model(input_ids=scored, use_cache=False).logits[0, len(prompt_ids)-1:]
                log_probs = logits.double().log_softmax(-1)
                target_log_probs = log_probs[torch.arange(len(target_ids)), torch.tensor(target_ids)]
                row = dict(probe, generated=tok.decode(generated), generated_ids=generated,
                           target_prefix_match=target_prefix(generated, target_ids),
                           target_sequence_probability=float(target_log_probs.sum().exp()),
                           target_mean_nll=float(-target_log_probs.mean()))
                if baseline_logits is not None:
                    base_lp = torch.from_numpy(baseline_logits[len(rows)]).double().log_softmax(-1)
                    current_lp = torch.from_numpy(first_logits[-1]).double().log_softmax(-1)
                    row["first_token_kl_base_to_current"] = float((base_lp.exp() * (base_lp-current_lp)).sum())
                rows.append(row)
        matrix = np.stack(first_logits)
        np.savez(work / (label + "-logits.npz"), logits=matrix)
        result = {"state_digest": state(), "logits_digest": tensor_digest(matrix), "probes": rows}
        return result, matrix

    baseline, baseline_logits = evaluate("baseline")
    repeat, _ = evaluate("no-edit-repeat", baseline_logits)
    if baseline["state_digest"] != repeat["state_digest"] or baseline["logits_digest"] != repeat["logits_digest"] or any(x["generated_ids"] != y["generated_ids"] for x, y in zip(baseline["probes"], repeat["probes"], strict=True)):
        raise RuntimeError("no-edit repeat is nondeterministic")
    results = {"baseline": baseline, "no_edit_repeat": repeat, "rows": []}
    def save():
        (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    save()
    for algorithm in ["rome", "memit"]:
        for case in cases:
            label = algorithm + "-" + case["case_id"]
            if state() != baseline["state_digest"]:
                raise RuntimeError("previous exact rollback did not restore base; refusing next edit")
            names = [f"transformer.h.{layer}.mlp.c_proj.weight" for layer in hp[algorithm].layers]
            original = {name: params[name].detach().clone() for name in names}
            request = {k: case[k] for k in ["prompt", "subject", "target_new"]}
            torch.manual_seed(7)
            with (out / (label + "-optimization.txt")).open("w") as log, contextlib.redirect_stdout(log):
                factors = rm.execute_rome(model, tok, request, hp[algorithm]) if algorithm == "rome" else mm.execute_memit(model, tok, [request], hp[algorithm])
            if state() != baseline["state_digest"]:
                raise RuntimeError("upstream execute changed model before applying factors")
            if set(factors) != set(names):
                raise ValueError("unexpected edited layers")
            records = {}
            with torch.no_grad():
                for name, (key, value) in factors.items():
                    if algorithm == "rome":
                        key, value = key[:, None], value[:, None]
                    params[name].add_((key @ value.T).float())
                    records[name] = (original[name].numpy().T.copy(), params[name].detach().numpy().T.copy(), value.detach().numpy(), key.detach().numpy())
                    np.savez(work / (label + "-" + name + ".npz"), **dict(zip(["original", "edited", "left", "right"], records[name], strict=True)))
            edited, _ = evaluate(label + "-edited", baseline_logits)
            row = {"algorithm": algorithm, "case_id": case["case_id"], "edited": edited,
                   "edited_controls": control_comparison(baseline, edited), "restorations": []}
            results["rows"].append(row)
            save()
            # Each restoration receives the identical edited state. Only edited
            # layers are reapplied between arms; no base-model reset is used.
            for method in ["checkpoint", "delta", "delta_exact"]:
                with torch.no_grad():
                    for name, (_, weight, _, _) in records.items():
                        params[name].copy_(torch.from_numpy(weight.T.copy()))
                if state() != edited["state_digest"]:
                    raise RuntimeError("restoration arms did not start from identical edited state")
                costs = []
                for index, (name, data) in enumerate(reversed(list(records.items()))):
                    original_array, edited_array, left, right = data
                    prefix = label + "-" + method + f"-{index}"
                    record_path = work / (prefix + "-record.npz")
                    record_path.write_bytes(chain.pack_record(original_array, edited_array, left, right, method))
                    edited_path = work / (prefix + "-edited.npz")
                    np.savez(edited_path, weight=edited_array)
                    restored_path = work / (prefix + "-restored.npz")
                    process = subprocess.run([sys.executable, "-I", chain.__file__, "--undo", "--edited", str(edited_path), "--record", str(record_path), "--out", str(restored_path)], capture_output=True, text=True, env=dict(os.environ, **THREAD_ENV), timeout=120)
                    if process.returncode:
                        raise RuntimeError(process.stderr)
                    costs.append(dict(json.loads(process.stdout), layer=name, record_bytes=record_path.stat().st_size, record_sha256=chain.digest_file(record_path)))
                    with torch.no_grad():
                        params[name].copy_(torch.from_numpy(chain.arrays(restored_path)["weight"].T.copy()))
                restored, _ = evaluate(label + "-" + method + "-restored", baseline_logits)
                assessment = assess_case(baseline, edited, restored, case["case_id"])
                row["restorations"].append({"method": method, "assessment": assessment, "restored": restored, "controls": control_comparison(baseline, restored), "costs": costs})
                save()
                print(label, method, assessment, flush=True)
                if method != "delta" and not all(assessment[k] for k in ["full_state_restored", "probe_logits_restored", "all_generations_restored"]):
                    raise RuntimeError("exact restoration failed; results retained")
    summary = {"status": "complete", "edits": len(results["rows"]),
               "behavioral_cycles": {method: sum(r["assessment"]["behavioral_cycle"] for row in results["rows"] for r in row["restorations"] if r["method"] == method) for method in ["checkpoint", "delta", "delta_exact"]},
               "exact_full_state_restorations": sum(r["assessment"]["full_state_restored"] for row in results["rows"] for r in row["restorations"] if r["method"] == "delta_exact"),
               "binary_artifacts": {str(p.relative_to(out)): chain.digest_file(p) for p in sorted(work.glob("*.npz"))}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
