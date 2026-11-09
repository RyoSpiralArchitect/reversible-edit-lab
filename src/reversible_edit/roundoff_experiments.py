"""v0.2: finite-storage correction budgets, repeated undo, and query fidelity.

Original states may be read while a write receipt is constructed. Restoration
receives only edited weights and that receipt. This is logged recovery, not
blind inference of a model's unknown history.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from time import perf_counter
import numpy as np

from .core import memit_edit, quantize, rome_edit, storage_nbytes
from .metrics import recovery_metrics, relative_frobenius
from .receipts import make_receipt, restore_from_receipt, serialize_receipt, tensor_digest


@dataclass(frozen=True)
class RefinementConfig:
    seeds: tuple[int, ...] = (7, 23, 61)
    precisions: tuple[str, ...] = ("float32", "float16", "bfloat16")
    shapes: tuple[tuple[int, int], ...] = ((12, 24), (64, 128))
    alphas: tuple[float, ...] = (0.0, 0.5, 0.95, 0.999, 1.0)
    sparse_budget_fractions: tuple[float, ...] = (0.05, 0.10, 0.25)
    cycle_count: int = 512
    active_rounds: int = 6

    def __post_init__(self):
        if not self.seeds or any(not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("nonempty nonnegative integer seeds required")
        if not self.precisions or any(p not in ("float64", "float32", "float16", "bfloat16") for p in self.precisions):
            raise ValueError("unsupported or empty precision sequence")
        if not self.shapes or any(m < 2 or n < 2 for m, n in self.shapes):
            raise ValueError("matrix dimensions must be at least two")
        if not self.alphas or any(not np.isfinite(a) or not 0 <= a <= 1 for a in self.alphas):
            raise ValueError("alphas must lie in [0, 1]")
        if any(not np.isfinite(f) or not 0 <= f <= 1 for f in self.sparse_budget_fractions):
            raise ValueError("sparse fractions must lie in [0, 1]")
        if self.cycle_count < 1 or self.active_rounds < 1:
            raise ValueError("cycle count and active rounds must be positive")


def _fixture(seed: int, m: int, n: int):
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=1/np.sqrt(n), size=(m, n))
    x = rng.normal(size=(n, max(128, 2*n)))
    c = x @ x.T / x.shape[1] + .1*np.eye(n)
    probes = rng.normal(size=(n, 128))
    probes /= np.linalg.norm(probes, axis=0, keepdims=True)
    return rng, w, c, probes


def _key(rng, n):
    k = rng.normal(size=n)
    return k/np.linalg.norm(k)


def _checkpoint_arrays(weight: np.ndarray, precision: str) -> dict[str, np.ndarray]:
    """A real compressed baseline, with bf16 stored as logical uint16 words."""
    if precision == "bfloat16":
        stored = (np.asarray(weight, dtype="<f4").view("<u4") >> 16).astype("<u2")
    else:
        stored = np.asarray(weight, dtype={"float64":"<f8", "float32":"<f4", "float16":"<f2"}[precision])
    metadata = {"format":"logical_checkpoint_v1", "precision":precision,
                "shape":list(weight.shape), "tensor_sha256":tensor_digest(weight)}
    return {"weight":stored, "__meta__":np.array(json.dumps(metadata, sort_keys=True))}


def _modes(editor: str, alpha: float | None, config: RefinementConfig):
    yield "delta", "delta", None
    yield "delta_exact", "delta_exact", None
    for fraction in config.sparse_budget_fractions:
        yield f"delta_sparse_{fraction:g}", "delta_sparse", fraction
    if editor == "memit" or alpha < 1:
        yield "analytic", "analytic", None
        yield "analytic_exact", "analytic_exact", None


def _checkpoint_nbytes(weight: np.ndarray, precision: str) -> int:
    stream=io.BytesIO()
    np.savez_compressed(stream,**_checkpoint_arrays(weight,precision))
    return len(stream.getvalue())


def _meta(receipt):
    return json.loads(str(receipt["__meta__"].item()))


def _storage_fields(receipt, original, precision):
    metadata = _meta(receipt)
    checkpoint_bytes = _checkpoint_nbytes(original, precision)
    receipt_bytes = len(serialize_receipt(receipt))
    return {
        "receipt_npz_bytes": receipt_bytes,
        "checkpoint_npz_bytes": checkpoint_bytes,
        "receipt_to_compressed_checkpoint": receipt_bytes/checkpoint_bytes,
        "checkpoint_raw_storage_bytes": storage_nbytes(original, precision),
        "base_numeric_payload_bytes": metadata.get("base_numeric_payload_bytes", 0),
        "correction_packet_bytes": metadata.get("correction_packet_bytes", 0),
        "correction_payload_bytes": metadata.get("correction_payload_bytes", 0),
        "correction_metadata_bytes": metadata.get("correction_metadata_bytes", 0),
        "recovery_declared_exact": metadata.get("recovery_exact", False),
    }


def run_tradeoff_sweep(config: RefinementConfig) -> list[dict]:
    rows = []
    for seed in config.seeds:
        for m,n in config.shapes:
            rng, unrounded, covariance, probes = _fixture(seed, m, n)
            key = _key(rng,n)
            offset = rng.normal(scale=.7,size=m)
            keys = np.column_stack([_key(rng,n) for _ in range(4)])
            offsets = rng.normal(scale=.7,size=(m,4))
            next_key = _key(rng,n)
            next_offset = rng.normal(scale=.3,size=m)
            for precision in config.precisions:
                original = quantize(unrounded,precision)
                value = original.astype(float)@key+offset
                values = original.astype(float)@keys+offsets
                next_value = original.astype(float)@next_key+next_offset
                expected_next = rome_edit(original,next_key,next_value,covariance,alpha=.6,precision=precision).updated
                writes = [("rome",a,None) for a in config.alphas]+[("memit",None,s) for s in (.1,1.0)]
                for editor,alpha,scale in writes:
                    if editor=="rome":
                        result=rome_edit(original,key,value,covariance,alpha=alpha,precision=precision)
                        before=np.linalg.norm(original.astype(float)@key-value)
                        after=np.linalg.norm(result.updated.astype(float)@key-value)
                    else:
                        result=memit_edit(original,keys,values,scale*covariance,precision=precision)
                        before=np.linalg.norm(original.astype(float)@keys-values)
                        after=np.linalg.norm(result.updated.astype(float)@keys-values)
                    for name,mode,fraction in _modes(editor,alpha,config):
                        budget=None if fraction is None else int(fraction*storage_nbytes(original,precision))
                        start=perf_counter()
                        receipt=make_receipt(original,result,editor=editor,recovery=mode,precision=precision,
                                             patch_budget_bytes=budget)
                        build_seconds=perf_counter()-start
                        start=perf_counter()
                        recovered=restore_from_receipt(result.updated,receipt)
                        undo_seconds=perf_counter()-start
                        future=rome_edit(recovered,next_key,next_value,covariance,alpha=.6,precision=precision).updated
                        row={"seed":seed,"precision":precision,"output_dim":m,"input_dim":n,
                             "editor":editor,"alpha":alpha,"covariance_scale":scale,"method":name,
                             "patch_payload_budget_bytes":budget,"patch_payload_budget_fraction":fraction,
                             "write_residual_fraction":float(after/before),
                             "bitwise_restored":tensor_digest(recovered)==tensor_digest(original),
                             "next_update_bitwise_equal":tensor_digest(future)==tensor_digest(expected_next),
                             "next_update_relative_error":relative_frobenius(future,expected_next),
                             "record_creation_seconds":build_seconds,"restore_seconds":undo_seconds}
                        row.update(_storage_fields(receipt,original,precision))
                        row.update(recovery_metrics(recovered,original,probes))
                        rows.append(row)
    return rows


def run_corrected_cycles(config: RefinementConfig) -> list[dict]:
    """Reuse a single temporary record, discarding it after every undo."""
    rows=[]
    m,n=12,24
    checkpoints=sorted({1,min(8,config.cycle_count),min(32,config.cycle_count),
                        min(128,config.cycle_count),config.cycle_count})
    modes=(("delta","delta",None),("delta_exact","delta_exact",None),
           ("delta_sparse_0.05","delta_sparse",.05),("delta_sparse_0.25","delta_sparse",.25),
           ("analytic","analytic",None),("analytic_exact","analytic_exact",None))
    for seed in config.seeds:
        rng,unrounded,covariance,probes=_fixture(seed+1000,m,n)
        keys=[_key(rng,n) for _ in range(config.cycle_count)]
        offsets=rng.normal(scale=.7,size=(config.cycle_count,m))
        next_key=_key(rng,n)
        next_offset=rng.normal(scale=.3,size=m)
        for precision in config.precisions:
            original=quantize(unrounded,precision)
            original_digest=tensor_digest(original)
            targets=[original.astype(float)@k+o for k,o in zip(keys,offsets)]
            next_value=original.astype(float)@next_key+next_offset
            expected_next=rome_edit(original,next_key,next_value,covariance,alpha=.6,precision=precision).updated
            checkpoint_bytes=_checkpoint_nbytes(original,precision)
            for name,mode,fraction in modes:
                live=original.copy()
                total_bytes=max_bytes=exact_steps=0
                total_patch_payload=0
                for step,(key,value) in enumerate(zip(keys,targets),1):
                    pre_digest=tensor_digest(live)
                    budget=None if fraction is None else int(fraction*storage_nbytes(live,precision))
                    result=rome_edit(live,key,value,covariance,alpha=.95,precision=precision)
                    receipt=make_receipt(live,result,editor="rome",recovery=mode,precision=precision,
                                         patch_budget_bytes=budget)
                    length=len(serialize_receipt(receipt))
                    total_bytes+=length; max_bytes=max(max_bytes,length)
                    total_patch_payload+=_meta(receipt).get("correction_payload_bytes",0)
                    live=restore_from_receipt(result.updated,receipt)
                    exact_steps+=int(tensor_digest(live)==pre_digest)
                    if step in checkpoints:
                        future=rome_edit(live,next_key,next_value,covariance,alpha=.6,precision=precision).updated
                        row={"seed":seed,"precision":precision,"method":name,"alpha":.95,"cycle":step,
                             "bitwise_restored_to_start":tensor_digest(live)==original_digest,
                             "exact_single_steps":exact_steps,
                             "mean_receipt_npz_bytes":total_bytes/step,"max_live_receipt_npz_bytes":max_bytes,
                             "total_generated_receipt_bytes":total_bytes,"checkpoint_npz_bytes":checkpoint_bytes,
                             "mean_patch_payload_bytes":total_patch_payload/step,
                             "next_update_bitwise_equal":tensor_digest(future)==tensor_digest(expected_next),
                             "next_update_relative_error":relative_frobenius(future,expected_next)}
                        row.update(recovery_metrics(live,original,probes))
                        rows.append(row)
    return rows


def run_receipt_chain(seed: int=7,precision: str="bfloat16",steps: int=8) -> dict:
    """Serialize each receipt and recover a committed history in strict LIFO order.

Only the final weight and serialized records go to the reverse loop. The initial
matrix is retained by the evaluator for the final comparison.
    """
    rng,base,covariance,probes=_fixture(seed+3000,16,32)
    original=quantize(base,precision)
    live=original.copy(); records=[]; expected_digests=[]
    for _ in range(steps):
        expected_digests.append(tensor_digest(live))
        k=_key(rng,32); v=rng.normal(size=16)
        edit=rome_edit(live,k,v,covariance,alpha=1.0,precision=precision)
        record=make_receipt(live,edit,editor="rome",recovery="delta_exact",precision=precision)
        records.append(serialize_receipt(record)); live=edit.updated
    reversed_digests=[]
    for blob in reversed(records):
        with np.load(io.BytesIO(blob),allow_pickle=False) as archive:
            payload={key:np.array(archive[key]) for key in archive.files}
        live=restore_from_receipt(live,payload)
        reversed_digests.append(tensor_digest(live))
    return {"seed":seed,"precision":precision,"steps":steps,"editor":"hard_rome",
            "every_intermediate_digest_matches":reversed_digests==list(reversed(expected_digests)),
            "final_bitwise_restored":tensor_digest(live)==tensor_digest(original),
            "total_receipt_npz_bytes":sum(map(len,records)),
            "weight_relative_error":relative_frobenius(live,original)}


def run_refinement(config: RefinementConfig, *, include_active: bool=True, progress=None) -> dict:
    start=perf_counter()
    if progress: progress("Measuring correction payload and compressed receipt costs")
    tradeoffs=run_tradeoff_sweep(config)
    if progress: progress("Measuring repeated rollback and subsequent updates")
    cycles=run_corrected_cycles(config)
    active=[]
    if include_active:
        from .active import run_active_experiment
        if progress: progress("Measuring corrected active-acquisition paths")
        for seed in config.seeds:
            for precision in config.precisions:
                active.append(run_active_experiment(seed,precision,n_rounds=config.active_rounds,
                    protocols=("checkpoint","factor","analytic","factor_exact","analytic_exact","factor_sparse"),
                    patch_budget_fraction=.25))
    chains=[run_receipt_chain(config.seeds[0],p) for p in config.precisions]
    return {"schema_version":2,"package_version":"0.2.0","config":asdict(config),
            "scope":"Synthetic local write kernels; compressed old-state correction is explicitly retained information.",
            "precision_model":"float64 work/metadata; weights stored at declared precision; bf16 emulated.",
            "budget_policy":"Sparse budget applies to correction data payload; report includes full patch metadata and full compressed NPZ receipt bytes.",
            "comparison_policy":"Snapshots and recovery receipts are both actually NPZ-compressed. Dense MEMIT covariance is included. No amortization is assumed.",
            "memory_scope":"Evaluator holds reference and temporary copies; disk-record size is not peak process memory.",
            "hypothesis_scope":"Numerical mechanism only: the v0.1 expected-entropy acquisition objective is unchanged.",
            "tradeoff_rows":tradeoffs,"cycle_rows":cycles,"active_runs":active,"receipt_chains":chains,
            "runtime_seconds":perf_counter()-start}
