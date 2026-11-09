"""Fresh-process time, serialized-size, and process high-water RSS measurements.

Fixtures are synthetic fixed-target ROME writes with identity covariance. Forward
editing and input loading are outside the measured region. Undo workers receive
only edited weights and a serialized record. No full-model performance claim is
made by this benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np

from .core import EditResult, LowRankDelta, PRECISIONS, RomeSpec, quantize
from .receipts import make_receipt, restore_from_receipt, serialize_receipt, tensor_digest
from .roundoff_experiments import _checkpoint_arrays


METHODS = ("checkpoint", "delta", "delta_exact")
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


@dataclass(frozen=True)
class CostConfig:
    seeds: tuple[int, ...] = (7, 23, 61)
    shapes: tuple[tuple[int, int], ...] = ((64, 128), (256, 512), (1024, 2048))
    precisions: tuple[str, ...] = ("float32", "float16", "bfloat16")
    alpha: float = 0.95
    repeats: int = 3
    worker_timeout_seconds: float = 120.0

    def __post_init__(self):
        if not self.seeds or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonempty nonnegative integers")
        if not self.shapes or any(len(s) != 2 or any(type(n) is not int or n < 2 for n in s) for s in self.shapes):
            raise ValueError("shapes must be output_dim x input_dim, each at least two")
        if not self.precisions or any(p not in PRECISIONS for p in self.precisions):
            raise ValueError("unsupported or empty precisions")
        if any(len(set(items)) != len(items) for items in (self.seeds, self.shapes, self.precisions)):
            raise ValueError("duplicate seeds, shapes, or precisions are not independent observations")
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        if type(self.repeats) is not int or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if not np.isfinite(self.worker_timeout_seconds) or self.worker_timeout_seconds <= 0:
            raise ValueError("worker timeout must be positive and finite")


def identity_rome_fixture(seed: int, shape: tuple[int, int], precision: str,
                          alpha: float) -> tuple[np.ndarray, EditResult, dict]:
    """Precompute q=k/(k.T k); avoid timing a dense solve for known C=I."""
    m, n = shape
    rng = np.random.default_rng(seed)
    original = quantize(rng.normal(scale=1 / np.sqrt(n), size=shape), precision)
    key = rng.normal(size=n)
    key /= np.linalg.norm(key)
    old_response = original.astype(np.float64) @ key
    value = old_response + rng.normal(scale=0.7, size=m)
    direction = key / (key @ key)
    spec = RomeSpec(key, value, direction, alpha)
    delta = LowRankDelta((alpha * (value - old_response))[:, None], direction[:, None])
    edited = quantize(original.astype(np.float64) + delta.matrix(), precision)
    result = EditResult(edited, delta, spec, {"algorithm": "identity_covariance_rome"})
    metadata = {
        "original_tensor_sha256": tensor_digest(original),
        "edited_tensor_sha256": tensor_digest(result.updated),
        "original_physical_bytes": original.nbytes,
        "original_logical_bytes": original.size * (2 if precision == "bfloat16" else original.itemsize),
        "factor_bytes": delta.nbytes,
        "write_residual_fraction": float(np.linalg.norm(edited.astype(np.float64) @ key - value)
                                         / np.linalg.norm(old_response - value)),
    }
    return original, result, metadata


def checkpoint_record(original: np.ndarray, edited: np.ndarray, precision: str) -> dict:
    payload = _checkpoint_arrays(original, precision)
    meta = json.loads(str(payload["__meta__"].item()))
    meta["edited_tensor_sha256"] = tensor_digest(edited)
    payload["__meta__"] = np.array(json.dumps(meta, sort_keys=True))
    return payload


def restore_checkpoint(edited: np.ndarray, payload: dict) -> np.ndarray:
    meta = json.loads(str(payload["__meta__"].item()))
    if meta.get("format") != "logical_checkpoint_v1":
        raise ValueError("unsupported benchmark checkpoint")
    if tensor_digest(edited) != meta.get("edited_tensor_sha256"):
        raise ValueError("checkpoint does not match edited tensor")
    stored = payload["weight"]
    if meta["precision"] == "bfloat16":
        restored = (stored.astype("<u4") << 16).view("<f4")
    else:
        restored = stored.copy()
    if list(restored.shape) != meta["shape"] or tensor_digest(restored) != meta["tensor_sha256"]:
        raise ValueError("checkpoint digest mismatch")
    return restored


def _rss_bytes() -> int:
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    if sys.platform.startswith("linux"):
        return int(value * 1024)
    raise RuntimeError("RSS units are implemented only for macOS and Linux")


def _npz_bytes(payload: dict) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    return stream.getvalue()


def _load_arrays(path: Path | io.BytesIO) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def _worker_build(inputs: Path, record: Path, method: str, precision: str, alpha: float) -> dict:
    arrays = _load_arrays(inputs)
    original = arrays.pop("original")
    edit = EditResult(arrays["edited"], LowRankDelta(arrays["left"], arrays["right"]),
                      RomeSpec(arrays["key"], arrays["value"], arrays["direction"], alpha), {})
    del arrays
    gc.collect()
    before = _rss_bytes()
    start = perf_counter()
    payload = (checkpoint_record(original, edit.updated, precision) if method == "checkpoint"
               else make_receipt(original, edit, editor="rome", recovery=method, precision=precision))
    constructed = perf_counter()
    blob = _npz_bytes(payload) if method == "checkpoint" else serialize_receipt(payload)
    finished = perf_counter()
    peak = _rss_bytes()
    # Disk output and the benchmark-only digest are outside the measured region.
    record.write_bytes(blob)
    return {"construct_seconds": constructed - start, "serialize_seconds": finished - constructed,
            "total_seconds": finished - start, "serialized_bytes": len(blob),
            "record_sha256": hashlib.sha256(blob).hexdigest(),
            "baseline_peak_rss_bytes": before, "process_peak_rss_bytes": peak,
            "high_water_growth_bytes": peak - before}


def _worker_undo(edited_path: Path, record: Path, method: str) -> dict:
    # The worker has no original input parameter and never opens a fixture file.
    edited = _load_arrays(edited_path)["weight"]
    blob = record.read_bytes()
    gc.collect()
    before = _rss_bytes()
    start = perf_counter()
    payload = _load_arrays(io.BytesIO(blob))
    decoded = perf_counter()
    restored = (restore_checkpoint(edited, payload) if method == "checkpoint"
                else restore_from_receipt(edited, payload))
    finished = perf_counter()
    peak = _rss_bytes()
    return {"decode_seconds": decoded - start, "restore_seconds": finished - decoded,
            "total_seconds": finished - start, "restored_tensor_sha256": tensor_digest(restored),
            "baseline_peak_rss_bytes": before, "process_peak_rss_bytes": peak,
            "high_water_growth_bytes": peak - before}


def _run_worker(arguments: list[str], timeout: float) -> dict:
    command = [sys.executable, "-I", "-m", "reversible_edit.cost_experiments", *arguments]
    try:
        process = subprocess.run(command, text=True, capture_output=True,
                                 env=dict(os.environ, **THREAD_ENV), timeout=timeout)
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        return {"status": "timeout", "stdout": decode(error.stdout), "stderr": decode(error.stderr)}
    if process.returncode:
        return {"status": "error", "returncode": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr}
    try:
        data = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {"status": "invalid_worker_output", "stdout": process.stdout, "stderr": process.stderr}
    return {"status": "ok", **data, "stderr": process.stderr}


def summarize_costs(rows: list[dict]) -> dict:
    checkpoints = {(r["fixture_id"], r["repeat"]): r["build"]["serialized_bytes"]
                   for r in rows if r["method"] == "checkpoint" and r["status"] == "ok"}
    groups = {}
    for row in rows:
        key = (row["shape"], row["precision"], row["method"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (shape, precision, method), group in sorted(groups.items()):
        valid = [row for row in group if row["status"] == "ok"]
        summary = {"shape": shape, "precision": precision, "method": method,
                   "observations": len(group), "successful_workers": len(valid),
                   "failed_observations": len(group) - len(valid),
                   "bitwise_restored": sum(row["bitwise_restored"] for row in valid)}
        if valid:
            def stats(values):
                return {"median": statistics.median(values), "min": min(values), "max": max(values)}
            summary["serialized_bytes"] = stats([r["build"]["serialized_bytes"] for r in valid])
            ratios = [r["build"]["serialized_bytes"] / checkpoints[(r["fixture_id"], r["repeat"])]
                      for r in valid if (r["fixture_id"], r["repeat"]) in checkpoints]
            summary["receipt_to_checkpoint"] = stats(ratios) if ratios else None
            for phase in ("build", "undo"):
                for metric in ("total_seconds", "process_peak_rss_bytes", "high_water_growth_bytes"):
                    summary[f"{phase}_{metric}"] = stats([r[phase][metric] for r in valid])
        summaries.append(summary)
    expected_exact = [r for r in rows if r["method"] in ("checkpoint", "delta_exact")]
    return {"observations": len(rows), "failed_observations": sum(r["status"] != "ok" for r in rows),
            "exact_observations": len(expected_exact),
            "exact_bitwise_restored": sum(r.get("bitwise_restored", False) for r in expected_exact),
            "groups": summaries}


def _report(summary: dict, config: CostConfig) -> str:
    lines = ["# Rollback resource-cost pilot", "",
             f"Synthetic identity-covariance ROME; alpha={config.alpha:g}. "
             f"{len(config.seeds)} seeds × {config.repeats} timing repeats per condition. "
             "All figures below are medians; raw observations and min/max are retained.", "",
             f"Completed observations: {summary['observations']}; worker failures: "
             f"{summary['failed_observations']}. Checkpoint/exact bitwise restores: "
             f"{summary['exact_bitwise_restored']}/{summary['exact_observations']}.", "",
             "| Shape | Precision | Method | Record KiB | /checkpoint | Build ms | Undo ms | Build peak MiB | Undo peak MiB | Bitwise |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["groups"]:
        if not row["successful_workers"]:
            lines.append(f"| {row['shape']} | {row['precision']} | {row['method']} | failed | — | — | — | — | — | 0/{row['observations']} |")
            continue
        ratio = row["receipt_to_checkpoint"]
        ratio_text = f"{ratio['median']:.1%}" if ratio else "unavailable"
        lines.append(f"| {row['shape']} | {row['precision']} | {row['method']} | "
                     f"{row['serialized_bytes']['median']/1024:.1f} | {ratio_text} | "
                     f"{row['build_total_seconds']['median']*1000:.2f} | "
                     f"{row['undo_total_seconds']['median']*1000:.2f} | "
                     f"{row['build_process_peak_rss_bytes']['median']/2**20:.1f} | "
                     f"{row['undo_process_peak_rss_bytes']['median']/2**20:.1f} | "
                     f"{row['bitwise_restored']}/{row['observations']} |")
    lines += ["", "## Measurement boundary", "",
              "Each build and undo uses a fresh process, one numerical operation, and single-thread BLAS settings. "
              "Process startup, input file loading, fixture generation, forward editing, and record file writing are excluded from time. "
              "Build includes record construction, integrity hashes, and NPZ encoding; undo includes NPZ decoding and restoration checks.", "",
              "RSS is the whole process lifetime high-water mark at the end of the measured region, including imports and input preparation. "
              "The pre-operation high-water mark and its growth are also recorded. Growth is not an allocation count or an isolated operation peak; "
              "a zero growth does not mean zero allocations. BF16 weights occupy float32 arrays in memory but uint16 words in checkpoint storage.", "",
              "The checkpoint is compressed and bound to the edited state and original digest. "
              "Undo processes load only that edited state and the selected record; the evaluator compares returned digests. "
              "Delta records may not restore bits. Exact records retain old-state information. "
              "These measurements do not establish native GPU speed, end-to-end LLM memory savings, or factual editing quality.", "",
              "`observations.jsonl` preserves every completed observation, including errors. `summary.json` contains min/median/max. "
              "`run.json` records configuration, environment, and source hashes; `fixtures.json` records input hashes and edit strength. "
              "Generated binary fixtures/records remain in ignored `work/` and can be regenerated from the configuration.", ""]
    return "\n".join(lines)


def run_cost_experiment(out: Path, config: CostConfig, *, provenance: dict | None = None,
                        progress=None) -> dict:
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        raise RuntimeError("cost measurements currently support macOS and Linux")
    out = Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Use an empty/new result directory to preserve evidence")
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir()
    with io.StringIO() as capture:
        from contextlib import redirect_stdout
        with redirect_stdout(capture):
            np.show_config()
        numpy_config = capture.getvalue()
    run = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
           "config": asdict(config), "methods": METHODS, "provenance": provenance,
           "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                           "machine": platform.machine(), "cpu_count": os.cpu_count(),
                           "numpy_config": numpy_config, "worker_thread_environment": THREAD_ENV},
           "protocol": "docs/COSTS.md", "planned_observations": len(config.seeds) * len(config.shapes)
           * len(config.precisions) * config.repeats * len(METHODS)}
    (out / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    rows, fixtures = [], []
    started = perf_counter()
    with (out / "observations.jsonl").open("x") as stream:
        for shape in config.shapes:
            for precision in config.precisions:
                for seed in config.seeds:
                    fixture_id = len(fixtures)
                    directory = work / f"fixture-{fixture_id:03d}"
                    directory.mkdir()
                    original, edit, metadata = identity_rome_fixture(seed, shape, precision, config.alpha)
                    inputs, edited_path = directory / "inputs.npz", directory / "edited.npz"
                    np.savez(inputs, original=original, edited=edit.updated,
                             left=edit.delta.left, right=edit.delta.right,
                             key=edit.spec.key, value=edit.spec.value, direction=edit.spec.direction)
                    np.savez(edited_path, weight=edit.updated)
                    metadata.update({"fixture_id": fixture_id, "seed": seed, "shape": list(shape),
                                     "precision": precision, "alpha": config.alpha,
                                     "input_file_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest()})
                    fixtures.append(metadata)
                    (out / "fixtures.json").write_text(json.dumps(fixtures, indent=2, allow_nan=False) + "\n")
                    del original, edit
                    for repeat in range(config.repeats):
                        offset = (fixture_id + repeat) % len(METHODS)
                        for method in METHODS[offset:] + METHODS[:offset]:
                            record = directory / f"{method}-{repeat}.npz"
                            build = _run_worker(["build", "--inputs", str(inputs), "--record", str(record),
                                                 "--method", method, "--precision", precision,
                                                 "--alpha", str(config.alpha)], config.worker_timeout_seconds)
                            undo = (_run_worker(["undo", "--edited", str(edited_path), "--record", str(record),
                                                 "--method", method], config.worker_timeout_seconds)
                                    if build["status"] == "ok" else {"status": "skipped_after_build_failure"})
                            ok = build["status"] == undo["status"] == "ok"
                            row = {"fixture_id": fixture_id, "seed": seed,
                                   "shape": f"{shape[0]}x{shape[1]}", "precision": precision,
                                   "alpha": config.alpha, "repeat": repeat, "method": method,
                                   "status": "ok" if ok else "worker_failure", "build": build, "undo": undo,
                                   "bitwise_restored": ok and undo["restored_tensor_sha256"] == metadata["original_tensor_sha256"]}
                            rows.append(row)
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                            stream.flush()
                    if progress:
                        progress(f"{len(rows)}/{run['planned_observations']} observations: "
                                 f"{shape[0]}x{shape[1]} {precision} seed={seed}")
    summary = summarize_costs(rows)
    summary.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                    "runtime_seconds": perf_counter() - started,
                    "status": "complete" if summary["failed_observations"] == 0
                    and summary["exact_bitwise_restored"] == summary["exact_observations"] else "failed"})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "RESULTS.md").write_text(_report(summary, config))
    return summary


def _worker_main() -> None:
    parser = argparse.ArgumentParser(description="Internal resource measurement worker")
    sub = parser.add_subparsers(dest="phase", required=True)
    build = sub.add_parser("build")
    build.add_argument("--inputs", type=Path, required=True)
    build.add_argument("--precision", choices=PRECISIONS, required=True)
    build.add_argument("--alpha", type=float, required=True)
    undo = sub.add_parser("undo")
    undo.add_argument("--edited", type=Path, required=True)
    for command in (build, undo):
        command.add_argument("--record", type=Path, required=True)
        command.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    result = (_worker_build(args.inputs, args.record, args.method, args.precision, args.alpha)
              if args.phase == "build" else _worker_undo(args.edited, args.record, args.method))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    _worker_main()
