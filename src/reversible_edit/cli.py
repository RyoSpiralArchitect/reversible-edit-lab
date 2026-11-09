"""Command-line entry points for reproducible runs and portable receipts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

from .experiments import ExperimentConfig, run_all
from .receipts import edit_npz, undo_npz


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _manifest(root: Path) -> dict:
    sources = {}
    for pattern in ("src/**/*.py", "tests/*.py", "pyproject.toml", "run_experiments.py", "references.bib"):
        for file in sorted(root.glob(pattern)):
            sources[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(), "source_sha256": sources}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run synthetic kernel, repetition, acquisition, and selective-undo experiments")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 61])
    run.add_argument("--precisions", nargs="+", choices=["float64", "float32", "float16", "bfloat16"],
                     default=["float64", "float32", "float16", "bfloat16"])
    run.add_argument("--cycles", type=int, default=128)
    run.add_argument("--active-rounds", type=int, default=6)
    run.add_argument("--no-active", action="store_true")
    run.add_argument("--plots", action="store_true", help="Also create PNG figures (requires matplotlib)")
    refine = sub.add_parser("refine", help="Run v0.2 compressed roundoff corrections and byte-cost comparisons")
    refine.add_argument("--out", type=Path, required=True)
    refine.add_argument("--seeds", type=int, nargs="+", default=[7,23,61])
    refine.add_argument("--precisions", nargs="+", choices=["float64","float32","float16","bfloat16"],
                        default=["float32","float16","bfloat16"])
    refine.add_argument("--cycles", type=int, default=512)
    refine.add_argument("--active-rounds", type=int, default=6)
    refine.add_argument("--no-active", action="store_true")
    refine.add_argument("--plots", action="store_true")
    costs = sub.add_parser("costs", help="Measure rollback record time and process peak RSS in fresh workers")
    costs.add_argument("--out", type=Path, required=True)
    costs.add_argument("--shapes", nargs="+", default=["64x128", "256x512", "1024x2048"])
    costs.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 61])
    costs.add_argument("--precisions", nargs="+", choices=["float64", "float32", "float16", "bfloat16"],
                       default=["float32", "float16", "bfloat16"])
    costs.add_argument("--alpha", type=float, default=0.95)
    costs.add_argument("--repeats", type=int, default=3)
    costs.add_argument("--worker-timeout", type=float, default=120.0)
    edit = sub.add_parser("edit", help="Edit a supplied NPZ weight matrix and write a compact recovery receipt")
    edit.add_argument("--input", type=Path, required=True)
    edit.add_argument("--output", type=Path, required=True)
    edit.add_argument("--receipt", type=Path, required=True)
    edit.add_argument("--editor", choices=["rome", "memit"], default="rome")
    edit.add_argument("--recovery", choices=["delta", "analytic", "old_response", "svd", "delta_exact", "analytic_exact", "delta_sparse"], default="delta")
    edit.add_argument("--alpha", type=float, default=0.75)
    edit.add_argument("--precision", choices=["float64", "float32", "float16", "bfloat16"], default="float32")
    edit.add_argument("--drop-rank", type=int, default=1)
    edit.add_argument("--patch-budget-bytes", type=int, default=None,
                      help="Sparse correction data-payload budget; metadata is counted separately")
    undo = sub.add_parser("undo", help="Restore a paired edited NPZ tensor using only its receipt")
    undo.add_argument("--edited", type=Path, required=True)
    undo.add_argument("--receipt", type=Path, required=True)
    undo.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "edit":
        print(json.dumps(edit_npz(args.input, args.output, args.receipt, editor=args.editor,
                                 recovery=args.recovery, alpha=args.alpha,
                                 precision=args.precision, drop_rank=args.drop_rank,
                                 patch_budget_bytes=args.patch_budget_bytes), indent=2))
    elif args.command == "undo":
        print(json.dumps(undo_npz(args.edited, args.receipt, args.output), indent=2))
    elif args.command == "costs":
        from .cost_experiments import CostConfig, run_cost_experiment
        try:
            config = CostConfig(seeds=tuple(args.seeds),
                                shapes=tuple(tuple(int(n) for n in shape.lower().split("x")) for shape in args.shapes),
                                precisions=tuple(args.precisions), alpha=args.alpha, repeats=args.repeats,
                                worker_timeout_seconds=args.worker_timeout)
            summary = run_cost_experiment(args.out, config, provenance=_manifest(Path(__file__).resolve().parents[2]),
                                          progress=lambda message: print(message, flush=True))
        except (ValueError, FileExistsError, RuntimeError) as error:
            parser.error(str(error))
        print(json.dumps({key: value for key, value in summary.items() if key != "groups"}, indent=2))
        return 0 if summary["status"] == "complete" else 1
    elif args.command == "refine":
        if args.cycles < 1 or args.active_rounds < 1:
            parser.error("--cycles and --active-rounds must be positive")
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("Use an empty/new result directory to preserve existing evidence")
        from .roundoff_experiments import RefinementConfig, run_refinement
        config=RefinementConfig(seeds=tuple(args.seeds),precisions=tuple(args.precisions),
                                cycle_count=args.cycles,active_rounds=args.active_rounds)
        data=run_refinement(config,include_active=not args.no_active,progress=lambda msg: print(msg,flush=True))
        args.out.mkdir(parents=True,exist_ok=True)
        root=Path(__file__).resolve().parents[2]
        data["provenance"]=_manifest(root)
        (args.out/"results.json").write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8")
        _csv(args.out/"tradeoffs.csv",data["tradeoff_rows"])
        _csv(args.out/"cycles.csv",data["cycle_rows"])
        from .roundoff_report import write_refinement_report
        write_refinement_report(data,args.out,plots=args.plots)
        print(json.dumps({"result_directory":str(args.out.resolve()),"tradeoff_comparisons":len(data["tradeoff_rows"]),
                          "cycle_observations":len(data["cycle_rows"]),"active_runs":len(data["active_runs"]),
                          "seconds":data["runtime_seconds"]},indent=2))
    else:
        if args.cycles < 1 or args.active_rounds < 1:
            parser.error("--cycles and --active-rounds must be positive")
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("Use an empty/new result directory to preserve existing evidence")
        config = ExperimentConfig(seeds=tuple(args.seeds), precisions=tuple(args.precisions),
                                  cycle_count=args.cycles, active_rounds=args.active_rounds)
        data = run_all(config, include_active=not args.no_active)
        args.out.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parents[2]
        data["provenance"] = _manifest(root)
        (args.out / "results.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        _csv(args.out / "kernel.csv", data["kernel_rows"])
        _csv(args.out / "cycles.csv", data["cycle_rows"])
        from .report import write_report
        write_report(data, args.out, plots=args.plots)
        print(json.dumps({"result_directory": str(args.out.resolve()),
                          "kernel_comparisons": len(data["kernel_rows"]),
                          "cycle_observations": len(data["cycle_rows"]),
                          "active_runs": len(data["active_runs"]),
                          "seconds": data["runtime_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

