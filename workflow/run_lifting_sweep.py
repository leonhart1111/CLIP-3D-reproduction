#!/usr/bin/env python3
"""Resume and parallelize one CLIP-3D lifting experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, write_json
from workflow.run_lifting_pipeline import DEFAULT_CONFIG, LAYOUT_METHODS, run_pipeline


DEFAULT_R1_ROOT = PROJECT_ROOT / "runs/architecture_sweep/r1/paper"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/architecture_sweep/lifting"


def discover(root: Path, workloads: set[str] | None = None) -> list[Path]:
    points = []
    for metadata_path in root.rglob("r1_metadata.json"):
        directory = metadata_path.parent
        if not (directory / "stats.txt").is_file():
            continue
        metadata = read_json(metadata_path)
        if workloads is not None and metadata.get("workload") not in workloads:
            continue
        status = directory / "status.json"
        if status.is_file() and read_json(status).get("state") != "success":
            continue
        points.append(directory)
    return sorted(points)


def completed(output: Path, config: dict, layout_method: str,
              require_r2: bool) -> bool:
    path = output / "pipeline_summary.json"
    run_config_path = output / "run_config.json"
    if not path.is_file() or not run_config_path.is_file():
        return False
    try:
        summary = read_json(path)
        recorded_config = read_json(run_config_path).get("config")
    except Exception:
        return False
    # Cooling alone is not a sufficient cache key: McPAT activity mapping,
    # local CACTI geometry, area calibration, or layer materials may change
    # while R_conv stays identical.
    if recorded_config != config:
        return False
    if summary.get("layout_method", summary.get("layout_mode")) != layout_method:
        return False
    cooling = summary.get("cooling", {})
    if float(cooling.get("r_convec_k_per_w", -1)) != float(
            config["physical"]["r_convec_k_per_w"]):
        return False
    if require_r2 and (summary.get("ipc2") is None or summary.get("bips2") is None):
        return False
    return True


def one_job(args_tuple):
    r1, output, config_path, layout_method, execute_r2, rerun_r2, reuse_r2 = args_tuple
    try:
        summary = run_pipeline(
            r1, output, config_path, layout_method, execute_r2, rerun_r2, reuse_r2
        )
        return {"r1": str(r1), "output": str(output), "state": "success",
                "summary": summary}
    except Exception as error:  # retain all independent completed points
        return {"r1": str(r1), "output": str(output), "state": "failed",
                "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-root", type=Path, default=DEFAULT_R1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--layout-method", choices=LAYOUT_METHODS, default="fixed-bin")
    parser.add_argument("--optimized-layout", action="store_true",
                        help="deprecated alias for --layout-method clip3d")
    parser.add_argument("--run-r2", action="store_true")
    parser.add_argument("--rerun-r2", action="store_true")
    parser.add_argument("--reuse-r2-root", type=Path,
                        help="reuse matching per-point R2 results from another sweep root")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.optimized_layout:
        if args.layout_method != "fixed-bin":
            parser.error("do not combine --optimized-layout and --layout-method")
        args.layout_method = "clip3d"
    if args.run_r2 and args.reuse_r2_root:
        parser.error("choose either --run-r2 or --reuse-r2-root")

    r1_root = args.r1_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    reuse_root = args.reuse_r2_root.resolve() if args.reuse_r2_root else None
    points = discover(r1_root, set(args.workloads) if args.workloads else None)
    jobs = []
    skipped = []
    for point in points:
        output = output_root / point.relative_to(r1_root)
        if not args.rerun and completed(
                output, config, args.layout_method, args.run_r2 or reuse_root is not None):
            skipped.append(str(point))
        else:
            reuse_r2 = reuse_root / point.relative_to(r1_root) if reuse_root else None
            jobs.append((point, output, config_path, args.layout_method,
                         args.run_r2, args.rerun_r2, reuse_r2))

    if args.jobs == 1:
        results = [one_job(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(one_job, jobs))
    report = {
        "schema_version": 2, "r1_root": str(r1_root),
        "output_root": str(output_root), "config": str(config_path),
        "experiment": config.get("name", config_path.stem),
        "layout_method": args.layout_method, "run_r2": args.run_r2,
        "reuse_r2_root": str(reuse_root) if reuse_root else None,
        "workloads": args.workloads, "discovered": len(points),
        "executed": len(jobs), "skipped": skipped, "results": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "sweep_status.json", report)
    success = sum(result["state"] == "success" for result in results)
    failed = len(results) - success
    print(f"lifting sweep: discovered={len(points)} success={success} "
          f"failed={failed} skipped={len(skipped)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
