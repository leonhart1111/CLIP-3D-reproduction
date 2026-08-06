#!/usr/bin/env python3
"""Resume and parallelize one CLIP-3D lifting experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, sha256_file, write_json
from workflow.r1_catalog import build_catalogue, canonical_directories
from workflow.run_lifting_pipeline import DEFAULT_CONFIG, LAYOUT_METHODS, run_pipeline


DEFAULT_R1_ROOT = PROJECT_ROOT / "runs/architecture_sweep/r1/paper"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/architecture_sweep/lifting"
DEFAULT_R1_EXPERIMENT = PROJECT_ROOT / "configs/experiments/r1_cache_sweep.json"


required_artifacts = (
    "run_config.json",
    "mcpat/mcpat.json",
    "cacti/cacti_characterization.json",
    "modules.json",
    "hotspot/layout.json",
    "hotspot/thermal_result.json",
    "performance.json",
    "r2_latency.json",
    "pipeline_summary.json",
)
CLIP3D_REQUIRED_ARTIFACTS = ("optimizer_report.json", "layout_selection.json")


def discover(root: Path, experiment_path: Path, profile: str = "paper",
             workloads: set[str] | None = None,
             require_complete: bool = True) -> tuple[list[Path], dict]:
    """Return valid configured canonical points after validating the full grid."""
    catalogue = build_catalogue(root, experiment_path, profile=profile)
    if require_complete and not catalogue["complete"]:
        raise ValueError("canonical R1 catalogue is incomplete or invalid")
    points = canonical_directories(catalogue)
    if workloads is not None:
        points = [point for point in points
                  if point.relative_to(root.resolve()).parts[0] in workloads]
    return points, catalogue


def completed(output: Path, config: dict, layout_method: str,
              require_r2: bool) -> bool:
    artifacts = required_artifacts
    if layout_method == "clip3d":
        artifacts += CLIP3D_REQUIRED_ARTIFACTS
    if any(not (output / artifact).is_file() for artifact in artifacts):
        return False
    path = output / "pipeline_summary.json"
    run_config_path = output / "run_config.json"
    try:
        summary = read_json(path)
        recorded_config = read_json(run_config_path).get("config")
    except (AttributeError, OSError, TypeError, ValueError):
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
    if not require_r2 and (summary.get("ipc2") is not None
                           or summary.get("bips2") is not None):
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
    parser.add_argument("--r1-experiment", type=Path, default=DEFAULT_R1_EXPERIMENT)
    parser.add_argument("--r1-profile", default="paper")
    parser.add_argument("--allow-incomplete-canonical", action="store_true")
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
    r1_experiment = args.r1_experiment.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    reuse_root = args.reuse_r2_root.resolve() if args.reuse_r2_root else None
    points, catalogue = discover(
        r1_root, r1_experiment, args.r1_profile,
        set(args.workloads) if args.workloads else None,
        not args.allow_incomplete_canonical,
    )
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
    skipped_summaries = [read_json(
        output_root / point.relative_to(r1_root) / "pipeline_summary.json"
    ) for point in points if str(point) in skipped]
    result_summaries = [result["summary"] for result in results
                        if result["state"] == "success"]
    all_summaries = skipped_summaries + result_summaries
    contains_r2 = any(summary.get("ipc2") is not None or summary.get("bips2") is not None
                      for summary in all_summaries)
    layout_only = not args.run_r2 and reuse_root is None
    if layout_only and contains_r2:
        raise RuntimeError("layout-only lifting sweep contains R2 performance results")
    failed = sum(result["state"] == "failed" for result in results)
    report = {
        "schema_version": 3, "r1_root": str(r1_root),
        "r1_experiment": str(r1_experiment), "r1_profile": args.r1_profile,
        "output_root": str(output_root), "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "experiment": config.get("name", config_path.stem),
        "classification": config.get("experiment_classification"),
        "layout_method": args.layout_method, "run_r2": args.run_r2,
        "reuse_r2_root": str(reuse_root) if reuse_root else None,
        "contains_r2": contains_r2,
        "workloads": args.workloads,
        "selected_workload_count": len({
            point.relative_to(r1_root).parts[0] for point in points
        }),
        "canonical_expected": catalogue["expected_count"],
        "canonical_valid": catalogue["valid_count"],
        "canonical_complete": catalogue["complete"],
        "canonical_selected": len(points),
        "excluded_noncanonical": catalogue["excluded_noncanonical"],
        "discovered": len(points), "executed": len(jobs),
        "skipped_count": len(skipped), "failed": failed,
        "skipped": skipped, "results": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "sweep_status.json", report)
    success = sum(result["state"] == "success" for result in results)
    print(f"lifting sweep: discovered={len(points)} success={success} "
          f"failed={failed} skipped={len(skipped)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
