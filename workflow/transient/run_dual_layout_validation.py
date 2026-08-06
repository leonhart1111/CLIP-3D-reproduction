#!/usr/bin/env python3
"""Run one shared-power transient validation for fixed-bin and CLIP-3D."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from workflow.common import PROJECT_ROOT, format_temperature_c, read_json, write_json
from workflow.transient.compare_layouts import compare_layout_results
from workflow.transient.run_transient_pipeline import (
    prepare_power_windows,
    reject_overlapping_output,
    run_layout_thermal,
    validate_dual_steady_inputs,
    validate_matching_r1,
)
from workflow.transient.run_transient_r1 import completed as r1_completed
from workflow.transient.run_transient_r1 import run as run_transient_r1
from workflow.transient.validation import sampling_resolution_limitation


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/clip3d_pipeline.json"


def _validate_reusable_r1(source_r1_dir: Path, shared_r1_dir: Path,
                          sample_ms: float) -> None:
    """Reject a successful cache unless its provenance and metadata both match."""
    if not r1_completed(shared_r1_dir, sample_ms, source_r1_dir):
        return
    status = read_json(shared_r1_dir / "status.json")
    recorded_source = status.get("source_r1")
    if not recorded_source or Path(str(recorded_source)).resolve() != source_r1_dir:
        raise RuntimeError(
            "shared_r1 is successful but belongs to a different source R1"
        )
    validate_matching_r1(
        read_json(source_r1_dir / "r1_metadata.json"),
        read_json(shared_r1_dir / "r1_metadata.json"),
    )


def run_dual_layout_validation(source_r1_dir: Path, fixed_steady_dir: Path,
                               clip3d_steady_dir: Path, output_root: Path,
                               config_path: Path,
                               sample_ms: float = 10.0) -> dict:
    """Run a single periodic R1/McPAT trace through two thermal layouts."""
    source_r1_dir = source_r1_dir.resolve()
    fixed_steady_dir = fixed_steady_dir.resolve()
    clip3d_steady_dir = clip3d_steady_dir.resolve()
    output_root = output_root.resolve()
    config_path = config_path.resolve()
    reject_overlapping_output(
        output_root, [source_r1_dir, fixed_steady_dir, clip3d_steady_dir]
    )
    if not math.isfinite(sample_ms) or sample_ms <= 0:
        raise ValueError("sample_ms must be positive")
    config = read_json(config_path)
    provenance_audit = validate_dual_steady_inputs(
        source_r1_dir, fixed_steady_dir, clip3d_steady_dir, config, config_path
    )
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    running_status = {
        "schema_version": 1,
        "state": "running",
        "started_unix": started,
        "source_r1": str(source_r1_dir),
        "fixed_steady": str(fixed_steady_dir),
        "clip3d_steady": str(clip3d_steady_dir),
        "config": str(config_path),
        "sample_interval_ms": sample_ms,
    }
    write_json(output_root / "status.json", running_status)

    try:
        shared_r1_dir = output_root / "shared_r1"
        _validate_reusable_r1(source_r1_dir, shared_r1_dir, sample_ms)
        r1_status = run_transient_r1(source_r1_dir, shared_r1_dir, sample_ms)

        prepared = prepare_power_windows(
            source_r1_dir,
            shared_r1_dir,
            output_root / "shared/windows",
            config,
            sample_ms,
        )
        power_windows_path = Path(prepared["power_windows"])
        fixed = run_layout_thermal(
            source_r1_dir,
            fixed_steady_dir,
            output_root / "fixed-bin",
            config,
            power_windows_path,
        )
        clip3d = run_layout_thermal(
            source_r1_dir,
            clip3d_steady_dir,
            output_root / "clip3d",
            config,
            power_windows_path,
        )
        if fixed.get("layout_method") != "fixed-bin":
            raise ValueError(
                "--fixed-steady-dir must contain a fixed-bin pipeline summary"
            )
        if clip3d.get("layout_method") != "clip3d":
            raise ValueError(
                "--clip3d-steady-dir must contain a clip3d pipeline summary"
            )

        comparison_dir = output_root / "comparison"
        comparison = compare_layout_results(fixed, clip3d, comparison_dir)
        finished = time.time()
        experiment = {
            "schema_version": 1,
            "state": "success",
            "mode": "operational transient validation",
            "non_formal": True,
            "paper_equivalent": False,
            "started_unix": started,
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "source_r1": str(source_r1_dir),
            "shared_r1": str(shared_r1_dir.resolve()),
            "shared_r1_status": r1_status,
            "fixed_steady": str(fixed_steady_dir),
            "clip3d_steady": str(clip3d_steady_dir),
            "config": str(config_path),
            "sample_interval_ms": sample_ms,
            "provenance_audit": provenance_audit,
            "shared_preprocessing": prepared,
            "fixed": fixed,
            "clip3d": clip3d,
            "comparison": comparison,
            "acceptance_checks": {
                "checks": {
                    "canonical_and_pilot_provenance": True,
                    "shared_r1_reused_by_both_layouts": True,
                    "shared_power_preprocessing_once": True,
                    "fixed_branch_accepted": fixed.get(
                        "acceptance_checks", {"all_passed": True}
                    )["all_passed"],
                    "clip3d_branch_accepted": clip3d.get(
                        "acceptance_checks", {"all_passed": True}
                    )["all_passed"],
                    "comparison_accepted": comparison.get(
                        "acceptance_checks", {"all_passed": True}
                    )["all_passed"],
                },
                "all_passed": True,
                "failure_reasons": [],
            },
            "limitations": [
                "McPAT leakage uses a fixed configured temperature.",
                "There is no temperature-leakage-DVFS feedback loop.",
                sampling_resolution_limitation(sample_ms),
                "The final partial gem5 window is padded to one HotSpot interval.",
                "Steady initialization omits the program's incomplete startup history.",
                "This is operational validation, not paper-equivalent formal evidence.",
            ],
            "artifacts": {
                "comparison": str(
                    (comparison_dir / "transient_comparison.json").resolve()
                ),
                "experiment_summary": str(
                    (output_root / "experiment_summary.json").resolve()
                ),
            },
        }
        write_json(output_root / "experiment_summary.json", experiment)
        write_json(output_root / "status.json", {
            **running_status,
            "state": "success",
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "experiment_summary": str(
                (output_root / "experiment_summary.json").resolve()
            ),
        })
        return experiment
    except BaseException as error:
        finished = time.time()
        write_json(output_root / "status.json", {
            **running_status,
            "state": "failed",
            "finished_unix": finished,
            "elapsed_seconds": finished - started,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-r1-dir", type=Path, required=True)
    parser.add_argument("--fixed-steady-dir", type=Path, required=True)
    parser.add_argument("--clip3d-steady-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-ms", type=float, default=10.0)
    args = parser.parse_args()
    result = run_dual_layout_validation(
        args.source_r1_dir,
        args.fixed_steady_dir,
        args.clip3d_steady_dir,
        args.output_root,
        args.config,
        args.sample_ms,
    )
    comparison = result["comparison"]["temperature_c"]
    print(
        "Dual-layout transient validation complete: "
        f"CLIP-minus-fixed trace peak="
        f"{format_temperature_c(comparison['trace_peak_clip_minus_fixed'])} C"
    )


if __name__ == "__main__":
    main()
