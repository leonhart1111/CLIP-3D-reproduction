#!/usr/bin/env python3
"""Run the optional gem5-window/McPAT/HotSpot transient branch for one point."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, write_json
from workflow.transient.generate_hotspot_trace import materialize_trace
from workflow.transient.run_hotspot_transient import run_hotspot_transient
from workflow.transient.run_transient_r1 import run as run_transient_r1
from workflow.transient.run_windowed_mcpat import run_windows
from workflow.transient.stats_windows import split_windows


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/clip3d_pipeline.json"


def validate_matching_r1(source: dict, transient: dict) -> None:
    keys = (
        "workload", "binary", "command", "num_cores", "cpu_clock",
        "l1i_size", "l1d_size", "l2_size", "memory_size",
        "warmup_insts", "measure_insts", "instruction_window_scope", "latencies",
    )
    mismatches = [key for key in keys if source.get(key) != transient.get(key)]
    if mismatches:
        raise ValueError(
            "transient R1 does not match the source steady R1 metadata: "
            + ", ".join(mismatches)
        )


def run_transient_pipeline(source_r1_dir: Path, steady_output_dir: Path,
                           output_dir: Path, config_path: Path,
                           sample_ms: float = 10.0,
                           transient_r1_dir: Path | None = None,
                           initial_temperature: str = "steady",
                           rerun_transient_r1: bool = False) -> dict:
    source_r1_dir = source_r1_dir.resolve()
    steady_output_dir = steady_output_dir.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    if not math.isfinite(sample_ms) or sample_ms <= 0:
        raise ValueError("sample_ms must be positive")
    required_steady = (
        steady_output_dir / "modules.json",
        steady_output_dir / "hotspot/layout.json",
        steady_output_dir / "hotspot/steady.txt",
    )
    for path in required_steady:
        if not path.is_file():
            raise FileNotFoundError(
                f"steady pipeline artifact required by transient branch: {path}"
            )
    config = read_json(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_seconds = {}

    if transient_r1_dir is None:
        transient_r1_dir = output_dir / "r1"
        started = time.perf_counter()
        print(
            f"Transient branch: launching/reusing periodic-statistics R1 in "
            f"{transient_r1_dir}",
            flush=True,
        )
        r1_status = run_transient_r1(
            source_r1_dir, transient_r1_dir, sample_ms,
            rerun=rerun_transient_r1,
        )
        stage_seconds["transient_r1"] = time.perf_counter() - started
        transient_r1_source = "generated"
    else:
        transient_r1_dir = transient_r1_dir.resolve()
        r1_status = (
            read_json(transient_r1_dir / "status.json")
            if (transient_r1_dir / "status.json").is_file()
            else None
        )
        transient_r1_source = "provided"
    metadata = read_json(transient_r1_dir / "r1_metadata.json")
    validate_matching_r1(read_json(source_r1_dir / "r1_metadata.json"), metadata)
    recorded_ms = float(metadata.get("sample_interval_ms", -1))
    if abs(recorded_ms - sample_ms) > 1e-12:
        raise ValueError(
            f"transient R1 sampling mismatch: requested {sample_ms} ms, "
            f"recorded {recorded_ms} ms"
        )

    started = time.perf_counter()
    windows_dir = output_dir / "windows/gem5"
    print("Transient branch: splitting cumulative gem5 statistics", flush=True)
    windows = split_windows(transient_r1_dir, windows_dir)
    stage_seconds["split_stats"] = time.perf_counter() - started

    started = time.perf_counter()
    mcpat_dir = output_dir / "windows/mcpat"
    print(f"Transient branch: running McPAT for {windows['window_count']} windows", flush=True)
    power_windows = run_windows(
        windows_dir / "windows_manifest.json", mcpat_dir, config
    )
    stage_seconds["windowed_mcpat"] = time.perf_counter() - started

    started = time.perf_counter()
    hotspot_dir = output_dir / "hotspot"
    print("Transient branch: mapping module powers to the selected 3-D layout", flush=True)
    trace = materialize_trace(
        steady_output_dir / "modules.json",
        steady_output_dir / "hotspot/layout.json",
        mcpat_dir / "power_windows.json",
        hotspot_dir,
        config,
    )
    stage_seconds["power_trace_mapping"] = time.perf_counter() - started

    started = time.perf_counter()
    print("Transient branch: running one multi-row detailed-3D HotSpot solve", flush=True)
    thermal = run_hotspot_transient(
        hotspot_dir,
        initial_temperature=initial_temperature,
        steady_source=steady_output_dir / "hotspot/steady.txt",
    )
    stage_seconds["hotspot_transient"] = time.perf_counter() - started

    steady_summary = read_json(steady_output_dir / "pipeline_summary.json")
    result = {
        "schema_version": 1,
        "mode": "optional transient validation; paper steady-state path is unchanged",
        "source_r1": str(source_r1_dir),
        "transient_r1": str(transient_r1_dir),
        "transient_r1_source": transient_r1_source,
        "transient_r1_status": r1_status,
        "steady_output": str(steady_output_dir),
        "output": str(output_dir),
        "config": str(config_path),
        "workload": steady_summary["workload"],
        "layout_method": steady_summary["layout_method"],
        "sample_interval_ms": sample_ms,
        "window_count": windows["window_count"],
        "actual_gem5_duration_s": trace["actual_gem5_duration_s"],
        "hotspot_trace_duration_s": trace["hotspot_trace_duration_s"],
        "initial_temperature": initial_temperature,
        "steady_tmax_c": steady_summary["tmax_c"],
        "transient_tmax_c": thermal["tmax_c"],
        "transient_minus_steady_c": thermal["tmax_c"] - steady_summary["tmax_c"],
        "stage_seconds": stage_seconds,
        "total_pipeline_seconds": sum(stage_seconds.values()),
        "artifacts": {
            "windows_manifest": str((windows_dir / "windows_manifest.json").resolve()),
            "power_windows": str((mcpat_dir / "power_windows.json").resolve()),
            "power_trace": str((hotspot_dir / "power_transient.ptrace").resolve()),
            "temperature_trace": str((hotspot_dir / "transient.ttrace").resolve()),
            "thermal_result": str((hotspot_dir / "transient_result.json").resolve()),
            "temperature_summary_csv": str(
                (hotspot_dir / "transient_summary.csv").resolve()
            ),
        },
        "limitations": [
            "McPAT leakage is evaluated at its configured fixed operating temperature.",
            "There is no temperature-leakage-DVFS feedback loop in this validation branch.",
            "A final partial gem5 window is padded to one full HotSpot interval.",
        ],
    }
    write_json(output_dir / "transient_pipeline_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-r1-dir", type=Path, required=True)
    parser.add_argument("--steady-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-ms", type=float, default=10.0)
    parser.add_argument("--transient-r1-dir", type=Path)
    parser.add_argument("--initial-temperature", choices=("steady", "ambient"),
                        default="steady")
    parser.add_argument("--rerun-transient-r1", action="store_true")
    args = parser.parse_args()
    result = run_transient_pipeline(
        args.source_r1_dir, args.steady_output_dir, args.output_dir,
        args.config, args.sample_ms, args.transient_r1_dir,
        args.initial_temperature, args.rerun_transient_r1,
    )
    print(
        f"Transient pipeline complete: windows={result['window_count']}, "
        f"Tmax={result['transient_tmax_c']:.3f} C"
    )


if __name__ == "__main__":
    main()
