#!/usr/bin/env python3
"""Run the optional gem5-window/McPAT/HotSpot transient branch for one point."""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_steady_output(steady_output_dir: Path,
                           expected_layout: str | None = None) -> dict:
    """Preflight one completed steady pipeline before expensive transient work."""
    steady_output_dir = steady_output_dir.resolve()
    required = (
        steady_output_dir / "modules.json",
        steady_output_dir / "hotspot/layout.json",
        steady_output_dir / "hotspot/steady.txt",
        steady_output_dir / "pipeline_summary.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"steady pipeline artifact required by transient branch: {path}"
            )
    summary = read_json(steady_output_dir / "pipeline_summary.json")
    if expected_layout is not None and summary.get("layout_method") != expected_layout:
        raise ValueError(
            f"steady pipeline layout must be {expected_layout}, "
            f"got {summary.get('layout_method')}"
        )
    return summary


def prepare_power_windows(source_r1_dir: Path, transient_r1_dir: Path,
                          output_dir: Path, config: dict,
                          sample_ms: float) -> dict:
    """Split one periodic R1 and run McPAT once for all thermal layouts."""
    source_r1_dir = source_r1_dir.resolve()
    transient_r1_dir = transient_r1_dir.resolve()
    output_dir = output_dir.resolve()
    if not math.isfinite(sample_ms) or sample_ms <= 0:
        raise ValueError("sample_ms must be positive")

    source_metadata = read_json(source_r1_dir / "r1_metadata.json")
    transient_metadata = read_json(transient_r1_dir / "r1_metadata.json")
    validate_matching_r1(source_metadata, transient_metadata)
    recorded_ms = float(transient_metadata.get("sample_interval_ms", -1))
    if abs(recorded_ms - sample_ms) > 1e-12:
        raise ValueError(
            f"transient R1 sampling mismatch: requested {sample_ms} ms, "
            f"recorded {recorded_ms} ms"
        )

    stage_seconds = {}
    gem5_windows_dir = output_dir / "gem5"
    started = time.perf_counter()
    print("Transient branch: splitting cumulative gem5 statistics", flush=True)
    windows = split_windows(transient_r1_dir, gem5_windows_dir)
    stage_seconds["split_stats"] = time.perf_counter() - started

    mcpat_dir = output_dir / "mcpat"
    started = time.perf_counter()
    print(
        f"Transient branch: running McPAT for {windows['window_count']} windows",
        flush=True,
    )
    power_windows = run_windows(
        gem5_windows_dir / "windows_manifest.json", mcpat_dir, config
    )
    stage_seconds["windowed_mcpat"] = time.perf_counter() - started
    if int(power_windows["window_count"]) != int(windows["window_count"]):
        raise ValueError("McPAT power window count does not match gem5 windows")
    actual_duration_s = sum(
        float(window["duration_s"]) for window in power_windows["windows"]
    )
    power_windows_path = (mcpat_dir / "power_windows.json").resolve()
    return {
        "schema_version": 1,
        "source_r1": str(source_r1_dir),
        "transient_r1": str(transient_r1_dir),
        "sample_interval_ms": sample_ms,
        "window_count": int(windows["window_count"]),
        "actual_gem5_duration_s": actual_duration_s,
        "hotspot_trace_duration_s": int(windows["window_count"]) * sample_ms / 1000.0,
        "padded_final_duration_s": max(
            int(windows["window_count"]) * sample_ms / 1000.0
            - actual_duration_s,
            0.0,
        ),
        "windows_manifest": str(
            (gem5_windows_dir / "windows_manifest.json").resolve()
        ),
        "power_windows": str(power_windows_path),
        "power_trace_identity": _sha256(power_windows_path),
        "stage_seconds": stage_seconds,
    }


def run_layout_thermal(source_r1_dir: Path, steady_output_dir: Path,
                       output_dir: Path, config: dict,
                       power_windows_path: Path,
                       initial_temperature: str = "steady") -> dict:
    """Map shared McPAT windows onto one layout and run transient HotSpot."""
    source_r1_dir = source_r1_dir.resolve()
    steady_output_dir = steady_output_dir.resolve()
    output_dir = output_dir.resolve()
    power_windows_path = power_windows_path.resolve()
    steady_summary = validate_steady_output(steady_output_dir)
    if initial_temperature not in ("steady", "ambient"):
        raise ValueError("initial_temperature must be steady or ambient")

    stage_seconds = {}
    hotspot_dir = output_dir / "hotspot"
    started = time.perf_counter()
    print("Transient branch: mapping module powers to the selected 3-D layout", flush=True)
    trace = materialize_trace(
        steady_output_dir / "modules.json",
        steady_output_dir / "hotspot/layout.json",
        power_windows_path,
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

    return {
        "schema_version": 1,
        "mode": "optional transient validation; paper steady-state path is unchanged",
        "source_r1": str(source_r1_dir),
        "steady_output": str(steady_output_dir),
        "output": str(output_dir),
        "workload": steady_summary["workload"],
        "layout_method": steady_summary["layout_method"],
        "sample_interval_ms": float(trace["sample_interval_s"]) * 1000.0,
        "window_count": int(trace["window_count"]),
        "actual_gem5_duration_s": float(trace["actual_gem5_duration_s"]),
        "hotspot_trace_duration_s": float(trace["hotspot_trace_duration_s"]),
        "padded_final_duration_s": float(trace["padded_final_duration_s"]),
        "initial_temperature": initial_temperature,
        "steady_tmax_c": float(steady_summary["tmax_c"]),
        "transient_tmax_c": float(thermal["tmax_c"]),
        "transient_minus_steady_c": (
            float(thermal["tmax_c"]) - float(steady_summary["tmax_c"])
        ),
        "temperature": thermal,
        "power_summary": trace["power_summary"],
        "power_trace_identity": _sha256(power_windows_path),
        "stage_seconds": stage_seconds,
        "artifacts": {
            "power_windows": str(power_windows_path),
            "power_trace": str((hotspot_dir / "power_transient.ptrace").resolve()),
            "temperature_trace": str((hotspot_dir / "transient.ttrace").resolve()),
            "thermal_result": str((hotspot_dir / "transient_result.json").resolve()),
            "temperature_summary_csv": str(
                (hotspot_dir / "transient_summary.csv").resolve()
            ),
        },
    }


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
    config = read_json(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_seconds = {}
    validate_steady_output(steady_output_dir)

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
    prepared = prepare_power_windows(
        source_r1_dir, transient_r1_dir, output_dir / "windows", config, sample_ms
    )
    stage_seconds.update(prepared["stage_seconds"])
    result = run_layout_thermal(
        source_r1_dir,
        steady_output_dir,
        output_dir,
        config,
        Path(prepared["power_windows"]),
        initial_temperature,
    )
    stage_seconds.update(result["stage_seconds"])
    result.update({
        "transient_r1": str(transient_r1_dir),
        "transient_r1_source": transient_r1_source,
        "transient_r1_status": r1_status,
        "config": str(config_path),
        "stage_seconds": stage_seconds,
        "total_pipeline_seconds": sum(stage_seconds.values()),
        "limitations": [
            "McPAT leakage is evaluated at its configured fixed operating temperature.",
            "There is no temperature-leakage-DVFS feedback loop in this validation branch.",
            "A final partial gem5 window is padded to one full HotSpot interval.",
        ],
    })
    result["artifacts"]["windows_manifest"] = prepared["windows_manifest"]
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
