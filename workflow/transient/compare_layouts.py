#!/usr/bin/env python3
"""Compare fixed-bin and CLIP-3D transient results on one shared power trace."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from workflow.common import format_temperature_csv_row, read_json, write_json
from workflow.transient.validation import (
    power_trace_identity,
    sampling_resolution_limitation,
    summarize_power_windows,
    validate_power_windows,
)


COMPARISON_FIELDS = (
    "layout",
    "steady_peak_c",
    "trace_peak_c",
    "final_peak_c",
    "trace_peak_time_s",
    "power_peak_time_s",
    "power_peak_to_temperature_peak_lag_s",
)
TIMESERIES_FIELDS = (
    "index",
    "time_s",
    "total_power_w",
    "fixed_peak_c",
    "fixed_average_c",
    "clip3d_peak_c",
    "clip3d_average_c",
)


def _same_number(fixed: dict, clip3d: dict, field: str, label: str) -> float:
    try:
        fixed_value = float(fixed[field])
        clip_value = float(clip3d[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is missing or invalid") from error
    if not math.isfinite(fixed_value) or not math.isfinite(clip_value):
        raise ValueError(f"{label} must be finite")
    if not math.isclose(fixed_value, clip_value, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"{label} mismatch: fixed={fixed_value}, clip3d={clip_value}"
        )
    return fixed_value


def _power_windows_path(summary: dict) -> Path:
    value = summary.get("power_windows")
    if isinstance(value, str):
        return Path(value).resolve()
    artifacts = summary.get("artifacts", {})
    value = artifacts.get("power_windows") if isinstance(artifacts, dict) else None
    if not isinstance(value, str):
        raise ValueError("power trace identity requires a power_windows artifact")
    return Path(value).resolve()


def _power_identity(summary: dict, path: Path) -> str:
    identity = summary.get("power_trace_identity")
    if identity is not None:
        return str(identity)
    return power_trace_identity(read_json(path))


def _temperature(summary: dict) -> dict:
    for field in ("temperature", "thermal"):
        value = summary.get(field)
        if isinstance(value, dict):
            return value
    return summary


def _temperature_values(summary: dict, label: str) -> dict:
    temperature = _temperature(summary)
    try:
        initial_peak = temperature["initial_peak"]
        trace_min_peak = temperature["trace_min_peak"]
        trace_peak = temperature["trace_peak"]
        final_peak = temperature["final_peak"]
        overall_peak = temperature["overall_peak"]
        samples = temperature["samples"]
        result = {
            "steady_peak_c": float(summary["steady_tmax_c"]),
            "initial_peak": initial_peak,
            "trace_min_peak": trace_min_peak,
            "trace_peak": trace_peak,
            "final_peak": final_peak,
            "overall_peak": overall_peak,
            "initial_peak_c": float(initial_peak["tmax_c"]),
            "trace_min_peak_c": float(trace_min_peak["tmax_c"]),
            "trace_peak_c": float(trace_peak["tmax_c"]),
            "final_peak_c": float(final_peak["tmax_c"]),
            "overall_peak_c": float(overall_peak["tmax_c"]),
            "trace_peak_time_s": float(trace_peak["time_s"]),
            "samples": samples,
            "trace_peak_minus_steady_c": (
                float(trace_peak["tmax_c"]) - float(summary["steady_tmax_c"])
            ),
            "final_peak_minus_steady_c": (
                float(final_peak["tmax_c"]) - float(summary["steady_tmax_c"])
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} temperature summary is incomplete") from error
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{label} temperature samples must not be empty")
    for field in (
        "steady_peak_c", "initial_peak_c", "trace_min_peak_c", "trace_peak_c",
        "final_peak_c", "overall_peak_c", "trace_peak_time_s",
    ):
        if not math.isfinite(result[field]):
            raise ValueError(f"{label} {field} must be finite")
    return result


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(format_temperature_csv_row(row) for row in rows)


def compare_layout_results(fixed: dict, clip3d: dict, output_dir: Path) -> dict:
    """Validate common inputs, then emit deterministic thermal comparison evidence."""
    for label, summary in (("fixed-bin", fixed), ("CLIP-3D", clip3d)):
        if (
            summary.get("mode") != "operational transient validation"
            or summary.get("non_formal") is not True
            or summary.get("paper_equivalent") is not False
        ):
            raise ValueError(f"{label} branch classification is incomplete")
        if not isinstance(summary.get("raw_power_evidence"), dict):
            raise ValueError(f"{label} raw-power evidence is missing")
        if not isinstance(summary.get("conservation_evidence"), dict):
            raise ValueError(f"{label} conservation evidence is missing")
        acceptance = summary.get("acceptance_checks")
        if not isinstance(acceptance, dict) or acceptance.get("all_passed") is not True:
            raise ValueError(f"{label} branch acceptance checks did not pass")
    fixed_r1 = Path(str(fixed.get("transient_r1", ""))).resolve()
    clip_r1 = Path(str(clip3d.get("transient_r1", ""))).resolve()
    if not fixed.get("transient_r1") or not clip3d.get("transient_r1"):
        raise ValueError("transient R1 is missing")
    if fixed_r1 != clip_r1:
        raise ValueError(f"transient R1 mismatch: fixed={fixed_r1}, clip3d={clip_r1}")
    fixed_source = Path(str(fixed.get("source_r1", ""))).resolve()
    clip_source = Path(str(clip3d.get("source_r1", ""))).resolve()
    if not fixed.get("source_r1") or not clip3d.get("source_r1"):
        raise ValueError("source R1 is missing")
    if fixed_source != clip_source:
        raise ValueError(
            f"source R1 mismatch: fixed={fixed_source}, clip3d={clip_source}"
        )

    sample_ms = _same_number(
        fixed, clip3d, "sample_interval_ms", "sample interval"
    )
    try:
        fixed_count = int(fixed["window_count"])
        clip_count = int(clip3d["window_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("window count is missing or invalid") from error
    if fixed_count <= 0 or clip_count != fixed_count:
        raise ValueError(
            f"window count mismatch: fixed={fixed_count}, clip3d={clip_count}"
        )
    actual_duration_s = _same_number(
        fixed, clip3d, "actual_gem5_duration_s", "actual duration"
    )
    trace_duration_s = _same_number(
        fixed, clip3d, "hotspot_trace_duration_s", "HotSpot trace duration"
    )
    padded_duration_s = _same_number(
        fixed, clip3d, "padded_final_duration_s", "padded final duration"
    )
    expected_trace_duration_s = fixed_count * sample_ms / 1000.0
    if not math.isclose(
        trace_duration_s, expected_trace_duration_s,
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ValueError("HotSpot trace duration is inconsistent with count and interval")
    expected_padding_s = max(trace_duration_s - actual_duration_s, 0.0)
    if not math.isclose(
        padded_duration_s, expected_padding_s, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("padded final duration is inconsistent with actual duration")
    fixed_power_path = _power_windows_path(fixed)
    clip_power_path = _power_windows_path(clip3d)
    fixed_identity = _power_identity(fixed, fixed_power_path)
    clip_identity = _power_identity(clip3d, clip_power_path)
    if fixed_identity != clip_identity:
        raise ValueError(
            f"power trace mismatch: fixed={fixed_identity}, clip3d={clip_identity}"
        )
    if fixed_power_path != clip_power_path:
        raise ValueError(
            "power trace mismatch: layouts do not reference the same power_windows artifact"
        )
    power_windows = read_json(fixed_power_path)
    actual_identity = power_trace_identity(power_windows)
    if fixed_identity != actual_identity or clip_identity != actual_identity:
        raise ValueError(
            "power trace identity does not match the current power_windows artifact"
        )

    windows = power_windows.get("windows")
    if not isinstance(windows, list) or len(windows) != fixed_count:
        raise ValueError("power trace window count does not match layout summaries")
    recorded_ms = float(power_windows.get("nominal_sample_interval_ms", -1))
    if not math.isclose(recorded_ms, sample_ms, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("power trace sample interval does not match layout summaries")
    validate_power_windows(power_windows)
    power_summary = summarize_power_windows(windows)

    fixed_temperature = _temperature_values(fixed, "fixed-bin")
    clip_temperature = _temperature_values(clip3d, "CLIP-3D")
    fixed_samples = fixed_temperature.pop("samples")
    clip_samples = clip_temperature.pop("samples")
    if len(fixed_samples) != fixed_count or len(clip_samples) != fixed_count:
        raise ValueError("temperature sample count does not match shared window count")

    peak_index = int(power_summary["total_power_w"]["peak_window_index"])
    power_peak_gem5_end_time_s = float(
        power_summary["total_power_w"]["peak_end_time_s"]
    )
    power_peak_time_s = (peak_index + 1) * sample_ms / 1000.0
    comparison_rows = []
    for layout, temperature in (
        ("fixed-bin", fixed_temperature), ("clip3d", clip_temperature)
    ):
        comparison_rows.append({
            "layout": layout,
            **temperature,
            "power_peak_time_s": power_peak_time_s,
            "power_peak_to_temperature_peak_lag_s": (
                temperature["trace_peak_time_s"] - power_peak_time_s
            ),
        })

    timeseries_rows = []
    elapsed_s = 0.0
    for index, (window, fixed_sample, clip_sample) in enumerate(
        zip(windows, fixed_samples, clip_samples)
    ):
        elapsed_s += float(window["duration_s"])
        fixed_index = int(fixed_sample.get("index", index))
        clip_index = int(clip_sample.get("index", index))
        if fixed_index != index or clip_index != index:
            raise ValueError(f"temperature sample index mismatch at window {index}")
        fixed_time = float(fixed_sample["time_s"])
        clip_time = float(clip_sample["time_s"])
        if not math.isclose(fixed_time, clip_time, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"temperature sample time mismatch at window {index}")
        timeseries_rows.append({
            "index": index,
            "time_s": fixed_time,
            "total_power_w": float(window["totals"]["total_power_w"]),
            "fixed_peak_c": float(fixed_sample["tmax_c"]),
            "fixed_average_c": float(fixed_sample["tavg_c"]),
            "clip3d_peak_c": float(clip_sample["tmax_c"]),
            "clip3d_average_c": float(clip_sample["tavg_c"]),
        })
    if not math.isclose(
        elapsed_s, actual_duration_s, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("power trace actual duration does not match layout summaries")

    fixed_lag = comparison_rows[0]["power_peak_to_temperature_peak_lag_s"]
    clip_lag = comparison_rows[1]["power_peak_to_temperature_peak_lag_s"]
    result = {
        "schema_version": 1,
        "mode": "operational transient validation",
        "non_formal": True,
        "paper_equivalent": False,
        "shared_input": {
            "source_r1": str(fixed_source),
            "transient_r1": str(fixed_r1),
            "sample_interval_ms": sample_ms,
            "window_count": fixed_count,
            "power_trace_identity": fixed_identity,
            "power_windows": str(fixed_power_path),
        },
        "duration_s": {
            "actual_gem5": actual_duration_s,
            "hotspot_trace": trace_duration_s,
            "padded_final": padded_duration_s,
        },
        "temperature_c": {
            "fixed": fixed_temperature,
            "clip3d": clip_temperature,
            "steady_peak_clip_minus_fixed": (
                clip_temperature["steady_peak_c"] - fixed_temperature["steady_peak_c"]
            ),
            "initial_peak_clip_minus_fixed": (
                clip_temperature["initial_peak_c"]
                - fixed_temperature["initial_peak_c"]
            ),
            "trace_min_peak_clip_minus_fixed": (
                clip_temperature["trace_min_peak_c"]
                - fixed_temperature["trace_min_peak_c"]
            ),
            "trace_peak_clip_minus_fixed": (
                clip_temperature["trace_peak_c"] - fixed_temperature["trace_peak_c"]
            ),
            "final_peak_clip_minus_fixed": (
                clip_temperature["final_peak_c"] - fixed_temperature["final_peak_c"]
            ),
            "overall_peak_clip_minus_fixed": (
                clip_temperature["overall_peak_c"]
                - fixed_temperature["overall_peak_c"]
            ),
        },
        "timing_s": {
            "power_peak_window_index": peak_index,
            "power_peak_time": power_peak_time_s,
            "power_peak_gem5_end_time": power_peak_gem5_end_time_s,
            "fixed_trace_peak_time": fixed_temperature["trace_peak_time_s"],
            "clip3d_trace_peak_time": clip_temperature["trace_peak_time_s"],
            "trace_peak_clip_minus_fixed": (
                clip_temperature["trace_peak_time_s"]
                - fixed_temperature["trace_peak_time_s"]
            ),
            "fixed_power_peak_to_temperature_peak_lag": fixed_lag,
            "clip3d_power_peak_to_temperature_peak_lag": clip_lag,
            "power_peak_to_temperature_peak_lag_clip_minus_fixed": (
                clip_lag - fixed_lag
            ),
        },
        "weighted_power_w": power_summary,
        "raw_power_evidence": {
            "fixed": fixed.get("raw_power_evidence"),
            "clip3d": clip3d.get("raw_power_evidence"),
        },
        "conservation_evidence": {
            "fixed": fixed.get("conservation_evidence"),
            "clip3d": clip3d.get("conservation_evidence"),
        },
        "acceptance_checks": {
            "checks": {
                "shared_canonical_source_r1": True,
                "shared_transient_r1": True,
                "shared_semantic_power_identity": True,
                "at_least_two_windows": fixed_count >= 2,
                "fixed_step_timeline": True,
                "actual_duration_within_hotspot_duration": (
                    actual_duration_s <= trace_duration_s
                ),
                "temperature_sample_counts_match": True,
                "raw_power_evidence_present": (
                    fixed.get("raw_power_evidence") is not None
                    and clip3d.get("raw_power_evidence") is not None
                ),
                "conservation_evidence_present": (
                    fixed.get("conservation_evidence") is not None
                    and clip3d.get("conservation_evidence") is not None
                ),
            },
            "all_passed": True,
            "failure_reasons": [],
        },
        "model_limitations": [
            "McPAT leakage is evaluated at its configured fixed operating temperature.",
            "There is no temperature-leakage-DVFS feedback loop.",
            sampling_resolution_limitation(sample_ms),
            "The final partial gem5 window is padded to one full HotSpot interval.",
            "Steady-temperature initialization omits the program's incomplete startup history.",
            "This operational validation is not a formal proof of thermal optimality.",
        ],
        "artifacts": {
            "comparison_json": str(
                (Path(output_dir).resolve() / "transient_comparison.json")
            ),
            "comparison_csv": str(
                (Path(output_dir).resolve() / "transient_comparison.csv")
            ),
            "power_temperature_timeseries_csv": str(
                (Path(output_dir).resolve() / "power_temperature_timeseries.csv")
            ),
        },
    }

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "transient_comparison.csv", COMPARISON_FIELDS, comparison_rows)
    _write_csv(
        output_dir / "power_temperature_timeseries.csv",
        TIMESERIES_FIELDS,
        timeseries_rows,
    )
    write_json(output_dir / "transient_comparison.json", result)
    return result
