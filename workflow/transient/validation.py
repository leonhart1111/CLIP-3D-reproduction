"""Pure validation and summary helpers for transient power windows."""

from __future__ import annotations

import math
from numbers import Real


POWER_FIELDS = ("dynamic_power_w", "leakage_power_w", "total_power_w")
_NEGATIVE_TOLERANCE = -1e-12


def _power_value(record: dict, field: str, context: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context}: {field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{context}: {field} must be a finite number")
    if value < _NEGATIVE_TOLERANCE:
        raise ValueError(f"{context}: {field} must not be negative")
    return value


def validate_power_triplet(record: dict, context: str) -> None:
    """Validate finite, non-negative, and conserving power values in *record*."""
    if not isinstance(record, dict):
        raise ValueError(f"{context}: power record must be a dictionary")
    dynamic, leakage, total = (
        _power_value(record, field, context) for field in POWER_FIELDS
    )
    if not math.isclose(
        total, dynamic + leakage, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(
            f"{context}: dynamic_power_w + leakage_power_w must equal total_power_w"
        )


def _duration(window: dict, context: str) -> float:
    value = window.get("duration_s")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context}: duration_s must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{context}: duration_s must be a finite positive number")
    return value


def validate_window_timeline(manifest: dict) -> dict:
    """Validate contiguous transient windows and return their timeline metadata."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("windows"), list):
        raise ValueError("manifest: windows must be a list")
    windows = manifest["windows"]
    if not windows:
        raise ValueError("manifest: windows must not be empty")

    total_duration_s = 0.0
    previous_end_tick = None
    first_tick = None
    last_tick = None
    for index, window in enumerate(windows):
        context = f"window {index}"
        if not isinstance(window, dict):
            raise ValueError(f"{context}: window must be a dictionary")
        start_tick = window.get("start_tick")
        end_tick = window.get("end_tick")
        if isinstance(start_tick, bool) or isinstance(end_tick, bool):
            raise ValueError(f"{context}: ticks must be numbers")
        if not isinstance(start_tick, Real) or not isinstance(end_tick, Real):
            raise ValueError(f"{context}: ticks must be numbers")
        if not math.isfinite(float(start_tick)) or not math.isfinite(float(end_tick)):
            raise ValueError(f"{context}: ticks must be finite")
        if end_tick <= start_tick:
            raise ValueError(f"{context}: end_tick must be greater than start_tick")
        if previous_end_tick is not None and start_tick != previous_end_tick:
            raise ValueError(
                f"window timeline gap: {context} start_tick must equal previous end_tick"
            )
        total_duration_s += _duration(window, context)
        previous_end_tick = end_tick
        first_tick = start_tick if first_tick is None else first_tick
        last_tick = end_tick

    return {
        "window_count": len(windows),
        "total_duration_s": total_duration_s,
        "first_tick": first_tick,
        "last_tick": last_tick,
    }


def summarize_power_windows(windows: list[dict]) -> dict:
    """Return duration-weighted power statistics for transient windows."""
    timeline = validate_window_timeline({"windows": windows})
    totals_by_field = {field: [] for field in POWER_FIELDS}
    elapsed_s = 0.0
    end_times_s = []
    for index, window in enumerate(windows):
        context = f"window {index} totals"
        totals = window.get("totals")
        validate_power_triplet(totals, context)
        duration_s = float(window["duration_s"])
        elapsed_s += duration_s
        end_times_s.append(elapsed_s)
        for field in POWER_FIELDS:
            totals_by_field[field].append(float(totals[field]))

    summary = {}
    for field, values in totals_by_field.items():
        peak_window_index = max(range(len(values)), key=values.__getitem__)
        summary[field] = {
            "min": min(values),
            "max": max(values),
            "weighted_mean": sum(
                value * float(window["duration_s"])
                for value, window in zip(values, windows)
            ) / timeline["total_duration_s"],
            "peak_window_index": peak_window_index,
            "peak_end_time_s": end_times_s[peak_window_index],
        }
    return summary
