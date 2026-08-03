"""Pure validation and summary helpers for transient power windows."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path


POWER_FIELDS = ("dynamic_power_w", "leakage_power_w", "total_power_w")
_NEGATIVE_TOLERANCE = -1e-12
_OMIT = object()


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


def _positive_number(record: dict, field: str, context: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context}: {field} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{context}: {field} must be a finite positive number")
    return value


def _validate_basic_window_timeline(windows: list[dict]) -> dict:
    """Validate fields that do not require independent run metadata."""
    if len(windows) < 2:
        raise ValueError("manifest: at least two transient windows are required")

    total_duration_s = 0.0
    total_duration_ticks = 0
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
            raise ValueError(f"{context}: ticks must be integers")
        if not isinstance(start_tick, int) or not isinstance(end_tick, int):
            raise ValueError(f"{context}: ticks must be integers")
        if end_tick <= start_tick:
            raise ValueError(f"{context}: end_tick must be greater than start_tick")
        if previous_end_tick is not None and start_tick != previous_end_tick:
            raise ValueError(
                f"window timeline gap: {context} start_tick must equal previous end_tick"
            )
        duration_ticks = end_tick - start_tick
        recorded_duration_ticks = window.get("duration_ticks", duration_ticks)
        if recorded_duration_ticks != duration_ticks:
            raise ValueError(f"{context}: duration_ticks does not match tick endpoints")
        total_duration_s += _duration(window, context)
        total_duration_ticks += duration_ticks
        previous_end_tick = end_tick
        first_tick = start_tick if first_tick is None else first_tick
        last_tick = end_tick

    return {
        "window_count": len(windows),
        "total_duration_s": total_duration_s,
        "total_duration_ticks": total_duration_ticks,
        "first_tick": first_tick,
        "last_tick": last_tick,
    }


def validate_window_timeline(manifest: dict) -> dict:
    """Validate fixed-step windows against independent measured-ROI endpoints."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("windows"), list):
        raise ValueError("manifest: windows must be a list")
    windows = manifest["windows"]
    audit = _validate_basic_window_timeline(windows)
    nominal_ticks_value = manifest.get("nominal_sample_interval_ticks")
    if (
        isinstance(nominal_ticks_value, bool)
        or not isinstance(nominal_ticks_value, int)
        or nominal_ticks_value <= 0
    ):
        raise ValueError(
            "manifest: nominal_sample_interval_ticks must be a positive integer"
        )
    nominal_ticks = nominal_ticks_value
    if "nominal_sample_interval_s" in manifest:
        nominal_s = _positive_number(
            manifest, "nominal_sample_interval_s", "manifest"
        )
    else:
        nominal_s = _positive_number(
            manifest, "nominal_sample_interval_ms", "manifest"
        ) / 1000.0

    measurement_start = manifest.get("measurement_start_tick")
    measurement_end = manifest.get("measurement_end_tick")
    if (
        isinstance(measurement_start, bool)
        or isinstance(measurement_end, bool)
        or not isinstance(measurement_start, int)
        or not isinstance(measurement_end, int)
        or measurement_end <= measurement_start
    ):
        raise ValueError(
            "manifest: independent measurement start/end ticks must be valid integers"
        )
    if audit["first_tick"] != measurement_start:
        raise ValueError("window first tick does not match measurement start tick")
    if audit["last_tick"] != measurement_end:
        raise ValueError("window last tick does not match measurement end tick")

    for index, window in enumerate(windows):
        context = f"window {index}"
        duration_ticks = int(window["end_tick"]) - int(window["start_tick"])
        duration_s = float(window["duration_s"])
        expected_duration_s = duration_ticks * nominal_s / nominal_ticks
        if not math.isclose(
            duration_s, expected_duration_s, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"{context}: duration_s does not match tick duration")
        if index < len(windows) - 1:
            if duration_ticks != nominal_ticks or not math.isclose(
                duration_s, nominal_s, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(
                    f"non-final {context} must equal the nominal sampling interval"
                )
        elif duration_ticks > nominal_ticks or duration_s > nominal_s:
            raise ValueError(
                "final window must be positive and no longer than the nominal interval"
            )

    independent_duration_ticks = measurement_end - measurement_start
    if audit["total_duration_ticks"] != independent_duration_ticks:
        raise ValueError("total ROI ticks do not match independent start/end ticks")
    independent_duration_s = independent_duration_ticks * nominal_s / nominal_ticks
    if not math.isclose(
        audit["total_duration_s"], independent_duration_s,
        rel_tol=1e-12, abs_tol=1e-15,
    ):
        raise ValueError("total ROI duration does not match independent start/end ticks")
    hotspot_duration_s = len(windows) * nominal_s
    if audit["total_duration_s"] > hotspot_duration_s and not math.isclose(
        audit["total_duration_s"], hotspot_duration_s,
        rel_tol=1e-12, abs_tol=1e-15,
    ):
        raise ValueError("actual duration exceeds HotSpot trace duration")

    return {
        **audit,
        "nominal_sample_interval_ticks": nominal_ticks,
        "nominal_sample_interval_s": nominal_s,
        "independent_roi_duration_ticks": independent_duration_ticks,
        "independent_roi_duration_s": independent_duration_s,
        "hotspot_trace_duration_s": hotspot_duration_s,
        "padded_final_duration_s": hotspot_duration_s - audit["total_duration_s"],
    }


def summarize_power_windows(windows: list[dict]) -> dict:
    """Return duration-weighted power statistics for transient windows."""
    timeline = _validate_basic_window_timeline(windows)
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


def validate_power_windows(manifest: dict,
                           expected_module_names: set[str] | None = None) -> dict:
    """Validate timeline, module triplets, and module-to-window aggregates."""
    timeline = validate_window_timeline(manifest)
    module_names: list[str] | None = None
    for index, window in enumerate(manifest["windows"]):
        modules = window.get("modules")
        if not isinstance(modules, list) or not modules:
            raise ValueError(f"window {index}: modules must be a non-empty list")
        names = [module.get("name") for module in modules]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"window {index}: module names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"window {index}: module names must be unique")
        if module_names is None:
            module_names = names
        elif names != module_names:
            raise ValueError(f"window {index}: module names or order changed")
        if expected_module_names is not None and set(names) != expected_module_names:
            missing = sorted(expected_module_names - set(names))
            extra = sorted(set(names) - expected_module_names)
            raise ValueError(
                f"window {index} module mismatch: missing={missing}, extra={extra}"
            )
        for module in modules:
            validate_power_triplet(module, f"window {index} module {module['name']}")
        totals = window.get("totals")
        validate_power_triplet(totals, f"window {index} totals")
        for field in POWER_FIELDS:
            aggregate = sum(float(module[field]) for module in modules)
            if not math.isclose(
                aggregate, float(totals[field]), rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(
                    f"window {index} aggregate module power for {field} "
                    "does not match window totals"
                )
    return {**timeline, "module_names": module_names or []}


def _location_independent(value: object, key: str = "") -> object:
    """Remove filesystem locations while retaining scientific run settings."""
    lowered = key.lower()
    if "path" in lowered or lowered.endswith("_dir") or lowered == "directory":
        return _OMIT
    if isinstance(value, str) and Path(value).is_absolute():
        return _OMIT
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in value.items():
            normalized = _location_independent(child_value, str(child_key))
            if normalized is not _OMIT:
                result[child_key] = normalized
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            normalized = _location_independent(child)
            if normalized is not _OMIT:
                result.append(normalized)
        return result
    return value


def canonical_power_payload(manifest: dict) -> dict:
    """Return only location-independent scientific power-trace content."""
    audit = validate_power_windows(manifest)
    run_settings = manifest.get("run_settings")
    if not isinstance(run_settings, dict):
        raise ValueError("power trace run_settings must be an object")
    windows = []
    for window in manifest["windows"]:
        source_hash = window.get("source_stats_sha256")
        if not isinstance(source_hash, str) or not source_hash:
            raise ValueError("power trace window lacks source statistics hash")
        windows.append({
            "index": int(window["index"]),
            "start_tick": int(window["start_tick"]),
            "end_tick": int(window["end_tick"]),
            "duration_ticks": int(window["duration_ticks"]),
            "duration_s": float(window["duration_s"]),
            "is_partial": bool(window.get("is_partial", False)),
            "source_stats_sha256": source_hash,
            "modules": [
                {
                    "name": module["name"],
                    **{field: float(module[field]) for field in POWER_FIELDS},
                }
                for module in window["modules"]
            ],
            "totals": {
                field: float(window["totals"][field]) for field in POWER_FIELDS
            },
        })
    return {
        "schema_version": 1,
        "nominal_sample_interval_ms": float(
            manifest["nominal_sample_interval_ms"]
        ),
        "nominal_sample_interval_ticks": int(
            manifest["nominal_sample_interval_ticks"]
        ),
        "measurement_start_tick": int(manifest["measurement_start_tick"]),
        "measurement_end_tick": int(manifest["measurement_end_tick"]),
        "window_count": audit["window_count"],
        "run_settings": _location_independent(run_settings),
        "module_names": audit["module_names"],
        "windows": windows,
    }


def power_trace_identity(manifest: dict) -> str:
    """Hash canonical scientific content, excluding elapsed time and paths."""
    payload = canonical_power_payload(manifest)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
