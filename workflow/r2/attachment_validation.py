#!/usr/bin/env python3
"""Shared physical-coherence checks and point attachment locking."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path

from workflow.common import aggregate_ipc, parse_gem5_stats_text
from workflow.thermal.run_hotspot import parse_temperatures
from workflow.thermal.sustainable_frequency import derive


_NO_R2 = object()


def _finite_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _matches_expected(value: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return value is expected
    if isinstance(expected, int):
        return (isinstance(value, int) and not isinstance(value, bool)
                and value == expected)
    if isinstance(expected, float):
        return (_finite_number(value) and math.isclose(
            float(value), expected, rel_tol=1e-12, abs_tol=1e-12
        ))
    return value == expected


def _path_matches(value: object, expected: Path) -> bool:
    return (isinstance(value, str) and bool(value)
            and Path(value).resolve() == Path(expected).resolve())


@contextmanager
def exclusive_point_lock(point: Path):
    """Serialize a complete attachment transaction on one physical point."""
    descriptor = os.open(Path(point), os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def validate_physical_coherence(
        metadata: dict, config: dict, modules: dict, thermal: dict,
        performance: dict, summary: dict, r1_dir: Path, point_dir: Path,
        reasons: list[str], expected_ipc2: object = _NO_R2, *,
        r1_stats_bytes: bytes | None = None,
        thermal_steady_bytes: bytes | None = None,
        thermal_manifest: dict | None = None) -> dict | None:
    """Validate one physical input chain and all equation-(13) derived fields."""
    r1_dir = Path(r1_dir).resolve()
    point_dir = Path(point_dir).resolve()
    architecture = modules.get("architecture")
    if not isinstance(architecture, dict):
        reasons.append("target modules architecture is missing")
    else:
        for field in (
                "workload", "l1i_size", "l1d_size", "l2_size", "num_cores",
                "instruction_window_scope", "warmup_insts_cpu0",
                "measure_insts_cpu0"):
            expected = metadata.get(
                field, "cpu0" if field == "instruction_window_scope" else None
            )
            if architecture.get(field) != expected:
                reasons.append(
                    f"target modules architecture {field} differs from canonical R1"
                )
    if not _path_matches(modules.get("source_r1"), r1_dir):
        reasons.append("target modules source_r1 differs from canonical R1")
    totals = modules.get("totals")
    if not isinstance(totals, dict):
        reasons.append("target modules totals are missing")
    else:
        dynamic = totals.get("dynamic_power_w")
        leakage = totals.get("leakage_power_w")
        total = totals.get("total_power_w")
        if (not all(_finite_number(value) for value in (dynamic, leakage, total))
                or float(dynamic) < 0 or float(leakage) < 0 or float(total) <= 0
                or not math.isclose(
                    float(dynamic) + float(leakage), float(total),
                    rel_tol=1e-12, abs_tol=1e-12,
                )):
            reasons.append("target modules power totals are inconsistent")
        gamma = modules.get("gamma")
        if (not _finite_number(gamma) or not _finite_number(leakage)
                or not _finite_number(total) or float(total) <= 0
                or not math.isclose(
                    float(gamma), float(leakage) / float(total),
                    rel_tol=1e-12, abs_tol=1e-12,
                )):
            reasons.append("target modules gamma differs from leakage/total power")
        module_records = modules.get("modules")
        if not isinstance(module_records, list) or not module_records:
            reasons.append("target module power records are missing")
        else:
            sums = {field: 0.0 for field in (
                "dynamic_power_w", "leakage_power_w", "total_power_w"
            )}
            valid_records = True
            for record in module_records:
                if not isinstance(record, dict):
                    valid_records = False
                    break
                values = [record.get(field) for field in sums]
                if (not all(_finite_number(value) for value in values)
                        or any(float(value) < 0 for value in values)
                        or not math.isclose(
                            float(values[0]) + float(values[1]), float(values[2]),
                            rel_tol=1e-12, abs_tol=1e-12,
                        )):
                    valid_records = False
                    break
                for field, value in zip(sums, values):
                    sums[field] += float(value)
            if not valid_records:
                reasons.append("target module power records are inconsistent")
            else:
                for field, expected in sums.items():
                    if not _matches_expected(totals.get(field), expected):
                        reasons.append(
                            f"target modules totals {field} differs from module records"
                        )

    if r1_stats_bytes is None:
        reasons.append("captured R1 stats evidence is missing")
    else:
        try:
            num_cores = metadata.get("num_cores", 4)
            if (not isinstance(num_cores, int) or isinstance(num_cores, bool)
                    or num_cores <= 0):
                raise ValueError("num_cores must be a positive integer")
            stats = parse_gem5_stats_text(r1_stats_bytes.decode("utf-8"))
            expected_ipc1 = aggregate_ipc(stats, num_cores)
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            reasons.append(f"cannot derive target module IPC1 from R1 stats: {error}")
        else:
            if not _matches_expected(modules.get("ipc1"), expected_ipc1):
                reasons.append("target modules IPC1 differs from captured R1 stats")

    frequency = config.get("frequency")
    physical = config.get("physical")
    if not isinstance(frequency, dict) or not isinstance(physical, dict):
        reasons.append("experiment config lacks frequency/physical inputs")
        return None
    if not isinstance(thermal_manifest, dict):
        reasons.append("captured HotSpot manifest is missing")
    else:
        if not _matches_expected(
                thermal_manifest.get("ambient_c"), frequency.get("ambient_c")):
            reasons.append("target HotSpot manifest ambient differs from config")
        if not _matches_expected(
                thermal_manifest.get("r_convec_k_per_w"),
                physical.get("r_convec_k_per_w")):
            reasons.append("target HotSpot manifest cooling differs from config")
    tmax_c = thermal.get("tmax_c")
    if not _finite_number(tmax_c):
        reasons.append("target thermal tmax_c must be finite")
    for field, expected in (
            ("power_trace", point_dir / "hotspot/power.ptrace"),
            ("steady_file", point_dir / "hotspot/steady.txt"),
            ("grid_steady_file", point_dir / "hotspot/grid.steady.txt")):
        if not _path_matches(thermal.get(field), expected):
            reasons.append(
                f"target thermal {field} does not identify the target HotSpot output"
            )
    if thermal.get("return_code") != 0:
        reasons.append("target thermal return_code is not zero")
    if (not _finite_number(thermal.get("ambient_c"))
            or thermal.get("ambient_c") != frequency.get("ambient_c")):
        reasons.append("target thermal ambient differs from experiment config")
    if (not _finite_number(thermal.get("r_convec_k_per_w"))
            or thermal.get("r_convec_k_per_w")
            != physical.get("r_convec_k_per_w")):
        reasons.append("target thermal cooling differs from experiment config")
    if not _finite_number(thermal.get("tmax_k")):
        reasons.append("target thermal tmax_k must be finite")
    elif _finite_number(tmax_c) and not math.isclose(
            float(thermal["tmax_k"]) - 273.15, float(tmax_c),
            rel_tol=0.0, abs_tol=1e-9):
        reasons.append("target thermal Kelvin/Celsius values disagree")
    if thermal_steady_bytes is None:
        reasons.append("captured HotSpot steady evidence is missing")
    else:
        try:
            samples = parse_temperatures(thermal_steady_bytes.decode("utf-8"))
            peak_unit, peak_k = max(samples, key=lambda item: item[1])
        except (UnicodeDecodeError, ValueError) as error:
            reasons.append(f"cannot derive target thermal peak from steady output: {error}")
        else:
            if not _matches_expected(thermal.get("tmax_k"), peak_k):
                reasons.append("target thermal Tmax differs from captured steady output")
            if not _matches_expected(thermal.get("tmax_c"), peak_k - 273.15):
                reasons.append("target thermal Celsius peak differs from steady output")
            if thermal.get("peak_unit") != peak_unit:
                reasons.append("target thermal peak unit differs from steady output")
            if thermal.get("sample_count") != len(samples):
                reasons.append("target thermal sample count differs from steady output")

    try:
        ipc2 = None if expected_ipc2 is _NO_R2 else expected_ipc2
        expected_performance = derive(
            modules, thermal, frequency["f0_ghz"], frequency["fmin_ghz"],
            frequency["tsafe_c"], frequency["ambient_c"], ipc2,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        reasons.append(f"cannot derive target thermal performance: {error}")
        return None
    for field, expected in expected_performance.items():
        if not _matches_expected(performance.get(field), expected):
            reasons.append(
                f"target performance {field} differs from target module/thermal inputs"
            )
    summary_fields = (
        ("gamma", "gamma",
         "target summary gamma differs from target performance"),
        ("tmax_c", "tmax_f0_c",
         "target summary thermal tmax_c differs from target performance"),
        ("sustainable_frequency_ghz", "sustainable_frequency_ghz",
         "target summary sustainable_frequency_ghz differs from target performance"),
        ("ipc1", "ipc1",
         "target summary ipc1 differs from target performance"),
        ("bips1_thermal", "bips1_thermal",
         "target summary bips1_thermal differs from target performance"),
    )
    if expected_ipc2 is not _NO_R2:
        summary_fields += (
            ("ipc2", "ipc2",
             "target summary ipc2 differs from target performance"),
            ("bips2", "bips2",
             "target summary bips2 differs from target performance"),
        )
    for summary_field, performance_field, message in summary_fields:
        if not _matches_expected(
                summary.get(summary_field), expected_performance[performance_field]):
            reasons.append(message)
    return expected_performance
