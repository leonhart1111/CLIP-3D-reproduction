#!/usr/bin/env python3
"""Real-HotSpot periodic-steady-state sustainable-frequency primitives."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.transient.generate_hotspot_trace import materialize_trace
from workflow.transient.run_hotspot_transient import (
    DEFAULT_HOTSPOT, parse_ttrace_grid, run_hotspot_transient,
    summarize_period_end_convergence,
)
from workflow.transient.run_transient_pipeline import validate_steady_output
from workflow.transient.validation import validate_power_windows


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return value


def _frequencies(values: list[float]) -> list[float]:
    result = sorted({_positive(value, "frequencies_ghz") for value in values})
    if not result:
        raise ValueError("frequencies_ghz must not be empty")
    return result


def last_period_peak(rows_k: list[list[float]], windows_per_period: int) -> dict:
    """Return the final period peak, including its initial (previous-end) state."""
    if (isinstance(windows_per_period, bool)
            or not isinstance(windows_per_period, int)
            or windows_per_period < 1):
        raise ValueError("windows_per_period must be a positive integer")
    if len(rows_k) <= windows_per_period or len(rows_k) % windows_per_period:
        raise ValueError("at least two complete temperature periods are required")
    width = len(rows_k[0])
    if width < 1 or any(len(row) != width for row in rows_k):
        raise ValueError("temperature rows must have one consistent nonzero width")
    if any(not math.isfinite(value) for row in rows_k for value in row):
        raise ValueError("temperature rows contain a non-finite value")
    start = len(rows_k) - windows_per_period - 1
    sample, unit = max(
        ((row, column) for row in range(start, len(rows_k))
         for column in range(width)), key=lambda item: rows_k[item[0]][item[1]]
    )
    return {
        "tmax_c": rows_k[sample][unit] - 273.15,
        "sample_index": sample, "unit_index": unit,
        "includes_period_initial_state": True,
        "phase": "period_initial_state" if sample == start else "window_end",
        "window_index_in_period": sample - start - 1,
    }


def find_sustainable_frequency(evaluate: Callable[[float], dict],
                               frequencies_ghz: list[float], tsafe_c: float,
                               frequency_tolerance_ghz: float) -> dict:
    """Grid-search and locally refine every observed safe/unsafe boundary."""
    if not math.isfinite(tsafe_c):
        raise ValueError("tsafe_c must be finite")
    tolerance = _positive(frequency_tolerance_ghz, "frequency_tolerance_ghz")
    grid = _frequencies(frequencies_ghz)
    evaluations: dict[float, dict] = {}

    def at(frequency: float) -> dict:
        if frequency not in evaluations:
            value = evaluate(frequency)
            if not isinstance(value, dict) or value.get("frequency_ghz") != frequency:
                raise ValueError("frequency evaluator recorded a different frequency")
            if not isinstance(value.get("converged"), bool):
                raise ValueError("frequency evaluator must record boolean convergence")
            peak = value.get("last_period_peak_c")
            if not isinstance(peak, (int, float)) or not math.isfinite(peak):
                raise ValueError("frequency evaluator must record a finite period peak")
            evaluations[frequency] = {**value, "safe": value["converged"] and peak <= tsafe_c}
        return evaluations[frequency]

    grid_values = [at(frequency) for frequency in grid]
    brackets = []
    for lower, upper in zip(grid, grid[1:]):
        lower_value, upper_value = at(lower), at(upper)
        if lower_value["safe"] == upper_value["safe"]:
            continue
        lower_safe = lower_value["safe"]
        lo, hi = lower, upper
        while hi - lo > tolerance:
            midpoint = (lo + hi) / 2.0
            if at(midpoint)["safe"] == lower_safe:
                lo = midpoint
            else:
                hi = midpoint
        brackets.append({
            "grid_lower_ghz": lower, "grid_upper_ghz": upper,
            "lower_is_safe": lower_safe,
            "refined_safe_frequency_ghz": lo if lower_safe else hi,
            "refined_unsafe_frequency_ghz": hi if lower_safe else lo,
            "final_bracket_width_ghz": hi - lo,
            "local_boundary_assumption": True,
        })
    safe = [frequency for frequency, value in evaluations.items() if value["safe"]]
    sustainable = max(safe) if safe else None
    floor_infeasible = not grid_values[0]["safe"]
    state = ("thermally_infeasible" if sustainable is None else
             "thermally_infeasible_at_fmin" if floor_infeasible else
             "thermal_headroom_within_grid" if grid_values[-1]["safe"] else
             "thermally_limited")
    return {
        "frequency_grid_ghz": grid, "tsafe_c": tsafe_c,
        "frequency_tolerance_ghz": tolerance,
        "monotonic_grid_safe_to_unsafe": not any(
            not left["safe"] and right["safe"]
            for left, right in zip(grid_values, grid_values[1:])
        ),
        "thermally_infeasible_at_fmin": floor_infeasible,
        "sustainable_frequency_ghz": sustainable, "state": state,
        "safe_unsafe_brackets": brackets, "grid_evaluations": grid_values,
        "evaluations": [evaluations[key] for key in sorted(evaluations)],
    }


def search_layout_frequency(modules_path: Path, layout_path: Path,
                            power_windows_path: Path, output_dir: Path,
                            config_path: Path, *, frequencies_ghz: list[float],
                            period_repeats: int, pss_tolerance_c: float,
                            frequency_tolerance_ghz: float,
                            hotspot: Path = DEFAULT_HOTSPOT) -> dict:
    """Search a supplied layout; no steady-run, architecture, or IPC2 dependency."""
    modules_path, layout_path = modules_path.resolve(), layout_path.resolve()
    power_windows_path, output_dir = power_windows_path.resolve(), output_dir.resolve()
    config_path, hotspot = config_path.resolve(), hotspot.resolve()
    for path in (modules_path, layout_path, power_windows_path, config_path, hotspot):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"transient verifier output directory is not empty: {output_dir}")
    if not isinstance(period_repeats, int) or period_repeats < 2:
        raise ValueError("period_repeats must be an integer of at least two")
    tolerance = _positive(pss_tolerance_c, "pss_tolerance_c")
    config = read_json(config_path)
    frequency = config.get("frequency", {})
    f0, tsafe = _positive(frequency.get("f0_ghz"), "frequency.f0_ghz"), _positive(frequency.get("tsafe_c"), "frequency.tsafe_c")
    grid = _frequencies(frequencies_ghz)
    modules = read_json(modules_path).get("modules")
    if not isinstance(modules, list):
        raise ValueError("modules must contain a module list")
    validate_power_windows(read_json(power_windows_path), {item.get("name") for item in modules})
    output_dir.mkdir(parents=True)

    def evaluate(frequency_ghz: float) -> dict:
        case = output_dir / f"frequency_{frequency_ghz.hex()}_ghz"
        trace = materialize_trace(modules_path, layout_path, power_windows_path, case,
                                  config, frequency_ghz / f0, period_repeats)
        thermal = run_hotspot_transient(case, hotspot=hotspot, initial_temperature="ambient")
        names, rows = parse_ttrace_grid(case / "transient.ttrace")
        convergence = summarize_period_end_convergence(rows, trace["windows_per_period"])
        peak = last_period_peak(rows, trace["windows_per_period"])
        return {"frequency_ghz": frequency_ghz, "converged": convergence["period_count"] >= 2 and convergence["last_delta_max_c"] <= tolerance,
                "last_period_peak_c": peak["tmax_c"], "last_period_peak_unit": names[peak["unit_index"]],
                "period_end_convergence": convergence, "trace_peak_c": thermal["trace_peak"]["tmax_c"]}

    search = find_sustainable_frequency(evaluate, grid, tsafe, frequency_tolerance_ghz)
    result = {"schema_version": 1, "mode": "operational transient sustainable-frequency verification", "non_formal": True, "paper_equivalent": False,
              "modules": str(modules_path), "layout": str(layout_path), "power_windows": str(power_windows_path), "config": str(config_path), "hotspot": str(hotspot),
              "frequency": {"f0_ghz": f0, "tsafe_c": tsafe, "grid_ghz": grid}, "periodic_steady_state": {"period_repeats": period_repeats, "pss_tolerance_c": tolerance},
              "search": search, "state": search["state"], "f_sus_trans_ghz": search["sustainable_frequency_ghz"]}
    write_json(output_dir / "transient_sustainable_frequency.json", result)
    return result


def verify_layout(steady_output_dir: Path, power_windows_path: Path, output_dir: Path,
                  config_path: Path, frequencies_ghz: list[float] | None = None,
                  period_repeats: int = 2, pss_tolerance_c: float = 0.01,
                  frequency_tolerance_ghz: float = 0.01,
                  hotspot: Path = DEFAULT_HOTSPOT) -> dict:
    """Compatibility wrapper that preflights a steady result and attaches IPC2."""
    steady_output_dir = steady_output_dir.resolve()
    power_windows = read_json(power_windows_path)
    source_r1 = power_windows.get("canonical_source_r1")
    if not isinstance(source_r1, str) or not source_r1:
        raise ValueError("power windows lack canonical_source_r1 provenance")
    config = read_json(config_path)
    audit = validate_steady_output(
        steady_output_dir, source_r1_dir=Path(source_r1), config=config,
        config_path=config_path,
    )
    summary = audit["summary"]
    ipc2 = _positive(summary.get("ipc2"), "pipeline_summary.ipc2")
    f = config["frequency"]
    grid = frequencies_ghz or [_positive(f["fmin_ghz"], "frequency.fmin_ghz"), _positive(f["f0_ghz"], "frequency.f0_ghz")]
    result = search_layout_frequency(steady_output_dir / "modules.json", steady_output_dir / "hotspot/layout.json", power_windows_path, output_dir, config_path,
                                     frequencies_ghz=grid, period_repeats=period_repeats, pss_tolerance_c=pss_tolerance_c, frequency_tolerance_ghz=frequency_tolerance_ghz, hotspot=hotspot)
    result.update({"steady_output_dir": str(steady_output_dir), "ipc2": ipc2, "bips2_trans": None if result["f_sus_trans_ghz"] is None else ipc2 * result["f_sus_trans_ghz"], "provenance_audit": audit})
    write_json(output_dir / "transient_sustainable_frequency.json", result)
    return result
