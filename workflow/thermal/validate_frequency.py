#!/usr/bin/env python3
"""Numerically validate equation (13) with scaled-power HotSpot runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.thermal.run_hotspot import grid_temperatures, run_hotspot
from workflow.thermal.sustainable_frequency import closed_form_frequency


def read_ptrace(path: Path) -> tuple[list[str], list[float]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"expected one-sample power trace in {path}")
    names = lines[0].split()
    values = [float(value) for value in lines[1].split()]
    if len(names) != len(values):
        raise ValueError(f"power trace header/value mismatch in {path}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"power trace contains non-finite values in {path}")
    return names, values


def write_scaled_ptrace(source: Path, destination: Path, scale: float) -> None:
    names, values = read_ptrace(source)
    destination.write_text(
        "\t".join(names) + "\n" +
        "\t".join(f"{value * scale:.17g}" for value in values) + "\n",
        encoding="utf-8",
    )


def validate_total_trace(names: list[str], dynamic_values: list[float],
                         leakage_values: list[float], total_path: Path | None) -> float | None:
    """Verify that an available f0 total trace preserves every power cell."""
    if total_path is None or not total_path.is_file():
        return None
    total_names, total_values = read_ptrace(total_path)
    if names != total_names:
        raise ValueError("total power trace headers must match dynamic/leakage order")
    if len(total_values) != len(dynamic_values):
        raise ValueError("total power trace length must match dynamic/leakage")
    for index, (dynamic, leakage, total) in enumerate(
            zip(dynamic_values, leakage_values, total_values)):
        if abs(dynamic + leakage - total) > 1e-9:
            raise ValueError(
                f"total power trace disagrees with dynamic plus leakage at cell {index}"
            )
    return sum(total_values)


def compose_separated_ptrace(dynamic_path: Path, leakage_path: Path,
                             destination: Path, frequency_ghz: float,
                             f0_ghz: float, total_path: Path | None = None) -> dict:
    """Compose a one-sample trace with frequency-scaled dynamic power only."""
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0:
        raise ValueError("frequency_ghz must be finite and positive")
    if not math.isfinite(f0_ghz) or f0_ghz <= 0:
        raise ValueError("f0_ghz must be finite and positive")

    names, dynamic_values = read_ptrace(dynamic_path)
    leakage_names, leakage_values = read_ptrace(leakage_path)
    if names != leakage_names:
        raise ValueError("dynamic and leakage power trace headers must match in order")
    if len(dynamic_values) != len(leakage_values):
        raise ValueError("dynamic and leakage power trace lengths must match")

    total_trace_sum_w = validate_total_trace(
        names, dynamic_values, leakage_values, total_path,
    )

    dynamic_scale = frequency_ghz / f0_ghz
    composed_values = [
        leakage + dynamic_scale * dynamic
        for dynamic, leakage in zip(dynamic_values, leakage_values)
    ]
    destination.write_text(
        "\t".join(names) + "\n" +
        "\t".join(f"{value:.17g}" for value in composed_values) + "\n",
        encoding="utf-8",
    )
    return {
        "dynamic_scale": dynamic_scale,
        "dynamic_trace_sum_w": sum(dynamic_values),
        "leakage_trace_sum_w": sum(leakage_values),
        "composed_trace_sum_w": sum(composed_values),
        "total_trace_sum_w": total_trace_sum_w,
        "power_trace": str(destination.resolve()),
    }


def safe_stem(frequency_ghz: float) -> str:
    return (f"{frequency_ghz:.6f}".rstrip("0").rstrip(".").replace(".", "p"))


def resolve_frequency_settings(frequency_settings: dict | None,
                               scaling_mode: str) -> dict:
    """Return validated frequency assumptions for a raw-power validation."""
    settings = {
        "f0_ghz": 2.0,
        "fmin_ghz": 0.4,
        "tsafe_c": 95.0,
        # Table III reports <=0.02 C error for the paper's validation
        # anchors.  Do not let a degree-scale miss silently endorse the
        # global-gamma shortcut merely because it lands near the DTM limit.
        "max_safe_error_c": 0.02,
        "scaling_mode": scaling_mode,
    }
    if frequency_settings:
        settings.update(frequency_settings)
    for key in ("f0_ghz", "fmin_ghz", "tsafe_c", "max_safe_error_c"):
        settings[key] = float(settings[key])
        if not math.isfinite(settings[key]):
            raise ValueError(f"{key} must be finite")
    if settings["f0_ghz"] <= 0 or settings["fmin_ghz"] <= 0:
        raise ValueError("f0_ghz and fmin_ghz must be positive")
    if settings["fmin_ghz"] > settings["f0_ghz"]:
        raise ValueError("fmin_ghz must not exceed f0_ghz")
    if settings["max_safe_error_c"] < 0.0:
        raise ValueError("max_safe_error_c must be non-negative")
    if settings["scaling_mode"] not in {
        "separated-dynamic-leakage", "paper-uniform-gamma",
    }:
        raise ValueError(f"unsupported frequency scaling mode: {settings['scaling_mode']}")
    return settings


def module_gamma_observability(modules: dict) -> dict:
    """Expose whether Equation (11)'s uniform-gamma shortcut is plausible.

    Equation (13) itself uses the power-weighted scalar gamma.  The paper
    explicitly identifies a separated dynamic/leakage HotSpot calibration as
    the fallback when module leakage fractions differ materially, so report
    that precondition instead of silently treating the scalar as exact.
    """
    records = modules.get("modules")
    if not isinstance(records, list):
        return {"available": False, "reason": "modules list unavailable"}
    fractions: list[float] = []
    dynamic_total = 0.0
    leakage_total = 0.0
    for record in records:
        if not isinstance(record, dict):
            continue
        dynamic = float(record.get("dynamic_power_w", 0.0))
        leakage = float(record.get("leakage_power_w", 0.0))
        total = dynamic + leakage
        if not all(math.isfinite(value) and value >= 0.0
                   for value in (dynamic, leakage, total)):
            raise ValueError("module power must be finite and non-negative")
        dynamic_total += dynamic
        leakage_total += leakage
        if total > 0.0:
            fractions.append(leakage / total)
    total_power = dynamic_total + leakage_total
    if not fractions or total_power <= 0.0:
        return {"available": False, "reason": "no positive-power modules"}
    lower, upper = min(fractions), max(fractions)
    return {
        "available": True,
        "positive_power_module_count": len(fractions),
        "dynamic_power_w": dynamic_total,
        "leakage_power_w": leakage_total,
        "power_weighted_gamma": leakage_total / total_power,
        "module_gamma_range": [lower, upper],
        "module_gamma_spread": upper - lower,
    }


def two_point_affine_frequency(base_thermal: dict, reference_thermal: dict,
                               manifest: dict, reference_frequency_ghz: float,
                               frequency_settings: dict) -> dict:
    """Solve Equation (9) from two same-layout HotSpot grid maps.

    ``base_thermal`` is the nominal-``f0`` total-power solve and
    ``reference_thermal`` is a lower-frequency *separated*
    dynamic/leakage solve.  Their cellwise difference identifies the dynamic
    map, while their affine intercept identifies the leakage map.  This is the
    paper's one-extra-HotSpot fallback for a non-uniform leakage fraction; it
    deliberately does not change the inexpensive Equation-(14) search proxy.

    The result is optional because lightweight unit tests and old artifacts do
    not always retain grid-steady paths.  Such a missing map is reported as an
    unavailable fallback rather than silently substituting the global-gamma
    shortcut.
    """
    f0 = float(frequency_settings["f0_ghz"])
    fmin = float(frequency_settings["fmin_ghz"])
    tsafe_k = float(frequency_settings["tsafe_c"]) + 273.15
    if not math.isfinite(reference_frequency_ghz) or not 0.0 < reference_frequency_ghz < f0:
        return {
            "available": False,
            "reason": "requires one finite separated-power reference frequency in (0,f0)",
        }
    base_path = base_thermal.get("grid_steady_file")
    reference_path = reference_thermal.get("grid_steady_file")
    if not isinstance(base_path, str) or not isinstance(reference_path, str):
        return {
            "available": False,
            "reason": "nominal or reference HotSpot grid-steady path is unavailable",
        }
    base_file, reference_file = Path(base_path), Path(reference_path)
    if not base_file.is_file() or not reference_file.is_file():
        return {
            "available": False,
            "reason": "nominal or reference HotSpot grid-steady file is unavailable",
        }

    # run_hotspot's module-input Tmax is deliberately defined over active power
    # layers, so the two-point fallback must use the identical peak domain.
    active_layers = manifest.get("active_power_layers")
    if not isinstance(active_layers, list) or not active_layers:
        return {
            "available": False,
            "reason": "active_power_layers are unavailable in the HotSpot manifest",
        }
    active = {int(layer) for layer in active_layers}

    def active_map(path: Path) -> dict[str, float]:
        return {
            name: value for name, value in grid_temperatures(path)
            if int(name.split("_")[1]) in active
        }

    base_map = active_map(base_file)
    reference_map = active_map(reference_file)
    if not base_map or set(base_map) != set(reference_map):
        return {
            "available": False,
            "reason": "nominal and reference active grid cells do not match",
        }

    x_reference = reference_frequency_ghz / f0
    denominator = 1.0 - x_reference
    intercepts: dict[str, float] = {}
    slopes: dict[str, float] = {}
    for name in sorted(base_map):
        dynamic_at_f0 = (base_map[name] - reference_map[name]) / denominator
        intercepts[name] = base_map[name] - dynamic_at_f0
        slopes[name] = dynamic_at_f0

    # The cellwise affine temperature is A_i + (f/f0) B_i.  The maximum
    # admissible normalized frequency is the tightest positive-slope cell.
    limits: list[tuple[float, str]] = []
    infeasible_static: list[str] = []
    for name in sorted(intercepts):
        slope = slopes[name]
        intercept = intercepts[name]
        if slope > 1.0e-9:
            limits.append(((tsafe_k - intercept) / slope, name))
        elif intercept > tsafe_k + 1.0e-9:
            infeasible_static.append(name)
    if infeasible_static:
        raw_frequency = -math.inf
        limiting_unit = infeasible_static[0]
    elif limits:
        normalized_limit, limiting_unit = min(limits, key=lambda item: (item[0], item[1]))
        raw_frequency = f0 * normalized_limit
    else:
        raw_frequency = f0
        limiting_unit = None

    frequency = min(f0, max(fmin, raw_frequency))
    normalized_frequency = frequency / f0
    temperatures = {
        name: intercepts[name] + normalized_frequency * slopes[name]
        for name in intercepts
    }
    temperatures_at_fmin = {
        name: intercepts[name] + (fmin / f0) * slopes[name]
        for name in intercepts
    }
    peak_unit, peak_k = max(temperatures.items(), key=lambda item: item[1])
    if raw_frequency >= f0:
        state = "thermal_headroom"
    elif raw_frequency < fmin:
        state = "frequency_floor"
    else:
        state = "thermally_limited"

    return {
        "available": True,
        "equation": 9,
        "reference_frequency_ghz": reference_frequency_ghz,
        "reference_dynamic_scale": x_reference,
        "active_cell_count": len(base_map),
        "sustainable_frequency_ghz": frequency,
        "unclamped_frequency_ghz": raw_frequency,
        "frequency_state": state,
        "limiting_unit": limiting_unit,
        "predicted_tmax_at_sustainable_frequency_c": peak_k - 273.15,
        "predicted_peak_unit_at_sustainable_frequency": peak_unit,
        "thermal_feasible_at_fmin": (
            max(temperatures_at_fmin.values()) <= tsafe_k + 1.0e-9
        ),
        "leakage_intercept_tmax_c": max(intercepts.values()) - 273.15,
        "dynamic_slope_at_f0_range_k": [min(slopes.values()), max(slopes.values())],
        "ambient_c": float(manifest["ambient_c"]),
    }


def validate_case(case_dir: Path, modules_path: Path, output: Path,
                  frequencies_ghz: list[float] | None = None,
                  validate_solution: bool = True,
                  frequency_settings: dict | None = None,
                  scaling_mode: str = "separated-dynamic-leakage",
                  hotspot: Path | None = None) -> dict:
    case_dir = case_dir.resolve()
    modules = read_json(modules_path)
    manifest = read_json(case_dir / "hotspot_manifest.json")
    base_thermal = read_json(case_dir / "thermal_result.json")
    gamma = float(modules["gamma"])
    ambient = float(manifest["ambient_c"])
    settings = resolve_frequency_settings(frequency_settings, scaling_mode)
    f0 = settings["f0_ghz"]
    tsafe = settings["tsafe_c"]
    fmin = settings["fmin_ghz"]
    selected_scaling_mode = settings["scaling_mode"]
    frequencies = frequencies_ghz or [0.5, 1.0, 2.0]
    if any(not math.isfinite(frequency) or frequency <= 0 or frequency > f0
           for frequency in frequencies):
        raise ValueError(f"validation frequencies must be in (0,{f0}]")

    dynamic_trace = case_dir / "power_dynamic.ptrace"
    leakage_trace = case_dir / "power_leakage.ptrace"
    total_trace = case_dir / "power.ptrace"

    def run_frequency(frequency: float, result_suffix: str = "",
                      record_hotspot_failure: bool = False) -> dict:
        stem = safe_stem(frequency)
        uniform_gamma_scale = gamma + (1.0 - gamma) * frequency / f0
        if selected_scaling_mode == "separated-dynamic-leakage":
            trace = case_dir / (
                f"power_separated_dynamic_leakage_{stem}GHz{result_suffix}.ptrace"
            )
            trace_info = compose_separated_ptrace(
                dynamic_trace, leakage_trace, trace, frequency, f0, total_trace,
            )
        else:
            trace = case_dir / f"power_uniform_gamma_{stem}GHz{result_suffix}.ptrace"
            write_scaled_ptrace(total_trace, trace, uniform_gamma_scale)
            dynamic_names, dynamic_values = read_ptrace(dynamic_trace)
            leakage_names, leakage_values = read_ptrace(leakage_trace)
            if dynamic_names != leakage_names:
                raise ValueError("dynamic and leakage power trace headers must match in order")
            total_trace_sum_w = validate_total_trace(
                dynamic_names, dynamic_values, leakage_values, total_trace,
            )
            composed_names, composed_values = read_ptrace(trace)
            if composed_names != dynamic_names:
                raise ValueError("scaled power trace headers must match dynamic/leakage order")
            trace_info = {
                "dynamic_scale": frequency / f0,
                "dynamic_trace_sum_w": sum(dynamic_values),
                "leakage_trace_sum_w": sum(leakage_values),
                "composed_trace_sum_w": sum(composed_values),
                "total_trace_sum_w": total_trace_sum_w,
                "power_trace": str(trace.resolve()),
            }
        try:
            result_name = (
                f"thermal_{selected_scaling_mode.replace('-', '_')}_"
                f"{stem}GHz{result_suffix}.json"
            )
            # Keep the default call shape for existing callers and mocked
            # tests.  A supplied binary is essential when this diagnostic is
            # run from a lightweight worktree without built tool artifacts.
            if hotspot is None:
                thermal = run_hotspot(
                    case_dir, ptrace_name=trace.name, result_name=result_name,
                )
            else:
                thermal = run_hotspot(
                    case_dir, hotspot=hotspot, ptrace_name=trace.name,
                    result_name=result_name,
                )
        except Exception as error:
            if not record_hotspot_failure:
                raise
            return {
                "frequency_ghz": frequency,
                "power_trace": trace_info["power_trace"],
                "hotspot_error": f"{type(error).__name__}: {error}",
            }
        uniform_gamma_tmax = ambient + uniform_gamma_scale * (
            float(base_thermal["tmax_c"]) - ambient
        )
        return {
            "frequency_ghz": frequency,
            "dynamic_scale": trace_info["dynamic_scale"],
            "hotspot_tmax_c": thermal["tmax_c"],
            "predicted_tmax_c": uniform_gamma_tmax,
            "power_trace": trace_info["power_trace"],
            # Retain the map path solely for the optional Eq.(9) fallback
            # below.  It is not a new proxy input and does not affect the
            # paper's global-gamma result reported alongside it.
            "grid_steady_file": thermal.get("grid_steady_file"),
            "trace_sums_w": {
                "dynamic": trace_info["dynamic_trace_sum_w"],
                "leakage": trace_info["leakage_trace_sum_w"],
                "composed": trace_info["composed_trace_sum_w"],
                "total_at_f0": trace_info["total_trace_sum_w"],
            },
            "uniform_gamma_comparison": {
                "scaling_mode": "paper-uniform-gamma",
                "total_power_scale": uniform_gamma_scale,
                "closed_form_tmax_c": uniform_gamma_tmax,
                "error_vs_hotspot_c": thermal["tmax_c"] - uniform_gamma_tmax,
            },
        }

    runs = [run_frequency(frequency) for frequency in frequencies]
    separated_reference = next(
        (run for run in runs if run["frequency_ghz"] < f0), None
    )
    if selected_scaling_mode != "separated-dynamic-leakage":
        two_point = {
            "available": False,
            "reason": "Equation (9) fallback requires a separated dynamic/leakage reference trace",
        }
    elif separated_reference is None:
        two_point = {
            "available": False,
            "reason": "no requested separated-power reference frequency below f0",
        }
    else:
        two_point = two_point_affine_frequency(
            base_thermal, separated_reference, manifest,
            float(separated_reference["frequency_ghz"]), settings,
        )

    fsus, state, raw = closed_form_frequency(
        float(base_thermal["tmax_c"]), gamma, f0, fmin, tsafe, ambient
    )
    solution_validation = None
    if validate_solution and fsus < f0:
        solution_run = run_frequency(fsus, "_fsus", record_hotspot_failure=True)
        if "hotspot_error" in solution_run:
            solution_validation = {
                "frequency_ghz": fsus,
                "hotspot_tmax_c": None,
                "safe_temperature_c": tsafe,
                "safe_error_c": None,
                "max_safe_error_c": settings["max_safe_error_c"],
                "accepted": False,
                "power_trace": solution_run["power_trace"],
                "error": solution_run["hotspot_error"],
            }
        else:
            hotspot_tmax = float(solution_run["hotspot_tmax_c"])
            safe_error = (
                abs(hotspot_tmax - tsafe) if math.isfinite(hotspot_tmax) else math.inf
            )
            solution_validation = {
                "frequency_ghz": fsus, "hotspot_tmax_c": hotspot_tmax,
                "safe_temperature_c": tsafe,
                "safe_error_c": safe_error,
                "max_safe_error_c": settings["max_safe_error_c"],
                "accepted": (
                    math.isfinite(hotspot_tmax)
                    and safe_error <= settings["max_safe_error_c"]
                ),
                "power_trace": solution_run["power_trace"],
            }

    accepted = (
        fsus >= f0 or
        (solution_validation is not None and solution_validation["accepted"])
    )
    if fsus >= f0:
        recommendation_basis = "f_sus is at f0; no below-f0 HotSpot safety solve is required"
    elif solution_validation is None:
        recommendation_basis = "below-f0 HotSpot safety solve was skipped"
    elif solution_validation.get("error"):
        recommendation_basis = (
            "below-f0 HotSpot safety solve failed: "
            f"{solution_validation['error']}"
        )
    else:
        recommendation_basis = (
            "below-f0 HotSpot safety error is finite and no greater than "
            f"{settings['max_safe_error_c']:.6g} C"
        )

    result = {
        "schema_version": 1, "equations": [11, 12, 13],
        "case_dir": str(case_dir), "modules": str(modules_path.resolve()),
        "r_convec_k_per_w": manifest["r_convec_k_per_w"],
        "ambient_c": ambient, "gamma": gamma, "f0_ghz": f0,
        "base_tmax_c": base_thermal["tmax_c"], "frequencies": runs,
        "max_abs_uniform_gamma_comparison_error_c": max(
            abs(run["uniform_gamma_comparison"]["error_vs_hotspot_c"])
            for run in runs
        ),
        "sustainable_frequency_ghz": fsus, "frequency_state": state,
        "unclamped_frequency_ghz": raw, "solution_validation": solution_validation,
        "two_point_affine_frequency": two_point,
        "frequency_settings": settings,
        "scaling_mode": selected_scaling_mode,
        "module_gamma_observability": module_gamma_observability(modules),
        "recommendation": {
            "accepted": accepted,
            "basis": recommendation_basis,
        },
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequencies-ghz", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--no-solution-validation", action="store_true")
    parser.add_argument(
        "--hotspot", type=Path,
        help="optional HotSpot executable; needed when the current worktree has no built binary",
    )
    args = parser.parse_args()
    result = validate_case(
        args.case_dir, args.modules, args.output, args.frequencies_ghz,
        not args.no_solution_validation, hotspot=args.hotspot,
    )
    print(
        "max uniform-gamma comparison error="
        f"{result['max_abs_uniform_gamma_comparison_error_c']:.6f} C"
    )


if __name__ == "__main__":
    main()
