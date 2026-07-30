#!/usr/bin/env python3
"""Numerically validate equation (13) with scaled-power HotSpot runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.thermal.run_hotspot import run_hotspot
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
        "\t".join(f"{value * scale:.12g}" for value in values) + "\n",
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
        "\t".join(f"{value:.12g}" for value in composed_values) + "\n",
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
        "scaling_mode": scaling_mode,
    }
    if frequency_settings:
        settings.update(frequency_settings)
    for key in ("f0_ghz", "fmin_ghz", "tsafe_c"):
        settings[key] = float(settings[key])
        if not math.isfinite(settings[key]):
            raise ValueError(f"{key} must be finite")
    if settings["f0_ghz"] <= 0 or settings["fmin_ghz"] <= 0:
        raise ValueError("f0_ghz and fmin_ghz must be positive")
    if settings["fmin_ghz"] > settings["f0_ghz"]:
        raise ValueError("fmin_ghz must not exceed f0_ghz")
    if settings["scaling_mode"] not in {
        "separated-dynamic-leakage", "paper-uniform-gamma",
    }:
        raise ValueError(f"unsupported frequency scaling mode: {settings['scaling_mode']}")
    return settings


def validate_case(case_dir: Path, modules_path: Path, output: Path,
                  frequencies_ghz: list[float] | None = None,
                  validate_solution: bool = True,
                  frequency_settings: dict | None = None,
                  scaling_mode: str = "separated-dynamic-leakage") -> dict:
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

    def run_frequency(frequency: float, result_suffix: str = "") -> dict:
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
        thermal = run_hotspot(
            case_dir, ptrace_name=trace.name,
            result_name=(
                f"thermal_{selected_scaling_mode.replace('-', '_')}_"
                f"{stem}GHz{result_suffix}.json"
            ),
        )
        uniform_gamma_tmax = ambient + uniform_gamma_scale * (
            float(base_thermal["tmax_c"]) - ambient
        )
        return {
            "frequency_ghz": frequency,
            "dynamic_scale": trace_info["dynamic_scale"],
            "hotspot_tmax_c": thermal["tmax_c"],
            "predicted_tmax_c": uniform_gamma_tmax,
            "power_trace": trace_info["power_trace"],
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

    fsus, state, raw = closed_form_frequency(
        float(base_thermal["tmax_c"]), gamma, f0, fmin, tsafe, ambient
    )
    solution_validation = None
    if validate_solution and fsus < f0:
        solution_run = run_frequency(fsus, "_fsus")
        hotspot_tmax = float(solution_run["hotspot_tmax_c"])
        safe_error = abs(hotspot_tmax - tsafe) if math.isfinite(hotspot_tmax) else math.inf
        solution_validation = {
            "frequency_ghz": fsus, "hotspot_tmax_c": hotspot_tmax,
            "safe_temperature_c": tsafe,
            "safe_error_c": safe_error,
            "accepted": math.isfinite(hotspot_tmax) and safe_error <= 1.0,
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
    else:
        recommendation_basis = "below-f0 HotSpot safety error is finite and no greater than 1.0 C"

    result = {
        "schema_version": 1, "equations": [11, 12, 13],
        "case_dir": str(case_dir), "modules": str(modules_path.resolve()),
        "r_convec_k_per_w": manifest["r_convec_k_per_w"],
        "ambient_c": ambient, "gamma": gamma, "f0_ghz": f0,
        "base_tmax_c": base_thermal["tmax_c"], "frequencies": runs,
        "max_abs_linear_error_c": max(
            abs(run["uniform_gamma_comparison"]["error_vs_hotspot_c"])
            for run in runs
        ),
        "sustainable_frequency_ghz": fsus, "frequency_state": state,
        "unclamped_frequency_ghz": raw, "solution_validation": solution_validation,
        "frequency_settings": settings,
        "scaling_mode": selected_scaling_mode,
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
    args = parser.parse_args()
    result = validate_case(
        args.case_dir, args.modules, args.output, args.frequencies_ghz,
        not args.no_solution_validation,
    )
    print(f"max linear error={result['max_abs_linear_error_c']:.6f} C")


if __name__ == "__main__":
    main()
