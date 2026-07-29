#!/usr/bin/env python3
"""Numerically validate equation (13) with scaled-power HotSpot runs."""

from __future__ import annotations

import argparse
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
    return names, values


def write_scaled_ptrace(source: Path, destination: Path, scale: float) -> None:
    names, values = read_ptrace(source)
    destination.write_text(
        "\t".join(names) + "\n" +
        "\t".join(f"{value * scale:.12g}" for value in values) + "\n",
        encoding="utf-8",
    )


def safe_stem(frequency_ghz: float) -> str:
    return (f"{frequency_ghz:.6f}".rstrip("0").rstrip(".").replace(".", "p"))


def validate_case(case_dir: Path, modules_path: Path, output: Path,
                  frequencies_ghz: list[float] | None = None,
                  validate_solution: bool = True) -> dict:
    case_dir = case_dir.resolve()
    modules = read_json(modules_path)
    manifest = read_json(case_dir / "hotspot_manifest.json")
    base_thermal = read_json(case_dir / "thermal_result.json")
    gamma = float(modules["gamma"])
    ambient = float(manifest["ambient_c"])
    f0 = 2.0
    tsafe = 95.0
    fmin = 0.4
    frequencies = frequencies_ghz or [0.5, 1.0, 2.0]
    if any(frequency <= 0 or frequency > f0 for frequency in frequencies):
        raise ValueError(f"validation frequencies must be in (0,{f0}]")

    runs = []
    for frequency in frequencies:
        scale = gamma + (1.0 - gamma) * frequency / f0
        stem = safe_stem(frequency)
        trace = case_dir / f"power_uniform_gamma_{stem}GHz.ptrace"
        write_scaled_ptrace(case_dir / "power.ptrace", trace, scale)
        thermal = run_hotspot(
            case_dir, ptrace_name=trace.name,
            result_name=f"thermal_uniform_gamma_{stem}GHz.json",
        )
        predicted = ambient + scale * (float(base_thermal["tmax_c"]) - ambient)
        runs.append({
            "frequency_ghz": frequency, "power_scale": scale,
            "hotspot_tmax_c": thermal["tmax_c"], "predicted_tmax_c": predicted,
            "linear_error_c": thermal["tmax_c"] - predicted,
            "power_trace": str(trace.resolve()),
        })

    fsus, state, raw = closed_form_frequency(
        float(base_thermal["tmax_c"]), gamma, f0, fmin, tsafe, ambient
    )
    solution_validation = None
    if validate_solution and fsus < f0:
        scale = gamma + (1.0 - gamma) * fsus / f0
        stem = safe_stem(fsus)
        trace = case_dir / f"power_uniform_gamma_fsus_{stem}GHz.ptrace"
        write_scaled_ptrace(case_dir / "power.ptrace", trace, scale)
        thermal = run_hotspot(
            case_dir, ptrace_name=trace.name,
            result_name=f"thermal_uniform_gamma_fsus_{stem}GHz.json",
        )
        solution_validation = {
            "frequency_ghz": fsus, "hotspot_tmax_c": thermal["tmax_c"],
            "safe_temperature_c": tsafe,
            "safe_error_c": abs(thermal["tmax_c"] - tsafe),
            "power_scale": scale, "power_trace": str(trace.resolve()),
        }

    result = {
        "schema_version": 1, "equations": [11, 12, 13],
        "case_dir": str(case_dir), "modules": str(modules_path.resolve()),
        "r_convec_k_per_w": manifest["r_convec_k_per_w"],
        "ambient_c": ambient, "gamma": gamma, "f0_ghz": f0,
        "base_tmax_c": base_thermal["tmax_c"], "frequencies": runs,
        "max_abs_linear_error_c": max(abs(run["linear_error_c"]) for run in runs),
        "sustainable_frequency_ghz": fsus, "frequency_state": state,
        "unclamped_frequency_ghz": raw, "solution_validation": solution_validation,
        "scaling_mode": "paper uniform-gamma total-power scaling",
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
