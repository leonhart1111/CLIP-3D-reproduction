#!/usr/bin/env python3
"""Evaluate CLIP-3D equation (13) and BIPS from one HotSpot result."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow.common import read_json, write_json


def closed_form_frequency(tmax_c: float, gamma: float, f0_ghz: float = 2.0,
                          fmin_ghz: float = 0.4, tsafe_c: float = 95.0,
                          ambient_c: float = 25.0) -> tuple[float, str, float]:
    if not 0 <= gamma < 1:
        raise ValueError(f"gamma must be in [0,1), got {gamma}")
    if tmax_c <= tsafe_c:
        return f0_ghz, "thermal_headroom", f0_ghz
    rise = tmax_c - ambient_c
    if rise <= 0:
        raise ValueError("Tmax must exceed ambient in a hot configuration")
    unconstrained = f0_ghz / (1.0 - gamma) * (
        (tsafe_c - ambient_c) / rise - gamma
    )
    return max(fmin_ghz, min(f0_ghz, unconstrained)), (
        "frequency_floor" if unconstrained < fmin_ghz else "thermally_limited"
    ), unconstrained


def evaluate(module_model: Path, thermal_result: Path, output: Path,
             f0_ghz: float = 2.0, fmin_ghz: float = 0.4,
             tsafe_c: float = 95.0, ambient_c: float = 25.0,
             ipc2: float | None = None) -> dict:
    model = read_json(module_model)
    thermal = read_json(thermal_result)
    gamma = model["gamma"]
    frequency, state, raw = closed_form_frequency(
        thermal["tmax_c"], gamma, f0_ghz, fmin_ghz, tsafe_c, ambient_c
    )
    floor_power_scale = gamma + (1.0 - gamma) * (fmin_ghz / f0_ghz)
    estimated_tmax_at_fmin = ambient_c + (
        thermal["tmax_c"] - ambient_c
    ) * floor_power_scale
    result = {
        "schema_version": 1, "equation": 13, "gamma": gamma,
        "leakage_power_w": model["totals"]["leakage_power_w"],
        "dynamic_power_w": model["totals"]["dynamic_power_w"],
        "tmax_f0_c": thermal["tmax_c"], "ambient_c": ambient_c,
        "tsafe_c": tsafe_c, "f0_ghz": f0_ghz, "fmin_ghz": fmin_ghz,
        "unclamped_solution_ghz": raw, "sustainable_frequency_ghz": frequency,
        "estimated_tmax_at_fmin_c": estimated_tmax_at_fmin,
        "thermal_feasible_at_fmin": estimated_tmax_at_fmin <= tsafe_c,
        "floor_power_scale": floor_power_scale,
        "state": state, "ipc1": model["ipc1"],
        "bips1_thermal": model["ipc1"] * frequency,
    }
    if ipc2 is not None:
        result["ipc2"] = ipc2
        result["bips2"] = ipc2 * frequency
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--f0-ghz", type=float, default=2.0)
    parser.add_argument("--fmin-ghz", type=float, default=0.4)
    parser.add_argument("--tsafe-c", type=float, default=95.0)
    parser.add_argument("--ambient-c", type=float, default=25.0)
    parser.add_argument("--ipc2", type=float)
    args = parser.parse_args()
    result = evaluate(args.modules, args.thermal, args.output, args.f0_ghz,
                      args.fmin_ghz, args.tsafe_c, args.ambient_c, args.ipc2)
    print(f"f_sus={result['sustainable_frequency_ghz']:.6f} GHz, state={result['state']}")


if __name__ == "__main__":
    main()
