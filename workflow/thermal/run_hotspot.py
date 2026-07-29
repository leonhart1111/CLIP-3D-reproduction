#!/usr/bin/env python3
"""Run one generated HotSpot detailed-3D steady-state case and parse Tmax."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, write_json


DEFAULT_HOTSPOT = PROJECT_ROOT / "tools/src/hotspot/hotspot"


def temperatures(path: Path) -> list[tuple[str, float]]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            value = float(fields[-1])
        except ValueError:
            continue
        values.append((fields[0], value))
    if not values:
        raise ValueError(f"no temperatures in {path}")
    return values


def run_hotspot(case_dir: Path, hotspot: Path = DEFAULT_HOTSPOT,
                ptrace_name: str = "power.ptrace", result_name: str = "thermal_result.json",
                steady_name: str | None = None,
                grid_steady_name: str | None = None) -> dict:
    case_dir = case_dir.resolve()
    result_stem = Path(result_name).stem
    steady = case_dir / (steady_name or (
        "steady.txt" if result_name == "thermal_result.json" else f"{result_stem}.steady.txt"
    ))
    grid_steady = case_dir / (grid_steady_name or (
        "grid.steady.txt" if result_name == "thermal_result.json"
        else f"{result_stem}.grid.steady.txt"
    ))
    command = [
        str(hotspot.resolve()), "-c", "hotspot.config", "-p", ptrace_name,
        "-grid_layer_file", "stack.lcf", "-materials_file", "materials.txt",
        "-model_type", "grid", "-detailed_3D", "on",
        "-steady_file", steady.name, "-grid_steady_file", grid_steady.name,
    ]
    process = subprocess.run(command, cwd=case_dir, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (case_dir / "hotspot.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0 or not steady.is_file():
        raise RuntimeError(f"HotSpot failed (rc={process.returncode}); see {case_dir / 'hotspot.log'}")
    samples = temperatures(steady)
    peak_name, peak_k = max(samples, key=lambda item: item[1])
    manifest = read_json(case_dir / "hotspot_manifest.json")
    result = {
        "schema_version": 1, "command": command, "return_code": process.returncode,
        "power_trace": str((case_dir / ptrace_name).resolve()),
        "steady_file": str(steady), "grid_steady_file": str(grid_steady),
        "tmax_k": peak_k, "tmax_c": peak_k - 273.15, "peak_unit": peak_name,
        "ambient_c": manifest["ambient_c"], "r_convec_k_per_w": manifest["r_convec_k_per_w"],
        "sample_count": len(samples),
    }
    write_json(case_dir / result_name, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--hotspot", type=Path, default=DEFAULT_HOTSPOT)
    parser.add_argument("--ptrace", default="power.ptrace")
    parser.add_argument("--result-name", default="thermal_result.json")
    args = parser.parse_args()
    result = run_hotspot(args.case_dir, args.hotspot, args.ptrace, args.result_name)
    print(f"HotSpot Tmax = {result['tmax_c']:.3f} C at {result['peak_unit']}")


if __name__ == "__main__":
    main()
