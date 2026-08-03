#!/usr/bin/env python3
"""Map time-varying McPAT module powers onto one fixed HotSpot 3-D layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import grid_power, materialize
from workflow.transient.validation import summarize_power_windows, validate_power_triplet


POWER_FIELDS = ("dynamic_power_w", "leakage_power_w", "total_power_w")


def write_trace(path: Path, names: list[str], rows: list[list[float]]) -> None:
    lines = ["\t".join(names)]
    lines.extend("\t".join(f"{value:.12g}" for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_sampling_interval(config_path: Path, interval_s: float) -> None:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith("-sampling_intvl "):
            lines[index] = f"-sampling_intvl {interval_s:.12g}"
            replaced = True
            break
    if not replaced:
        raise ValueError(f"HotSpot config lacks -sampling_intvl: {config_path}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_trace(modules_path: Path, layout_path: Path, power_windows_path: Path,
                      output_dir: Path, config: dict) -> dict:
    physical = config["physical"]
    frequency = config["frequency"]
    grid_size = int(physical["grid_size"])
    power_windows = read_json(power_windows_path)
    windows = power_windows["windows"]
    power_summary = summarize_power_windows(windows)
    sample_interval_s = float(power_windows["nominal_sample_interval_ms"]) / 1000.0
    materialize(
        modules_path, output_dir, grid_size, float(physical["utilization"]),
        float(frequency["ambient_c"]), float(physical["r_convec_k_per_w"]),
        layout_path, physical.get("thermal_stack"),
    )
    set_sampling_interval(output_dir / "hotspot.config", sample_interval_s)

    layout = read_json(output_dir / "layout.json")
    layout_names = [module["name"] for module in layout["modules"]]
    rows_by_field: dict[str, list[list[float]]] = {field: [] for field in POWER_FIELDS}
    trace_names: list[str] | None = None
    conservation = []
    for window in power_windows["windows"]:
        by_name = {module["name"]: module for module in window["modules"]}
        if set(by_name) != set(layout_names):
            missing = sorted(set(layout_names) - set(by_name))
            extra = sorted(set(by_name) - set(layout_names))
            raise ValueError(
                f"window {window['index']} module mismatch: missing={missing}, extra={extra}"
            )
        window_layout = {**layout, "modules": []}
        for source in layout["modules"]:
            module = dict(source)
            powers = by_name[module["name"]]
            validate_power_triplet(
                powers, f"window {window['index']} module {module['name']}"
            )
            for field in POWER_FIELDS:
                module[field] = float(powers[field])
            window_layout["modules"].append(module)
        gridded = grid_power(window_layout, grid_size)
        cells = [cell for tier in gridded["tiers"] for cell in tier["cells"]]
        names = [cell["name"] for cell in cells]
        trace_names = trace_names or names
        if names != trace_names:
            raise ValueError("HotSpot grid cell order changed between windows")
        for field in POWER_FIELDS:
            rows_by_field[field].append([float(cell[field]) for cell in cells])
        conservation.append({
            "window_index": window["index"],
            "tiers": gridded["power_conservation"],
        })

    if not trace_names or not rows_by_field["total_power_w"]:
        raise ValueError("no transient power rows were generated")
    filenames = {
        "total_power_w": "power_transient.ptrace",
        "dynamic_power_w": "power_dynamic_transient.ptrace",
        "leakage_power_w": "power_leakage_transient.ptrace",
    }
    for field, filename in filenames.items():
        write_trace(output_dir / filename, trace_names, rows_by_field[field])

    actual_duration_s = sum(float(window["duration_s"]) for window in windows)
    hotspot_duration_s = len(windows) * sample_interval_s
    maximum_grid_residual_w = max(
        abs(tier[field]["residual"])
        for window in conservation
        for tier in window["tiers"]
        for field in POWER_FIELDS
    )
    result = {
        "schema_version": 1,
        "source_modules": str(modules_path.resolve()),
        "source_layout": str(layout_path.resolve()),
        "source_power_windows": str(power_windows_path.resolve()),
        "sample_interval_s": sample_interval_s,
        "window_count": len(windows),
        "grid_cell_count": len(trace_names),
        "actual_gem5_duration_s": actual_duration_s,
        "hotspot_trace_duration_s": hotspot_duration_s,
        "padded_final_duration_s": max(hotspot_duration_s - actual_duration_s, 0.0),
        "partial_window_policy": (
            "The last partial gem5 window is held constant for one full HotSpot "
            "sampling interval; the padding is recorded explicitly."
        ),
        "power_summary": power_summary,
        "maximum_grid_residual_w": maximum_grid_residual_w,
        "files": {
            field: str((output_dir / filename).resolve())
            for field, filename in filenames.items()
        },
        "power_conservation": conservation,
    }
    write_json(output_dir / "transient_trace_manifest.json", result)
    hotspot_manifest_path = output_dir / "hotspot_manifest.json"
    hotspot_manifest = read_json(hotspot_manifest_path)
    hotspot_manifest["transient_trace"] = result
    write_json(hotspot_manifest_path, hotspot_manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--power-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_trace(
        args.modules.resolve(), args.layout.resolve(), args.power_windows.resolve(),
        args.output_dir.resolve(), read_json(args.config),
    )
    print(
        f"HotSpot transient trace: {result['window_count']} rows, "
        f"dt={result['sample_interval_s'] * 1000:.6g} ms"
    )


if __name__ == "__main__":
    main()
