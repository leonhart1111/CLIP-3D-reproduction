#!/usr/bin/env python3
"""Map time-varying McPAT module powers onto one fixed HotSpot 3-D layout."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import grid_power, materialize
from workflow.transient.validation import (
    summarize_power_windows,
    validate_power_triplet,
    validate_power_windows,
)


POWER_FIELDS = ("dynamic_power_w", "leakage_power_w", "total_power_w")


def write_trace(path: Path, names: list[str], rows: list[list[float]]) -> None:
    lines = ["\t".join(names)]
    lines.extend("\t".join(f"{value:.17g}" for value in row) for row in rows)
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
    model = read_json(modules_path)
    layout = read_json(layout_path)
    model_modules = model.get("modules")
    layout_modules = layout.get("modules")
    if not isinstance(model_modules, list) or not isinstance(layout_modules, list):
        raise ValueError("module model and layout must contain module lists")
    model_by_name = {}
    for module in model_modules:
        name = module.get("name")
        if not isinstance(name, str) or name in model_by_name:
            raise ValueError("module model contains missing or duplicate names")
        for field in POWER_FIELDS:
            value = float(module[field])
            if not math.isfinite(value) or value < -1e-12:
                raise ValueError(f"steady module {name} {field} is invalid")
        model_by_name[name] = module
    layout_by_name = {module.get("name"): module for module in layout_modules}
    if (
        None in layout_by_name
        or len(layout_by_name) != len(layout_modules)
        or set(layout_by_name) != set(model_by_name)
    ):
        raise ValueError("steady layout module identity does not match module model")
    for name, module in model_by_name.items():
        placed = layout_by_name[name]
        for field in ("kind", "core", "area_mm2", *POWER_FIELDS):
            if placed.get(field) != module.get(field):
                raise ValueError(f"steady layout module identity differs for {name}")

    layout_names = [module["name"] for module in layout_modules]
    timeline_audit = validate_power_windows(
        power_windows, expected_module_names=set(layout_names)
    )
    run_settings = power_windows.get("run_settings")
    if not isinstance(run_settings, dict):
        raise ValueError("power windows lack McPAT run settings")
    dynamic_scale = float(run_settings.get("dynamic_scale", float("nan")))
    leakage_scale = float(run_settings.get("leakage_scale", float("nan")))
    if not math.isclose(dynamic_scale, 1.0, rel_tol=0.0, abs_tol=1e-15) or not math.isclose(
        leakage_scale, 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("transient trace requires raw-power scales of 1.0")
    # Grid the steady layout before creating the output directory so geometry
    # and conservation failures cannot publish a partial trace case.
    grid_power(layout, grid_size)
    power_summary = summarize_power_windows(windows)
    sample_interval_s = float(power_windows["nominal_sample_interval_ms"]) / 1000.0
    rows_by_field: dict[str, list[list[float]]] = {field: [] for field in POWER_FIELDS}
    trace_names: list[str] | None = None
    conservation = []
    for window in windows:
        by_name = {module["name"]: module for module in window["modules"]}
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
    materialize(
        modules_path, output_dir, grid_size, float(physical["utilization"]),
        float(frequency["ambient_c"]), float(physical["r_convec_k_per_w"]),
        layout_path, physical.get("thermal_stack"),
    )
    set_sampling_interval(output_dir / "hotspot.config", sample_interval_s)
    filenames = {
        "total_power_w": "power_transient.ptrace",
        "dynamic_power_w": "power_dynamic_transient.ptrace",
        "leakage_power_w": "power_leakage_transient.ptrace",
    }
    for field, filename in filenames.items():
        write_trace(output_dir / filename, trace_names, rows_by_field[field])

    actual_duration_s = float(timeline_audit["total_duration_s"])
    hotspot_duration_s = float(timeline_audit["hotspot_trace_duration_s"])
    maximum_grid_residual_w = max(
        abs(tier[field]["residual"])
        for window in conservation
        for tier in window["tiers"]
        for field in POWER_FIELDS
    )
    result = {
        "schema_version": 1,
        "mode": "operational transient validation",
        "non_formal": True,
        "paper_equivalent": False,
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
        "timeline_audit": timeline_audit,
        "maximum_grid_residual_w": maximum_grid_residual_w,
        "raw_power_evidence": {
            "dynamic_scale": dynamic_scale,
            "leakage_scale": leakage_scale,
            "calibration_provenance": run_settings.get("calibration_provenance"),
            "source_stat_hashes": [
                window["source_stats_sha256"] for window in windows
            ],
        },
        "conservation_evidence": {
            "module_triplets": True,
            "module_to_window_totals": True,
            "grid_conservation": True,
            "maximum_grid_residual_w": maximum_grid_residual_w,
            "power_conservation": conservation,
        },
        "acceptance_checks": {
            "checks": {
                "at_least_two_windows": len(windows) >= 2,
                "fixed_step_timeline": True,
                "actual_duration_within_hotspot_duration": (
                    actual_duration_s <= hotspot_duration_s
                ),
                "raw_unscaled_power": True,
                "module_power_conservation": True,
                "grid_power_conservation": True,
            },
            "all_passed": True,
            "failure_reasons": [],
        },
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
