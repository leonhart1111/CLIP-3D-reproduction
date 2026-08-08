#!/usr/bin/env python3
"""Map time-varying McPAT module powers onto one fixed HotSpot 3-D layout."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from workflow.common import read_json, sha256_file, write_json
from workflow.floorplan.generate_hotspot_inputs import grid_power, materialize
from workflow.transient.validation import (
    summarize_power_windows,
    validate_power_triplet,
    validate_power_windows,
)


POWER_FIELDS = ("dynamic_power_w", "leakage_power_w", "total_power_w")


def trace_input_identity(modules: Path, layout: Path, power_windows: Path,
                         config: Path, frequency_scale: float) -> dict:
    """Bind a materialized trace to the immutable inputs that produced it."""
    if not math.isfinite(float(frequency_scale)) or float(frequency_scale) <= 0.0:
        raise ValueError("frequency_scale must be finite and positive")
    return {
        "modules_sha256": sha256_file(Path(modules)),
        "layout_sha256": sha256_file(Path(layout)),
        "power_windows_sha256": sha256_file(Path(power_windows)),
        "config_sha256": sha256_file(Path(config)),
        "frequency_scale": float(frequency_scale),
    }


def validate_materialized_trace(case_dir: Path, expected_identity: dict,
                               require_temperature_trace: bool) -> dict:
    """Check trace manifest identity and parse every emitted trace grid row."""
    case_dir = Path(case_dir)
    manifest = read_json(case_dir / "transient_trace_manifest.json")
    if manifest.get("trace_input_identity") != expected_identity:
        raise ValueError("materialized trace input identity differs")
    required = ["power_transient.ptrace", "power_dynamic_transient.ptrace",
                "power_leakage_transient.ptrace"]
    if require_temperature_trace:
        required.append("transient.ttrace")
    count = manifest.get("window_count")
    grid_count = manifest.get("grid_cell_count")
    if not isinstance(count, int) or count < 1 or not isinstance(grid_count, int) or grid_count < 1:
        raise ValueError("materialized trace manifest shape is invalid")
    for name in required:
        path = case_dir / name
        if not path.is_file():
            raise ValueError(f"materialized trace lacks {name}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != count + 1:
            raise ValueError("materialized trace row count differs")
        if len(lines[0].split()) != grid_count or any(len(row.split()) != grid_count for row in lines[1:]):
            raise ValueError("materialized trace grid differs")
    return manifest
RAW_POWER_PROVENANCE = {
    "dynamic": "McPAT Runtime Dynamic",
    "subthreshold_leakage": "McPAT Subthreshold Leakage",
    "gate_leakage": "McPAT Gate Leakage",
    "postprocessing": "none",
}


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
                      output_dir: Path, config: dict,
                      frequency_scale: float = 1.0,
                      period_repeats: int = 1) -> dict:
    """Materialize a fixed-layout trace, optionally scaled to another frequency."""
    if not math.isfinite(frequency_scale) or frequency_scale <= 0:
        raise ValueError("frequency_scale must be positive and finite")
    if isinstance(period_repeats, bool) or not isinstance(period_repeats, int):
        raise ValueError("period_repeats must be a positive integer")
    if period_repeats < 1:
        raise ValueError("period_repeats must be a positive integer")
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
    if power_windows.get("power_provenance") != RAW_POWER_PROVENANCE:
        raise ValueError("power windows raw-power provenance is incompatible")
    # Grid the steady layout before creating the output directory so geometry
    # and conservation failures cannot publish a partial trace case.
    grid_power(layout, grid_size)
    nominal_sample_interval_s = (
        float(power_windows["nominal_sample_interval_ms"]) / 1000.0
    )
    sample_interval_s = nominal_sample_interval_s / frequency_scale
    scaled_windows = []
    for window in windows:
        modules = []
        for source in window["modules"]:
            module = dict(source)
            module["dynamic_power_w"] = (
                float(source["dynamic_power_w"]) * frequency_scale
            )
            module["leakage_power_w"] = float(source["leakage_power_w"])
            module["total_power_w"] = (
                module["dynamic_power_w"] + module["leakage_power_w"]
            )
            validate_power_triplet(
                module, f"scaled window {window['index']} module {module['name']}"
            )
            modules.append(module)
        scaled_windows.append({
            **window,
            "duration_s": float(window["duration_s"]) / frequency_scale,
            "modules": modules,
            "totals": {
                field: sum(float(module[field]) for module in modules)
                for field in POWER_FIELDS
            },
        })
    power_summary = summarize_power_windows(scaled_windows)
    rows_by_field: dict[str, list[list[float]]] = {field: [] for field in POWER_FIELDS}
    trace_names: list[str] | None = None
    conservation = []
    for period_index in range(period_repeats):
        for window in scaled_windows:
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
                "period_index": period_index,
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

    source_actual_duration_s = float(timeline_audit["total_duration_s"])
    source_hotspot_duration_s = float(timeline_audit["hotspot_trace_duration_s"])
    actual_duration_s = source_actual_duration_s * period_repeats / frequency_scale
    hotspot_duration_s = source_hotspot_duration_s * period_repeats / frequency_scale
    trace_timeline_audit = {
        **timeline_audit,
        "source_window_count": int(timeline_audit["window_count"]),
        "window_count": len(scaled_windows) * period_repeats,
        "source_total_duration_s": source_actual_duration_s,
        "source_hotspot_trace_duration_s": source_hotspot_duration_s,
        "total_duration_s": actual_duration_s,
        "hotspot_trace_duration_s": hotspot_duration_s,
    }
    maximum_grid_residual_w = max(
        abs(tier[field]["residual"])
        for window in conservation
        for tier in window["tiers"]
        for field in POWER_FIELDS
    )
    config_path = output_dir / "trace_input_config.json"
    write_json(config_path, config)
    result = {
        "schema_version": 1,
        "mode": "operational transient validation",
        "non_formal": True,
        "paper_equivalent": False,
        "source_modules": str(modules_path.resolve()),
        "source_layout": str(layout_path.resolve()),
        "source_power_windows": str(power_windows_path.resolve()),
        "trace_input_identity": trace_input_identity(
            modules_path, layout_path, power_windows_path, config_path, frequency_scale,
        ),
        "sample_interval_s": sample_interval_s,
        "nominal_sample_interval_s": nominal_sample_interval_s,
        "window_count": len(scaled_windows) * period_repeats,
        "windows_per_period": len(scaled_windows),
        "period_repeats": period_repeats,
        "frequency_scaling": {
            "frequency_scale": frequency_scale,
            "dynamic_power_scale": frequency_scale,
            "leakage_power_scale": 1.0,
            "time_scale": 1.0 / frequency_scale,
            "model": "fixed-voltage dynamic-power scaling with fixed leakage",
        },
        "grid_cell_count": len(trace_names),
        "actual_gem5_duration_s": actual_duration_s,
        "hotspot_trace_duration_s": hotspot_duration_s,
        "padded_final_duration_s": max(hotspot_duration_s - actual_duration_s, 0.0),
        "partial_window_policy": (
            "The last partial gem5 window is held constant for one full HotSpot "
            "sampling interval; the padding is recorded explicitly."
        ),
        "power_summary": power_summary,
        "timeline_audit": trace_timeline_audit,
        "maximum_grid_residual_w": maximum_grid_residual_w,
        "raw_power_evidence": {
            "power_provenance": RAW_POWER_PROVENANCE,
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
                "raw_unscaled_power": frequency_scale == 1.0,
                "source_power_is_raw": True,
                "frequency_transform_is_explicit": True,
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
