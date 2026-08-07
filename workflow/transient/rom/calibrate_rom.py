#!/usr/bin/env python3
"""CLI entry point for transient-ROM calibration case materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy

from workflow.common import read_json, write_json
from workflow.transient.rom.contracts import (
    ROMSettings,
    parse_settings,
    rom_input_identity,
    write_rom_artifact_manifest,
)
from workflow.transient.rom.calibration_design import calibration_design_hash
from workflow.transient.rom.layout_rom import evaluate_layout_rom, validate_holdouts
from workflow.transient.rom.materialize_calibration import execute_calibration_cases
from workflow.transient.rom.pod_state_space import (
    StateSpaceModel,
    fit_state_space,
    save_model,
)
from workflow.transient.run_hotspot_transient import (
    parse_ttrace_grid,
    summarize_period_end_convergence,
)
from workflow.transient.verify_sustainable_frequency import last_period_peak


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _artifact(case: dict, name: str, package_dir: Path) -> tuple[Path, bool]:
    artifacts = case.get("artifacts")
    value = artifacts.get(name) if isinstance(artifacts, dict) else None
    hashes = artifacts.get("sha256") if isinstance(artifacts, dict) else None
    recorded = hashes.get(name) if isinstance(hashes, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"holdout {case.get('id')!r} lacks {name} artifact")
    path = Path(value)
    if not path.is_absolute():
        path = Path(package_dir) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, isinstance(recorded, str) and recorded == _sha256(path)


def _same_point(layout: dict, point: dict, l2_name: str) -> bool:
    if not isinstance(layout, dict) or not isinstance(layout.get("modules"), list):
        return False
    l2s = [module for module in layout["modules"]
           if isinstance(module, dict) and module.get("kind") == "l2"]
    if len(l2s) != 1 or not isinstance(point, dict):
        return False
    l2 = l2s[0]
    return (
        l2.get("name") == l2_name
        and l2.get("tier") == point.get("tier")
        and l2.get("x_mm") == point.get("x_mm")
        and l2.get("y_mm") == point.get("y_mm")
    )


def validate_calibration_holdouts(
    model: StateSpaceModel, design: dict, holdout_cases: list[dict],
    settings: ROMSettings, config: dict, *, output_dir: Path,
    identity: dict,
) -> dict:
    """Evaluate and gate the two materialized HotSpot holdouts."""
    if not isinstance(holdout_cases, list) or len(holdout_cases) != 2:
        raise ValueError("holdout_cases must contain exactly two cases")
    if not isinstance(design, dict) or not isinstance(design.get("l2_name"), str):
        raise ValueError("design lacks l2_name")
    frequency_config = config.get("frequency") if isinstance(config, dict) else None
    if not isinstance(frequency_config, dict):
        raise ValueError("config lacks frequency settings")
    f0 = frequency_config.get("f0_ghz")
    tsafe = frequency_config.get("tsafe_c")
    if (isinstance(f0, bool) or not isinstance(f0, (int, float))
            or not math.isfinite(float(f0)) or float(f0) <= 0.0):
        raise ValueError("frequency.f0_ghz must be finite and positive")
    if (isinstance(tsafe, bool) or not isinstance(tsafe, (int, float))
            or not math.isfinite(float(tsafe))):
        raise ValueError("frequency.tsafe_c must be finite")
    comparisons = []
    package_dir = Path(output_dir).resolve()
    for case in holdout_cases:
        if not isinstance(case, dict):
            raise ValueError("holdout case must be a dictionary")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("holdout case lacks an id")
        layout_path, layout_hash_match = _artifact(case, "layout", package_dir)
        power_path, power_hash_match = _artifact(
            case, "power_windows", package_dir,
        )
        trace_path, trace_hash_match = _artifact(
            case, "temperature_trace", package_dir,
        )
        layout = read_json(layout_path)
        power_windows = read_json(power_path)
        frequency = case.get("frequency_ghz")
        if (isinstance(frequency, bool) or not isinstance(frequency, (int, float))
                or not math.isfinite(float(frequency)) or float(frequency) <= 0.0):
            raise ValueError(f"holdout {identifier} has invalid frequency_ghz")
        rom = evaluate_layout_rom(
            model, design, power_windows, layout, float(frequency), settings, config
        )
        names, rows_k = parse_ttrace_grid(trace_path)
        windows = power_windows.get("windows") if isinstance(power_windows, dict) else None
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"holdout {identifier} has invalid power windows")
        windows_per_period = len(windows)
        convergence = summarize_period_end_convergence(rows_k, windows_per_period)
        recorded_pss = case.get("periodic_steady_state")
        recorded_names = (
            recorded_pss.get("grid_unit_names") if isinstance(recorded_pss, dict) else None
        )
        temperature_grid_identity_match = (
            recorded_names == names and tuple(names) == model.grid_unit_names
        )
        hotspot_converged = (
            isinstance(recorded_pss, dict)
            and recorded_pss.get("full_grid_converged") is True
            and recorded_names == names
            and len(names) == model.temperature_basis.shape[0]
            and convergence["period_count"] >= 2
            and convergence["last_delta_max_c"] <= settings.pss_tolerance_c
        )
        peak = last_period_peak(rows_k, windows_per_period)
        final_grid_c = (
            numpy.asarray(rows_k[-windows_per_period:], dtype=float) - 273.15
        )
        hotspot = {
            "converged": hotspot_converged,
            "final_period_grid_c": final_grid_c.tolist(),
            "last_period_peak_c": float(peak["tmax_c"]),
            "last_period_peak": peak,
            "safe": hotspot_converged and float(peak["tmax_c"]) <= float(tsafe),
            "period_end_convergence": convergence,
            "grid_unit_names": names,
        }
        expected_scale = float(frequency) / float(f0)
        recorded_scale = case.get("frequency_scale")
        frequency_match = (
            rom["frequency_ghz"] == float(frequency)
            and isinstance(recorded_scale, (int, float))
            and not isinstance(recorded_scale, bool)
            and math.isclose(
                float(recorded_scale), expected_scale, rel_tol=1e-12, abs_tol=1e-15
            )
        )
        comparisons.append({
            "id": identifier,
            "frequency_ghz": float(frequency),
            "geometry_match": layout_hash_match and _same_point(
                layout, case.get("point"), design["l2_name"]
            ),
            "input_identity_match": power_hash_match,
            "frequency_match": frequency_match,
            "hotspot_trace_identity_match": trace_hash_match,
            "temperature_grid_identity_match": temperature_grid_identity_match,
            "rom": rom,
            "hotspot": hotspot,
        })
    return validate_holdouts(
        comparisons, settings, output_dir=output_dir, identity=identity
    )


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _package_identity(modules_path: Path, power_windows_path: Path,
                      config_path: Path, hotspot: Path, design: dict,
                      config: dict) -> dict:
    power_windows = read_json(power_windows_path)
    canonical_source = power_windows.get("canonical_source_r1")
    if not isinstance(canonical_source, str) or not canonical_source:
        raise ValueError("power windows lack canonical_source_r1 provenance")
    canonical_metadata = Path(canonical_source) / "r1_metadata.json"
    if not canonical_metadata.is_file():
        raise FileNotFoundError(canonical_metadata)
    physical = config.get("physical") if isinstance(config, dict) else None
    frequency = config.get("frequency") if isinstance(config, dict) else None
    if not isinstance(physical, dict) or not isinstance(frequency, dict):
        raise ValueError("config lacks physical/frequency identity settings")
    grid_size = physical.get("grid_size")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("physical.grid_size must be a positive integer")
    from workflow.transient.validation import power_trace_identity

    return rom_input_identity(
        canonical_r1_metadata_hash=_sha256(canonical_metadata),
        power_trace=power_trace_identity(power_windows),
        modules_geometry_hash=_sha256(modules_path),
        layout_geometry_hash=_json_sha256(design.get("base_layout")),
        configuration_hash=_sha256(config_path),
        hotspot_hash=_sha256(hotspot),
        grid={"rows": grid_size, "columns": grid_size},
        stack=physical.get("thermal_stack"),
        cooling={
            "ambient_c": frequency.get("ambient_c"),
            "r_convec_k_per_w": physical.get("r_convec_k_per_w"),
        },
        allowed_l2_tiers=design.get("allowed_l2_tiers"),
        calibration_design_hash=calibration_design_hash(design),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--power-windows", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hotspot", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    design = read_json(args.design)
    identity = _package_identity(
        args.modules, args.power_windows, args.config, args.hotspot, design, config,
    )
    report = execute_calibration_cases(
        args.modules, args.power_windows, args.config, design,
        args.output_dir, parse_settings(config), hotspot=args.hotspot,
        identity=identity,
    )
    model, fit_report = fit_state_space(
        report["training_cases"], parse_settings(config),
        package_dir=args.output_dir,
    )
    fit_report = {
        **fit_report,
        "calibration_design_hash": calibration_design_hash(design),
    }
    save_model(args.output_dir / "pod_model.npz", model, fit_report)
    write_json(args.output_dir / "fit_report.json", fit_report)
    validation = validate_calibration_holdouts(
        model, design, report["holdout_cases"],
        parse_settings(config), config, output_dir=args.output_dir,
        identity=identity,
    )
    if not validation["accepted"]:
        raise ValueError(
            "transient ROM holdout validation failed: "
            + ", ".join(validation["failure_reasons"])
        )
    write_rom_artifact_manifest(args.output_dir)
    print(
        "Transient ROM cases: "
        f"{report['training_hotspot_calls']} training, "
        f"{report['holdout_hotspot_calls']} holdout; "
        f"POD rank {fit_report['pod']['rank']}"
    )


if __name__ == "__main__":
    main()
