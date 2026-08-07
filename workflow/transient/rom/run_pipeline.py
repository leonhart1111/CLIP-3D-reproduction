#!/usr/bin/env python3
"""Orchestrate the opt-in transient-ROM layout and validation pipeline."""

from __future__ import annotations

import math
from pathlib import Path

from workflow.common import read_json, sha256_file, write_json
from workflow.r2.build_latency_vector import build_vector
from workflow.r2.run_r2 import run as run_r2
from workflow.transient.rom.calibrate_rom import (
    _package_identity,
    validate_calibration_holdouts,
)
from workflow.transient.rom.calibration_design import build_design
from workflow.transient.rom.contracts import parse_settings, require_accepted_package
from workflow.transient.rom.materialize_calibration import execute_calibration_cases
from workflow.transient.rom.optimize_layout import (
    _requested_identity,
    optimize_transient_layout,
)
from workflow.transient.rom.pod_state_space import fit_state_space, save_model
from workflow.transient.run_hotspot_transient import DEFAULT_HOTSPOT
from workflow.transient.run_transient_pipeline import (
    prepare_power_windows,
    reject_overlapping_output,
    validate_matching_r1,
    validate_source_r1,
    validate_steady_output,
)
from workflow.transient.run_transient_r1 import run as run_transient_r1
from workflow.transient.validation import power_trace_identity, validate_power_windows
from workflow.transient.verify_sustainable_frequency import search_layout_frequency


def package_identity(modules_path: Path, power_windows_path: Path,
                     config_path: Path, hotspot: Path, design: dict,
                     config: dict) -> dict:
    """Return the exact identity used by calibration and optimizer reuse."""
    return _package_identity(
        modules_path, power_windows_path, config_path, hotspot, design, config
    )


def _frequency_grid(config: dict) -> list[float]:
    frequency = config.get("frequency")
    if not isinstance(frequency, dict):
        raise ValueError("config lacks frequency settings")
    transient_rom = config.get("transient_rom", {})
    if transient_rom is None:
        transient_rom = {}
    if not isinstance(transient_rom, dict):
        raise ValueError("transient_rom must be a dictionary")
    values = transient_rom.get("frequencies_ghz", frequency.get("grid_ghz"))
    if values is None:
        values = [frequency.get("fmin_ghz"), frequency.get("f0_ghz")]
    if not isinstance(values, list) or not values:
        raise ValueError("transient ROM frequencies_ghz must be a non-empty list")
    try:
        result = sorted({float(value) for value in values})
    except (TypeError, ValueError) as error:
        raise ValueError("transient ROM frequencies_ghz must be numeric") from error
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("transient ROM frequencies_ghz must be finite and positive")
    return result


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(
            _contains_key(nested, key) for nested in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(nested, key) for nested in value)
    return False


_CLASSIFICATION = {
    "thermal_mode": "transient-rom",
    "non_formal": True,
    "paper_equivalent": False,
}


def _write_artifact_manifest(output_dir: Path) -> dict:
    """Bind every ROM-owned output file to the non-formal classification."""
    manifest_path = output_dir / "rom_artifact_manifest.json"
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        artifacts.append({
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(path),
            "classification": dict(_CLASSIFICATION),
        })
    manifest = {
        "schema_version": 1,
        **_CLASSIFICATION,
        "scope": str(output_dir),
        "artifacts": artifacts,
    }
    write_json(manifest_path, manifest)
    return manifest


def _validate_artifact_manifest(package_dir: Path) -> dict:
    manifest_path = package_dir / "rom_artifact_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("reusable ROM package lacks rom_artifact_manifest.json")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or any(
        manifest.get(field) != value for field, value in _CLASSIFICATION.items()
    ):
        raise ValueError("reusable ROM package manifest classification is invalid")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("reusable ROM package manifest artifacts must be a list")
    recorded: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("reusable ROM package manifest record is invalid")
        relative = record.get("path")
        if (not isinstance(relative, str) or not relative
                or relative in recorded or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise ValueError("reusable ROM package manifest path is invalid")
        if record.get("classification") != _CLASSIFICATION:
            raise ValueError("reusable ROM package artifact classification is invalid")
        recorded[relative] = record

    actual: dict[str, Path] = {}
    for path in sorted(package_dir.rglob("*")):
        if path == manifest_path or not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError("reusable ROM package manifest cannot bind symlinks")
        actual[path.relative_to(package_dir).as_posix()] = path
    if set(recorded) != set(actual):
        raise ValueError("reusable ROM package manifest inventory is incomplete or stale")
    for relative, path in actual.items():
        if recorded[relative].get("sha256") != sha256_file(path):
            raise ValueError(f"reusable ROM package manifest hash differs for {relative}")
    return manifest


def _reuse_prepared_power_windows(source_r1_dir: Path,
                                  transient_r1_dir: Path,
                                  output_dir: Path, sample_interval_ms: float) -> dict:
    power_windows_path = (
        output_dir / "windows/mcpat/power_windows.json"
    ).resolve()
    if not power_windows_path.is_file():
        raise FileNotFoundError(power_windows_path)
    power_windows = read_json(power_windows_path)
    audit = validate_power_windows(power_windows)
    expected_paths = {
        "canonical_source_r1": source_r1_dir.resolve(),
        "transient_r1": transient_r1_dir.resolve(),
    }
    for field, expected in expected_paths.items():
        value = power_windows.get(field)
        if not isinstance(value, str) or Path(value).resolve() != expected:
            raise ValueError(f"reusable power windows {field} provenance differs")
    if not math.isclose(
        float(power_windows.get("nominal_sample_interval_ms", float("nan"))),
        sample_interval_ms, rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ValueError("reusable power windows sampling interval differs")
    return {
        "power_windows": str(power_windows_path),
        "transient_r1": str(transient_r1_dir.resolve()),
        "power_trace_identity": power_trace_identity(power_windows),
        "timeline_audit": audit,
        "stage_seconds": {},
        "reused": True,
    }


def _reuse_optimization(modules_path: Path, package_dir: Path,
                        optimization_dir: Path, config_path: Path,
                        power_windows_path: Path, hotspot: Path,
                        config: dict, settings: object) -> dict:
    report_path = optimization_dir / "optimization_report.json"
    proposed_path = optimization_dir / "proposed_layout.json"
    for path in (report_path, proposed_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    modules = read_json(modules_path)
    optimizer = config.get("layout_optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("config lacks layout_optimizer settings")
    design = build_design(modules, optimizer.get("allowed_l2_tiers"), settings)
    requested = _requested_identity(
        modules_path, config_path, hotspot, read_json(power_windows_path),
        config, design,
    )
    acceptance = require_accepted_package(package_dir, requested)
    report = read_json(report_path)
    expected_paths = {
        "modules": modules_path,
        "package_dir": package_dir,
        "power_windows": power_windows_path,
        "config": config_path,
        "hotspot": hotspot,
        "proposed_layout": proposed_path,
    }
    if not isinstance(report, dict) or any(
        report.get(field) != value
        for field, value in _CLASSIFICATION.items()
    ):
        raise ValueError("reusable ROM optimization classification is invalid")
    for field, expected in expected_paths.items():
        value = report.get(field)
        if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
            raise ValueError(f"reusable ROM optimization {field} provenance differs")
    if report.get("hotspot_calls_inside_optimizer") != 0:
        raise ValueError("reusable ROM optimization has invalid HotSpot call count")
    if report.get("requested_identity") != requested:
        raise ValueError("reusable ROM optimization identity differs")
    if report.get("package_acceptance") != acceptance:
        raise ValueError("reusable ROM optimization acceptance differs")
    if report.get("selected_layout") != read_json(proposed_path):
        raise ValueError("reusable ROM proposed layout differs from report")
    if not isinstance(report.get("selected"), dict):
        raise ValueError("reusable ROM optimization lacks selected candidate")
    return report


def _calibrate_package(modules_path: Path, power_windows_path: Path,
                       config_path: Path, package_dir: Path, hotspot: Path,
                       config: dict, settings: object) -> tuple[dict, dict]:
    optimizer = config.get("layout_optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("config lacks layout_optimizer settings")
    design = build_design(
        read_json(modules_path), optimizer.get("allowed_l2_tiers"), settings
    )
    package_dir.mkdir(parents=True, exist_ok=True)
    cases = execute_calibration_cases(
        modules_path, power_windows_path, config_path, design, package_dir,
        settings, hotspot=hotspot,
    )
    model, fit_report = fit_state_space(cases["training_cases"], settings)
    save_model(package_dir / "pod_model.npz", model, fit_report)
    write_json(package_dir / "fit_report.json", fit_report)
    acceptance = validate_calibration_holdouts(
        model, design, cases["holdout_cases"], settings, config,
        output_dir=package_dir,
        identity=package_identity(
            modules_path, power_windows_path, config_path, hotspot, design, config
        ),
    )
    if acceptance.get("accepted") is not True:
        reasons = acceptance.get("failure_reasons", [])
        raise ValueError(
            "transient ROM holdout validation failed: " + ", ".join(reasons)
        )
    _write_artifact_manifest(package_dir)
    return cases, acceptance


def run_transient_rom_pipeline(source_r1_dir: Path, steady_preflight_dir: Path,
                               output_dir: Path, config_path: Path,
                               transient_r1_dir: Path | None,
                               calibrate: bool, execute_r2: bool, *,
                               rom_package_dir: Path | None = None,
                               rerun_r2: bool = False) -> dict:
    """Run one gated ROM optimization and final real-HotSpot validation."""
    source_r1_dir = Path(source_r1_dir).resolve()
    steady_preflight_dir = Path(steady_preflight_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    hotspot = Path(DEFAULT_HOTSPOT).resolve()
    reject_overlapping_output(output_dir, [source_r1_dir, steady_preflight_dir])
    final_validation_dir = output_dir / "final_hotspot_validation"
    if (rerun_r2 and final_validation_dir.exists()
            and (not final_validation_dir.is_dir()
                 or any(final_validation_dir.iterdir()))):
        raise ValueError(
            "refusing to reuse existing final HotSpot validation artifacts: "
            "--rerun-r2 is only for retries that failed before final HotSpot"
        )
    if transient_r1_dir is not None:
        transient_r1_dir = Path(transient_r1_dir).resolve()
        reject_overlapping_output(output_dir, [transient_r1_dir])
    if calibrate and rom_package_dir is not None:
        raise ValueError("cannot combine ROM calibration with a reusable package path")
    if rom_package_dir is not None:
        rom_package_dir = Path(rom_package_dir).resolve()
        default_package = (output_dir / "rom_package").resolve()
        if rom_package_dir != default_package:
            reject_overlapping_output(output_dir, [rom_package_dir])
    for path in (config_path, hotspot):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = read_json(config_path)
    settings = parse_settings(config)
    canonical_r1 = validate_source_r1(source_r1_dir)
    steady_audit = validate_steady_output(
        steady_preflight_dir, "fixed-bin", source_r1_dir, config, config_path
    )
    steady_summary = steady_audit.get("summary")
    if not isinstance(steady_summary, dict):
        raise ValueError("steady preflight validation lacks its summary")
    if any(
        steady_summary.get(field) is not None
        for field in ("ipc2", "bips2", "r2_source")
    ):
        raise ValueError("transient ROM requires an R2-disabled steady preflight")
    modules_path = (steady_preflight_dir / "modules.json").resolve()
    cacti_path = (
        steady_preflight_dir / "cacti/cacti_characterization.json"
    ).resolve()
    for path in (modules_path, cacti_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    steady_artifacts = steady_summary.get("artifacts")
    recorded_cacti = (
        steady_artifacts.get("cacti")
        if isinstance(steady_artifacts, dict) else None
    )
    if (not isinstance(recorded_cacti, str)
            or Path(recorded_cacti).resolve() != cacti_path):
        raise ValueError(
            "steady preflight CACTI artifact path does not match "
            "steady_preflight/cacti/cacti_characterization.json"
        )
    artifact_sha256 = steady_summary.get("artifact_sha256")
    recorded_cacti_sha256 = (
        artifact_sha256.get("cacti")
        if isinstance(artifact_sha256, dict) else None
    )
    if (not isinstance(recorded_cacti_sha256, str)
            or recorded_cacti_sha256 != sha256_file(cacti_path)):
        raise ValueError("steady preflight CACTI artifact sha256 does not match")

    output_dir.mkdir(parents=True, exist_ok=True)
    if transient_r1_dir is None:
        transient_r1_dir = (output_dir / "r1").resolve()
        r1_status = run_transient_r1(
            source_r1_dir, transient_r1_dir, settings.sample_interval_ms
        )
        transient_r1_source = "generated"
    else:
        r1_status = (
            read_json(transient_r1_dir / "status.json")
            if (transient_r1_dir / "status.json").is_file()
            else None
        )
        transient_r1_source = "provided"
    periodic_r1 = validate_source_r1(transient_r1_dir)
    validate_matching_r1(canonical_r1["metadata"], periodic_r1["metadata"])

    cached_power_windows = output_dir / "windows/mcpat/power_windows.json"
    if rerun_r2 and cached_power_windows.is_file():
        prepared = _reuse_prepared_power_windows(
            source_r1_dir, transient_r1_dir, output_dir,
            settings.sample_interval_ms,
        )
        power_windows_reused = True
    else:
        prepared = prepare_power_windows(
            source_r1_dir, transient_r1_dir, output_dir / "windows", config,
            settings.sample_interval_ms,
        )
        power_windows_reused = False
    power_windows_path = Path(prepared["power_windows"]).resolve()
    package_dir = (
        rom_package_dir
        if rom_package_dir is not None
        else (output_dir / "rom_package").resolve()
    )
    calibration_cases = None
    completed_calibration = (
        rerun_r2 and calibrate
        and (package_dir / "rom_artifact_manifest.json").is_file()
        and (package_dir / "rom_acceptance.json").is_file()
        and (package_dir / "validation_report.json").is_file()
    )
    if completed_calibration:
        _validate_artifact_manifest(package_dir)
        calibration_acceptance = read_json(package_dir / "validation_report.json")
        if (not isinstance(calibration_acceptance, dict)
                or calibration_acceptance.get("accepted") is not True):
            raise ValueError("completed ROM calibration validation is not accepted")
        package_status = "reused-calibration"
    elif calibrate:
        calibration_cases, calibration_acceptance = _calibrate_package(
            modules_path, power_windows_path, config_path, package_dir, hotspot,
            config, settings,
        )
        package_status = "calibrated"
    else:
        if not package_dir.is_dir():
            raise FileNotFoundError(
                f"accepted transient ROM package is missing: {package_dir}; "
                "rerun with calibration enabled"
            )
        _validate_artifact_manifest(package_dir)
        calibration_acceptance = None
        package_status = "reused"

    optimization_dir = output_dir / "optimization"
    if (rerun_r2
            and (optimization_dir / "optimization_report.json").is_file()
            and (optimization_dir / "proposed_layout.json").is_file()):
        optimization = _reuse_optimization(
            modules_path, package_dir, optimization_dir, config_path,
            power_windows_path, hotspot, config, settings,
        )
        optimization_reused = True
    else:
        optimization = optimize_transient_layout(
            modules_path, package_dir, optimization_dir, config_path,
            power_windows_path, hotspot=hotspot,
        )
        optimization_reused = False
    proposed_layout = Path(optimization["proposed_layout"]).resolve()
    selected = optimization.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("ROM optimization lacks a selected candidate")
    package_acceptance = optimization.get("package_acceptance")
    if (not isinstance(package_acceptance, dict)
            or package_acceptance.get("accepted") is not True
            or not isinstance(package_acceptance.get("identity"), dict)):
        raise ValueError("ROM optimization lacks complete accepted-package identity")

    delay = config.get("delay")
    if not isinstance(delay, dict):
        raise ValueError("config lacks delay settings")
    latency_path = (output_dir / "r2_latency.json").resolve()
    vector = build_vector(
        modules_path, cacti_path, latency_path, None, None, proposed_layout,
        delay.get("wire_rounding", "nearest"),
        int(delay.get("cycles_per_tsv", 2)),
        int(delay.get("l1_pipeline_cycles", 1)),
        delay.get("wire_aggregation", "mean"),
    )
    r2_result = None
    if execute_r2:
        r2_result = run_r2(
            source_r1_dir, latency_path, output_dir / "gem5_r2",
            rerun=rerun_r2,
        )

    frequency_grid = optimization.get("parameters", {}).get("frequency_grid_ghz")
    if frequency_grid is None:
        frequency_grid = _frequency_grid(config)
    final_validation = search_layout_frequency(
        modules_path, proposed_layout, power_windows_path,
        final_validation_dir, config_path,
        frequencies_ghz=frequency_grid,
        period_repeats=settings.pss_period_repeats,
        pss_tolerance_c=settings.pss_tolerance_c,
        frequency_tolerance_ghz=settings.frequency_tolerance_ghz,
        hotspot=hotspot,
    )
    f_rom = selected.get("f_sus_trans_rom_ghz")
    bips1_rom = selected.get("bips1_trans_rom_pred")
    f_hotspot = final_validation.get("f_sus_trans_ghz")
    cases = calibration_cases or {}
    summary = {
        "schema_version": 1,
        "mode": "transient ROM layout optimization with real-HotSpot validation",
        "thermal_mode": "transient-rom",
        "non_formal": True,
        "paper_equivalent": False,
        "source_r1": str(source_r1_dir),
        "steady_preflight": str(steady_preflight_dir),
        "output": str(output_dir),
        "config": str(config_path),
        "transient_r1": str(transient_r1_dir),
        "transient_r1_source": transient_r1_source,
        "transient_r1_status": r1_status,
        "sample_interval_ms": settings.sample_interval_ms,
        "power_windows_preparations": 1,
        "power_windows_reused": power_windows_reused,
        "power_windows_preparations_this_run": 0 if power_windows_reused else 1,
        "power_trace_identity": prepared.get("power_trace_identity"),
        "rom_package": str(package_dir),
        "rom_package_status": package_status,
        "rom_acceptance": package_acceptance,
        "rom_holdout_validation": calibration_acceptance,
        "training_hotspot_calls": int(cases.get("training_hotspot_calls", 0)),
        "holdout_hotspot_calls": int(cases.get("holdout_hotspot_calls", 0)),
        "calibration_hotspot_calls": (
            int(cases.get("training_hotspot_calls", 0))
            + int(cases.get("holdout_hotspot_calls", 0))
        ),
        "optimizer_hotspot_calls": optimization.get(
            "hotspot_calls_inside_optimizer"
        ),
        "optimization_reused": optimization_reused,
        "final_validation_hotspot_calls": len(
            final_validation.get("search", {}).get("evaluations", [])
        ),
        "f_sus_trans_rom_pred_ghz": f_rom,
        "f_sus_trans_hotspot_ghz": f_hotspot,
        "bips1_trans_rom_pred": bips1_rom,
        "ipc2_trans": r2_result.get("ipc2") if r2_result else None,
        "bips2_trans": (
            float(r2_result["ipc2"]) * float(f_hotspot)
            if r2_result is not None and f_hotspot is not None else None
        ),
        "r2_executed": execute_r2,
        "r2_critical_path_cycles": vector.get("critical_l1d_to_l2_cycles"),
        "artifacts": {
            "modules": str(modules_path),
            "cacti": str(cacti_path),
            "power_windows": str(power_windows_path),
            "rom_package": str(package_dir),
            "rom_package_manifest": (
                str((package_dir / "rom_artifact_manifest.json").resolve())
                if (package_dir / "rom_artifact_manifest.json").is_file()
                else None
            ),
            "optimization_report": str(
                (optimization_dir / "optimization_report.json").resolve()
            ),
            "proposed_layout": str(proposed_layout),
            "r2_latency": str(latency_path),
            "r2_result": (
                str((output_dir / "gem5_r2/r2_result.json").resolve())
                if r2_result else None
            ),
            "final_hotspot_validation": str(
                (final_validation_dir / "transient_sustainable_frequency.json").resolve()
            ),
            "rom_artifact_manifest": str(
                (output_dir / "rom_artifact_manifest.json").resolve()
            ),
        },
        "steady_preflight_layout_method": steady_summary.get("layout_method"),
        "final_hotspot_state": final_validation.get("state"),
    }
    if _contains_key(summary, "bips2"):
        raise AssertionError("transient ROM summary must not contain ambiguous bips2")
    write_json(output_dir / "transient_rom_summary.json", summary)
    _write_artifact_manifest(output_dir)
    return summary
