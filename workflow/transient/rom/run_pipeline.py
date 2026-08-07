#!/usr/bin/env python3
"""Orchestrate the opt-in transient-ROM layout and validation pipeline."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

from workflow.common import read_json, sha256_file, write_json
from workflow.r2.build_latency_vector import build_vector
from workflow.r2.run_r2 import run as run_r2
from workflow.transient.rom.calibrate_rom import (
    _package_identity,
    validate_calibration_holdouts,
)
from workflow.transient.rom.calibration_design import (
    build_design,
    calibration_design_hash,
    load_calibration_design,
)
from workflow.transient.rom.contracts import (
    ROMSettings,
    parse_settings,
    require_accepted_package,
    require_package_calibration_evidence,
    require_rom_artifact_manifest,
    write_rom_artifact_manifest,
)
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


def _validate_final_search_result(
    value: object, frequency_grid: list[float], settings: ROMSettings,
    config: dict,
) -> dict:
    """Require one internally consistent real-HotSpot search result."""
    if not isinstance(value, dict):
        raise ValueError("final HotSpot validation result must be a dictionary")
    search = value.get("search")
    if not isinstance(search, dict):
        raise ValueError("final HotSpot validation lacks a search report")
    evaluations = search.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("final HotSpot validation evaluations must be non-empty")
    frequency = config.get("frequency") if isinstance(config, dict) else None
    if not isinstance(frequency, dict):
        raise ValueError("config lacks frequency settings")
    tsafe_c = frequency.get("tsafe_c")
    f0_ghz = frequency.get("f0_ghz")
    if (isinstance(tsafe_c, bool) or not isinstance(tsafe_c, Real)
            or not math.isfinite(float(tsafe_c))):
        raise ValueError("frequency.tsafe_c must be finite")
    if (isinstance(f0_ghz, bool) or not isinstance(f0_ghz, Real)
            or not math.isfinite(float(f0_ghz)) or float(f0_ghz) <= 0.0):
        raise ValueError("frequency.f0_ghz must be finite and positive")
    grid = sorted({float(item) for item in frequency_grid})
    if (not grid or any(not math.isfinite(item) or item <= 0.0 for item in grid)):
        raise ValueError("final HotSpot frequency grid is invalid")

    by_frequency: dict[float, dict] = {}
    ordered_frequencies = []
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, dict):
            raise ValueError(
                f"final HotSpot evaluation {index} must be a dictionary"
            )
        item_frequency = evaluation.get("frequency_ghz")
        if (isinstance(item_frequency, bool) or not isinstance(item_frequency, Real)
                or not math.isfinite(float(item_frequency))
                or not grid[0] <= float(item_frequency) <= grid[-1]):
            raise ValueError(
                f"final HotSpot evaluation {index} has invalid frequency"
            )
        item_frequency = float(item_frequency)
        if item_frequency in by_frequency:
            raise ValueError("final HotSpot evaluations contain duplicate frequencies")
        converged = evaluation.get("converged")
        peak_c = evaluation.get("last_period_peak_c")
        safe = evaluation.get("safe")
        if not isinstance(converged, bool):
            raise ValueError(
                f"final HotSpot evaluation {index} lacks boolean convergence"
            )
        if (isinstance(peak_c, bool) or not isinstance(peak_c, Real)
                or not math.isfinite(float(peak_c))):
            raise ValueError(
                f"final HotSpot evaluation {index} lacks a finite period peak"
            )
        expected_safe = converged and float(peak_c) <= float(tsafe_c)
        if not isinstance(safe, bool) or safe is not expected_safe:
            raise ValueError(
                f"final HotSpot evaluation {index} safety is inconsistent"
            )
        by_frequency[item_frequency] = evaluation
        ordered_frequencies.append(item_frequency)
    if ordered_frequencies != sorted(ordered_frequencies):
        raise ValueError("final HotSpot evaluations are not frequency ordered")
    if any(item not in by_frequency for item in grid):
        raise ValueError("final HotSpot evaluations do not cover the frequency grid")
    grid_evaluations = [by_frequency[item] for item in grid]

    safe_frequencies = [
        item for item, evaluation in by_frequency.items()
        if evaluation["safe"]
    ]
    expected_sustainable = max(safe_frequencies) if safe_frequencies else None
    reported_sustainable = value.get("f_sus_trans_ghz")
    search_sustainable = search.get("sustainable_frequency_ghz")
    if expected_sustainable is None:
        if reported_sustainable is not None or search_sustainable is not None:
            raise ValueError(
                "final HotSpot sustainable frequency is inconsistent with safety"
            )
    else:
        for candidate in (reported_sustainable, search_sustainable):
            if (isinstance(candidate, bool) or not isinstance(candidate, Real)
                    or not math.isfinite(float(candidate))
                    or float(candidate) != expected_sustainable):
                raise ValueError(
                    "final HotSpot sustainable frequency is missing or inconsistent"
                )

    floor_infeasible = not grid_evaluations[0]["safe"]
    expected_state = (
        "thermally_infeasible" if expected_sustainable is None
        else "thermally_infeasible_at_fmin" if floor_infeasible
        else "thermal_headroom_within_grid" if grid_evaluations[-1]["safe"]
        else "thermally_limited"
    )
    if value.get("state") != expected_state or search.get("state") != expected_state:
        raise ValueError("final HotSpot state is inconsistent with evaluations")
    monotonic = not any(
        not left["safe"] and right["safe"]
        for left, right in zip(grid_evaluations, grid_evaluations[1:])
    )
    if (search.get("frequency_grid_ghz") != grid
            or search.get("grid_evaluations") != grid_evaluations
            or search.get("tsafe_c") != float(tsafe_c)
            or search.get("frequency_tolerance_ghz")
            != settings.frequency_tolerance_ghz
            or search.get("thermally_infeasible_at_fmin") is not floor_infeasible
            or search.get("monotonic_grid_safe_to_unsafe") is not monotonic
            or not isinstance(search.get("safe_unsafe_brackets"), list)):
        raise ValueError("final HotSpot search evidence is inconsistent")
    if value.get("frequency") != {
        "f0_ghz": float(f0_ghz), "tsafe_c": float(tsafe_c), "grid_ghz": grid,
    }:
        raise ValueError("final HotSpot frequency evidence is inconsistent")
    if value.get("periodic_steady_state") != {
        "period_repeats": settings.pss_period_repeats,
        "pss_tolerance_c": settings.pss_tolerance_c,
    }:
        raise ValueError("final HotSpot PSS settings are inconsistent")
    return value


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
    return write_rom_artifact_manifest(output_dir)


def _validate_artifact_manifest(package_dir: Path) -> dict:
    return require_rom_artifact_manifest(package_dir)


def _load_calibration_evidence(package_dir: Path, settings: ROMSettings) -> dict:
    """Load and cross-check the immutable 8+2 evidence behind a reused ROM."""
    package_dir = Path(package_dir).resolve()
    acceptance_path = package_dir / "rom_acceptance.json"
    if not acceptance_path.is_file():
        raise ValueError("reusable ROM package lacks rom_acceptance.json")
    acceptance = read_json(acceptance_path)
    identity = acceptance.get("identity") if isinstance(acceptance, dict) else None
    if not isinstance(identity, dict):
        raise ValueError("reusable ROM acceptance evidence is incomplete")
    return require_package_calibration_evidence(package_dir, identity, settings)


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
    design = load_calibration_design(package_dir)
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
    identity = package_identity(
        modules_path, power_windows_path, config_path, hotspot, design, config
    )
    cases = execute_calibration_cases(
        modules_path, power_windows_path, config_path, design, package_dir,
        settings, hotspot=hotspot, identity=identity,
    )
    model, fit_report = fit_state_space(
        cases["training_cases"], settings, package_dir=package_dir,
    )
    fit_report = {
        **fit_report,
        "calibration_design_hash": calibration_design_hash(design),
    }
    save_model(package_dir / "pod_model.npz", model, fit_report)
    write_json(package_dir / "fit_report.json", fit_report)
    acceptance = validate_calibration_holdouts(
        model, design, cases["holdout_cases"], settings, config,
        output_dir=package_dir,
        identity=identity,
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
    if (final_validation_dir.exists()
            and (not final_validation_dir.is_dir()
                 or any(final_validation_dir.iterdir()))):
        if rerun_r2:
            raise ValueError(
                "refusing to reuse existing final HotSpot validation artifacts: "
                "--rerun-r2 is only for retries that failed before final HotSpot"
            )
        raise FileExistsError(
            "final HotSpot validation output directory is not empty: "
            f"{final_validation_dir}"
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
    calibration_evidence = None
    completed_calibration = (
        rerun_r2 and calibrate
        and (package_dir / "rom_artifact_manifest.json").is_file()
        and (package_dir / "rom_acceptance.json").is_file()
        and (package_dir / "validation_report.json").is_file()
    )
    if completed_calibration:
        _validate_artifact_manifest(package_dir)
        calibration_evidence = _load_calibration_evidence(package_dir, settings)
        calibration_cases = calibration_evidence["cases"]
        calibration_acceptance = calibration_evidence["validation"]
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
        calibration_evidence = _load_calibration_evidence(package_dir, settings)
        calibration_cases = calibration_evidence["cases"]
        calibration_acceptance = calibration_evidence["validation"]
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
    if (calibration_evidence is not None
            and package_acceptance != calibration_evidence["acceptance"]):
        raise ValueError("ROM optimization acceptance differs from saved calibration evidence")

    delay = config.get("delay")
    if not isinstance(delay, dict):
        raise ValueError("config lacks delay settings")
    frequency_grid = optimization.get("parameters", {}).get("frequency_grid_ghz")
    if frequency_grid is None:
        frequency_grid = _frequency_grid(config)
    for path in (
        modules_path, proposed_layout, power_windows_path, config_path, hotspot,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    final_validation_failure = None
    try:
        final_validation = search_layout_frequency(
            modules_path, proposed_layout, power_windows_path,
            final_validation_dir, config_path,
            frequencies_ghz=frequency_grid,
            period_repeats=settings.pss_period_repeats,
            pss_tolerance_c=settings.pss_tolerance_c,
            frequency_tolerance_ghz=settings.frequency_tolerance_ghz,
            hotspot=hotspot,
        )
        final_validation = _validate_final_search_result(
            final_validation, frequency_grid, settings, config,
        )
    except (OSError, RuntimeError, ValueError) as error:
        final_validation_dir.mkdir(parents=True, exist_ok=True)
        final_validation_failure = {
            "category": (
                "validation_contract_error"
                if isinstance(error, ValueError) else "tool_error"
            ),
            "error_type": type(error).__name__,
            "message": str(error),
            "artifact_dir": str(final_validation_dir),
        }
        write_json(
            final_validation_dir / "final_validation_failure.json",
            final_validation_failure,
        )
        final_validation = {
            "schema_version": 1,
            "state": "rom_final_validation_failed",
            "f_sus_trans_ghz": None,
            "search": {"evaluations": []},
            "failure": final_validation_failure,
        }

    search = final_validation.get("search")
    evaluations = search.get("evaluations") if isinstance(search, dict) else None
    if not isinstance(evaluations, list):
        evaluations = []
    materialized_final_cases = (
        sum(
            1 for path in final_validation_dir.iterdir()
            if path.is_dir() and path.name.startswith("frequency_")
        )
        if final_validation_dir.is_dir() else 0
    )
    final_validation_hotspot_calls = max(len(evaluations), materialized_final_cases)
    f_hotspot = final_validation.get("f_sus_trans_ghz")
    if final_validation_failure is None:
        nonconverged = [
            evaluation for evaluation in evaluations
            if not isinstance(evaluation, dict)
            or evaluation.get("converged") is not True
        ]
        if not evaluations:
            final_validation_failure = {
                "category": "missing_pss_evidence",
                "message": "final HotSpot validation recorded no PSS evaluations",
                "artifact_dir": str(final_validation_dir),
                "evaluations": [],
            }
        elif nonconverged:
            final_validation_failure = {
                "category": "pss_nonconvergence",
                "message": "one or more final HotSpot evaluations did not reach PSS",
                "artifact_dir": str(final_validation_dir),
                "evaluations": nonconverged,
            }
        if final_validation_failure is not None:
            f_hotspot = None
            write_json(
                final_validation_dir / "final_validation_failure.json",
                final_validation_failure,
            )

    if final_validation_failure is not None:
        pipeline_state = "rom_final_validation_failed"
        final_validation_classification = final_validation_failure["category"]
    elif f_hotspot is None:
        pipeline_state = "thermally_infeasible"
        final_validation_classification = "true_thermal_infeasible"
    else:
        pipeline_state = str(final_validation.get("state") or "validated")
        final_validation_classification = "validated"

    latency_path = (output_dir / "r2_latency.json").resolve()
    vector = None
    r2_result = None
    if final_validation_classification == "validated":
        vector = build_vector(
            modules_path, cacti_path, latency_path, None, None, proposed_layout,
            delay.get("wire_rounding", "nearest"),
            int(delay.get("cycles_per_tsv", 2)),
            int(delay.get("l1_pipeline_cycles", 1)),
            delay.get("wire_aggregation", "mean"),
        )
        if execute_r2:
            r2_result = run_r2(
                source_r1_dir, latency_path, output_dir / "gem5_r2",
                rerun=rerun_r2,
            )

    f_rom = selected.get("f_sus_trans_rom_ghz")
    bips1_rom = selected.get("bips1_trans_rom_pred")
    cases = calibration_cases or {}
    historical_training_calls = int(cases.get("training_hotspot_calls", 0))
    historical_holdout_calls = int(cases.get("holdout_hotspot_calls", 0))
    invocation_training_calls = historical_training_calls if package_status == "calibrated" else 0
    invocation_holdout_calls = historical_holdout_calls if package_status == "calibrated" else 0
    summary = {
        "schema_version": 1,
        "mode": "transient ROM layout optimization with real-HotSpot validation",
        "thermal_mode": "transient-rom",
        "non_formal": True,
        "paper_equivalent": False,
        "state": pipeline_state,
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
        "training_hotspot_calls": historical_training_calls,
        "holdout_hotspot_calls": historical_holdout_calls,
        "calibration_hotspot_calls": historical_training_calls + historical_holdout_calls,
        "training_hotspot_calls_this_invocation": invocation_training_calls,
        "holdout_hotspot_calls_this_invocation": invocation_holdout_calls,
        "calibration_hotspot_calls_this_invocation": (
            invocation_training_calls + invocation_holdout_calls
        ),
        "optimizer_hotspot_calls": optimization.get(
            "hotspot_calls_inside_optimizer"
        ),
        "optimization_reused": optimization_reused,
        "final_validation_hotspot_calls": final_validation_hotspot_calls,
        "final_validation_classification": final_validation_classification,
        "final_validation_failure": final_validation_failure,
        "f_sus_trans_rom_pred_ghz": f_rom,
        "f_sus_trans_hotspot_ghz": f_hotspot,
        "bips1_trans_rom_pred": bips1_rom,
        "ipc2_trans": r2_result.get("ipc2") if r2_result else None,
        "bips2_trans": (
            float(r2_result["ipc2"]) * float(f_hotspot)
            if r2_result is not None and f_hotspot is not None else None
        ),
        "r2_requested": execute_r2,
        "r2_executed": r2_result is not None,
        "r2_critical_path_cycles": (
            vector.get("critical_l1d_to_l2_cycles") if vector else None
        ),
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
            "calibration_manifest": str(
                (package_dir / "calibration_manifest.json").resolve()
            ),
            "anchors": str((package_dir / "anchors.json").resolve()),
            "calibration_cases": str(
                (package_dir / "calibration_cases.json").resolve()
            ),
            "holdout_validation": str(
                (package_dir / "validation_report.json").resolve()
            ),
            "optimization_report": str(
                (optimization_dir / "optimization_report.json").resolve()
            ),
            "proposed_layout": str(proposed_layout),
            "r2_latency": str(latency_path) if vector else None,
            "r2_result": (
                str((output_dir / "gem5_r2/r2_result.json").resolve())
                if r2_result else None
            ),
            "final_hotspot_validation": (
                str((final_validation_dir / "transient_sustainable_frequency.json").resolve())
                if (final_validation_dir / "transient_sustainable_frequency.json").is_file()
                else None
            ),
            "final_validation_failure": (
                str((final_validation_dir / "final_validation_failure.json").resolve())
                if (final_validation_dir / "final_validation_failure.json").is_file()
                else None
            ),
            "rom_artifact_manifest": str(
                (output_dir / "rom_artifact_manifest.json").resolve()
            ),
        },
        "steady_preflight_layout_method": steady_summary.get("layout_method"),
        "final_hotspot_state": (
            "rom_final_validation_failed"
            if final_validation_failure is not None
            else final_validation.get("state")
        ),
    }
    if _contains_key(summary, "bips2"):
        raise AssertionError("transient ROM summary must not contain ambiguous bips2")
    write_json(output_dir / "transient_rom_summary.json", summary)
    _write_artifact_manifest(output_dir)
    return summary
