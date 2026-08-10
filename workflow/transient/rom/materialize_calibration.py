"""Materialize the fixed transient-ROM calibration and holdout case set.

This module deliberately consumes existing raw McPAT power windows.  PRBS is
an explicit calibration excitation: it scales each raw module's dynamic and
leakage components together, recomputes totals, and records the raw source
identity instead of treating the synthetic trace as a new measurement.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import shutil

from workflow.common import read_json, write_json
from workflow.transient.generate_hotspot_trace import materialize_trace
from workflow.transient.rom.calibration_design import (
    calibration_design_hash,
    layout_for_point,
    require_design_matches_modules,
)
from workflow.transient.rom.contracts import ROMSettings
from workflow.transient.rom.evidence import (
    ROM_CLASSIFICATION,
    require_new_output_root,
    sha256_identity,
)
from workflow.transient.run_hotspot_transient import (
    DEFAULT_HOTSPOT,
    parse_ttrace_grid,
    run_hotspot_transient,
    summarize_period_end_convergence,
)
from workflow.transient.run_hotspot_steady import run_hotspot_steady
from workflow.transient.validation import (
    power_trace_identity,
    validate_power_triplet,
    validate_power_windows,
)


POWER_FIELDS = ("dynamic_power_w", "leakage_power_w", "total_power_w")


def _sha256(path: Path) -> str:
    return sha256_identity(path)


def _positive_multiplier(value: object, name: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"PRBS multiplier for {name} at window {index} is invalid")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"PRBS multiplier for {name} at window {index} must be positive")
    return result


def build_prbs_power_windows(source_windows: dict,
                             multipliers: dict[str, list[float]],
                             window_count: int) -> dict:
    """Return a schema-valid PRBS excitation derived from raw power windows.

    Each output window selects one source window cyclically so a short raw
    workload trace can seed the fixed 64-window calibration period.  The
    timeline is made whole-window and contiguous, while every source statistics
    hash and the top-level raw-power provenance remain recorded.
    """
    if isinstance(window_count, bool) or not isinstance(window_count, int) or window_count < 1:
        raise ValueError("window_count must be a positive integer")
    source_audit = validate_power_windows(source_windows)
    source_records = source_windows["windows"]
    source_names = source_audit["module_names"]
    if not isinstance(multipliers, dict) or set(multipliers) != set(source_names):
        raise ValueError("PRBS multipliers must exactly match source module names")
    for name in source_names:
        values = multipliers[name]
        if not isinstance(values, list) or len(values) != window_count:
            raise ValueError(f"PRBS multipliers for {name} must contain {window_count} values")
        for index, value in enumerate(values):
            _positive_multiplier(value, name, index)

    nominal_ticks = int(source_windows["nominal_sample_interval_ticks"])
    nominal_ms = float(source_windows["nominal_sample_interval_ms"])
    nominal_s = nominal_ms / 1000.0
    start_tick = int(source_windows["measurement_start_tick"])
    records = []
    for index in range(window_count):
        source = source_records[index % len(source_records)]
        modules = []
        for raw_module in source["modules"]:
            module = deepcopy(raw_module)
            multiplier = _positive_multiplier(
                multipliers[module["name"]][index], module["name"], index
            )
            module["dynamic_power_w"] = float(raw_module["dynamic_power_w"]) * multiplier
            module["leakage_power_w"] = float(raw_module["leakage_power_w"]) * multiplier
            module["total_power_w"] = (
                module["dynamic_power_w"] + module["leakage_power_w"]
            )
            validate_power_triplet(module, f"PRBS window {index} module {module['name']}")
            modules.append(module)
        record_start = start_tick + index * nominal_ticks
        record = {
            **deepcopy(source),
            "index": index,
            "source_window_index": int(source["index"]),
            "start_tick": record_start,
            "end_tick": record_start + nominal_ticks,
            "duration_ticks": nominal_ticks,
            "duration_s": nominal_s,
            "is_partial": False,
            "modules": modules,
            "totals": {
                field: sum(float(module[field]) for module in modules)
                for field in POWER_FIELDS
            },
        }
        validate_power_triplet(record["totals"], f"PRBS window {index} totals")
        records.append(record)

    result = {
        **{key: deepcopy(value) for key, value in source_windows.items()
           if key not in ("windows", "window_count", "measurement_end_tick", "timeline_audit", "power_trace_identity")},
        "window_count": window_count,
        "measurement_end_tick": start_tick + window_count * nominal_ticks,
        "windows": records,
        "prbs_excitation": {
            "window_count": window_count,
            "multipliers": deepcopy(multipliers),
            "source_power_trace_identity": power_trace_identity(source_windows),
            "source_window_count": len(source_records),
            "component_scaling": "dynamic and leakage scaled together; total recomputed",
        },
    }
    result["timeline_audit"] = validate_power_windows(result)
    result["power_trace_identity"] = power_trace_identity(result)
    return result


def _require_case_inputs(modules_path: Path, source_power_windows_path: Path,
                         config_path: Path, hotspot: Path, design: dict,
                         settings: ROMSettings) -> tuple[dict, dict, dict]:
    for path in (modules_path, source_power_windows_path, config_path, hotspot):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    if not isinstance(settings, ROMSettings):
        raise ValueError("settings must be ROMSettings")
    if not isinstance(design, dict):
        raise ValueError("design must be a dictionary")
    training, holdout = design.get("training"), design.get("holdout")
    if not isinstance(training, list) or len(training) != 8:
        raise ValueError("design must contain exactly eight training points")
    if not isinstance(holdout, list) or len(holdout) != 2:
        raise ValueError("design must contain exactly two holdout points")
    if not isinstance(design.get("base_layout"), dict):
        raise ValueError("design lacks base_layout")
    modules = read_json(modules_path)
    if not isinstance(modules.get("modules"), list):
        raise ValueError("modules must contain a module list")
    require_design_matches_modules(design, modules)
    source = read_json(source_power_windows_path)
    names = {module.get("name") for module in modules["modules"]}
    validate_power_windows(source, names)
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    frequency = config.get("frequency")
    if not isinstance(frequency, dict):
        raise ValueError("config lacks frequency settings")
    for field in ("f0_ghz", "fmin_ghz"):
        value = frequency.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"frequency.{field} must be finite and positive")
    return modules, source, config


def _case_artifacts(case_dir: Path, package_dir: Path, modules_path: Path,
                    layout_path: Path, power_path: Path, *,
                    include_initialization: bool = False) -> dict:
    package_dir = package_dir.resolve()
    paths = {
        "modules": modules_path.resolve(),
        "layout": layout_path.resolve(),
        "power_windows": power_path.resolve(),
        "power_trace": (case_dir / "power_transient.ptrace").resolve(),
        "temperature_trace": (case_dir / "transient.ttrace").resolve(),
    }
    if include_initialization:
        paths.update({
            "steady_initialization": (
                case_dir / "steady_initialization.json"
            ).resolve(),
            "initialization_steady": (
                case_dir / "initialization.steady.txt"
            ).resolve(),
            "initialization_grid_steady": (
                case_dir / "initialization.grid.steady.txt"
            ).resolve(),
        })
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise ValueError(f"calibration case lacks required artifacts: {missing}")
    try:
        relative_paths = {
            name: path.relative_to(package_dir).as_posix()
            for name, path in paths.items()
        }
    except ValueError as error:
        raise ValueError("calibration case artifact is outside its package") from error
    return {
        **relative_paths,
        "sha256": {name: _sha256(path) for name, path in paths.items()},
    }


def _materialize_case(*, kind: str, point: dict, case_dir: Path,
                      package_dir: Path, modules_path: Path,
                      power_windows: dict, config: dict,
                      design: dict, frequency_ghz: float, f0_ghz: float,
                      period_repeats: int, pss_tolerance_c: float,
                      hotspot: Path) -> dict:
    case_dir.mkdir(parents=True, exist_ok=False)
    layout_path = case_dir / "layout.json"
    power_path = case_dir / "power_windows.json"
    layout = layout_for_point(design["base_layout"], point)
    write_json(layout_path, layout)
    write_json(power_path, power_windows)
    trace = materialize_trace(
        modules_path, layout_path, power_path, case_dir, config,
        frequency_scale=frequency_ghz / f0_ghz,
        period_repeats=period_repeats,
    )
    trace_manifest = read_json(case_dir / "transient_trace_manifest.json")
    trace_manifest["trace_input_identity"]["config_sha256"] = _sha256(
        package_dir / "config.json"
    )
    trace_manifest["hotspot_sha256"] = _sha256(hotspot)
    write_json(case_dir / "transient_trace_manifest.json", trace_manifest)
    initialization = None
    if kind == "training":
        initial_temperature = "ambient"
        thermal = run_hotspot_transient(
            case_dir, hotspot=hotspot, initial_temperature="ambient"
        )
    elif kind == "holdout":
        initial_temperature = "average_power_steady"
        initialization = run_hotspot_steady(case_dir, hotspot=hotspot)
        thermal = run_hotspot_transient(
            case_dir,
            hotspot=hotspot,
            initial_temperature="steady",
            steady_source=case_dir / "initialization.steady.txt",
        )
    else:
        raise ValueError(f"unsupported calibration case kind: {kind!r}")
    trace_manifest = read_json(case_dir / "transient_trace_manifest.json")
    temperature_names, _ = parse_ttrace_grid(case_dir / "transient.ttrace")
    trace_manifest["temperature_grid_unit_names"] = temperature_names
    write_json(case_dir / "transient_trace_manifest.json", trace_manifest)
    result = {
        "id": point["id"],
        "kind": kind,
        "point": deepcopy(point),
        "initial_temperature": initial_temperature,
        "frequency_ghz": frequency_ghz,
        "frequency_scale": frequency_ghz / f0_ghz,
        "window_count": int(trace["window_count"]),
        "hotspot": {
            "command": thermal.get("command"),
            "elapsed_seconds": thermal.get("elapsed_seconds"),
        },
        "artifacts": _case_artifacts(
            case_dir, package_dir, modules_path, layout_path, power_path,
            include_initialization=initialization is not None,
        ),
    }
    if initialization is not None:
        result["steady_initialization"] = {
            "command": initialization.get("command"),
            "return_code": initialization.get("return_code"),
            "elapsed_seconds": initialization.get("elapsed_seconds"),
            "input_sha256": initialization.get("input_sha256"),
            "output_sha256": initialization.get("output_sha256"),
        }
    if kind == "holdout":
        names, rows = parse_ttrace_grid(case_dir / "transient.ttrace")
        convergence = summarize_period_end_convergence(
            rows, int(trace["windows_per_period"])
        )
        converged = (
            convergence["period_count"] >= 2
            and convergence["last_delta_max_c"] <= pss_tolerance_c
        )
        result["periodic_steady_state"] = {
            "period_repeats": period_repeats,
            "pss_tolerance_c": pss_tolerance_c,
            "full_grid_converged": converged,
            "grid_unit_names": names,
            "evidence": convergence,
        }
        if not converged:
            raise ValueError(f"holdout {point['id']} did not reach full-grid PSS")
    return result


def execute_calibration_cases(modules_path: Path, source_power_windows_path: Path,
                              config_path: Path, design: dict, output_dir: Path,
                              settings: ROMSettings, *,
                              hotspot: Path = DEFAULT_HOTSPOT,
                              identity: dict) -> dict:
    """Materialize eight PRBS anchors and two preconditioned holdouts.

    This is intentionally a caller-controlled real-HotSpot boundary.  Unit
    tests mock ``run_hotspot_transient``; this function never launches R1, R2,
    gem5, or McPAT.
    """
    modules_path = Path(modules_path).resolve()
    source_power_windows_path = Path(source_power_windows_path).resolve()
    config_path, hotspot = Path(config_path).resolve(), Path(hotspot).resolve()
    output_dir = require_new_output_root(
        Path(output_dir),
        [modules_path, source_power_windows_path, config_path, hotspot],
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"calibration output directory is not empty: {output_dir}")
    modules, source_power_windows, config = _require_case_inputs(
        modules_path, source_power_windows_path, config_path, hotspot, design, settings
    )
    design_identity = calibration_design_hash(design)
    if not isinstance(identity, dict):
        raise ValueError("calibration identity must be a dictionary")
    if identity.get("calibration_design_hash") != design_identity:
        raise ValueError("calibration design hash identity differs")
    if identity.get("allowed_l2_tiers") != design.get("allowed_l2_tiers"):
        raise ValueError("calibration allowed tiers identity differs")
    output_dir.mkdir(parents=True, exist_ok=True)
    package_modules_path = output_dir / "modules.json"
    package_power_windows_path = output_dir / "source_power_windows.json"
    package_config_path = output_dir / "config.json"
    shutil.copyfile(modules_path, package_modules_path)
    shutil.copyfile(source_power_windows_path, package_power_windows_path)
    shutil.copyfile(config_path, package_config_path)
    write_json(output_dir / "anchors.json", design)
    write_json(output_dir / "calibration_manifest.json", {
        "schema_version": 1,
        "mode": "transient ROM calibration",
        **ROM_CLASSIFICATION,
        "calibration_runs": 8,
        "validation_runs": 2,
        "initialization_runs": 2,
        "calibration_hotspot_calls": 12,
        "calibration_design_hash": design_identity,
        "identity": deepcopy(identity),
        "settings": {
            **asdict(settings),
            "calibration_runs": 8,
            "validation_runs": 2,
            "initialization_runs": 2,
            "calibration_hotspot_calls": 12,
        },
        "sources": {
            "modules": "modules.json",
            "modules_sha256": _sha256(package_modules_path),
            "power_windows": "source_power_windows.json",
            "power_windows_sha256": _sha256(package_power_windows_path),
            "power_trace_identity": power_trace_identity(source_power_windows),
            "config": "config.json",
            "config_sha256": _sha256(package_config_path),
            "hotspot_sha256": _sha256(hotspot),
        },
        "training_ids": [point["id"] for point in design["training"]],
        "holdout_ids": [point["id"] for point in design["holdout"]],
    })
    prbs = design.get("prbs")
    if not isinstance(prbs, dict) or not isinstance(prbs.get("multipliers"), dict):
        raise ValueError("design lacks PRBS multipliers")
    prbs_power = build_prbs_power_windows(
        source_power_windows, prbs["multipliers"], settings.calibration_windows
    )
    f0_ghz = float(config["frequency"]["f0_ghz"])
    fmin_ghz = float(config["frequency"]["fmin_ghz"])
    training = []
    for point in design["training"]:
        training.append(_materialize_case(
            kind="training", point=point, case_dir=output_dir / f"training_{point['id']}",
            package_dir=output_dir, modules_path=package_modules_path,
            power_windows=prbs_power, config=config,
            design=design, frequency_ghz=f0_ghz, f0_ghz=f0_ghz,
            period_repeats=1, pss_tolerance_c=settings.pss_tolerance_c, hotspot=hotspot,
        ))
    holdout_frequencies = (f0_ghz, 0.6 * f0_ghz + 0.4 * fmin_ghz)
    holdout = []
    for point, frequency_ghz in zip(design["holdout"], holdout_frequencies):
        holdout.append(_materialize_case(
            kind="holdout", point=point, case_dir=output_dir / f"holdout_{point['id']}",
            package_dir=output_dir, modules_path=package_modules_path,
            power_windows=source_power_windows, config=config,
            design=design, frequency_ghz=frequency_ghz, f0_ghz=f0_ghz,
            period_repeats=settings.pss_period_repeats,
            pss_tolerance_c=settings.pss_tolerance_c, hotspot=hotspot,
        ))
    report = {
        "schema_version": 1,
        "mode": "transient ROM calibration materialization",
        **ROM_CLASSIFICATION,
        "source_modules": "modules.json",
        "source_power_windows": "source_power_windows.json",
        "source_power_trace_identity": power_trace_identity(source_power_windows),
        "settings": {
            "calibration_windows": settings.calibration_windows,
            "pss_period_repeats": settings.pss_period_repeats,
            "pss_tolerance_c": settings.pss_tolerance_c,
        },
        "training_hotspot_calls": len(training),
        "holdout_initialization_hotspot_calls": len(holdout),
        "holdout_transient_hotspot_calls": len(holdout),
        "calibration_hotspot_calls": len(training) + 2 * len(holdout),
        "training_cases": training,
        "holdout_cases": holdout,
    }
    write_json(output_dir / "calibration_cases.json", report)
    return report
