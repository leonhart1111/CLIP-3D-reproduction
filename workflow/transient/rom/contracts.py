"""Strict settings and provenance contracts for reusable transient ROM packages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from workflow.common import read_json, sha256_file, write_json
from workflow.transient.rom.evidence import (
    ROM_CLASSIFICATION,
    require_regular_descendant,
    require_rom_classification,
    sha256_identity,
)


@dataclass(frozen=True)
class ROMSettings:
    sample_interval_ms: float
    calibration_windows: int
    prbs_seed: int
    prbs_fraction: float
    pod_energy_threshold: float
    max_pod_rank: int
    ridge: float
    max_condition_number: float
    max_logm_condition_number: float
    max_logm_error_estimate: float
    max_exp_log_reconstruction_error: float
    pss_period_repeats: int
    pss_tolerance_c: float
    frequency_tolerance_ghz: float
    max_holdout_peak_error_c: float
    max_holdout_grid_rmse_c: float
    search_grid_points_per_axis: int
    refinement_starts: int


_DEFAULTS = {
    "sample_interval_ms": 2.0,
    "calibration_windows": 64,
    "prbs_seed": 20260807,
    "prbs_fraction": 0.20,
    "pod_energy_threshold": 0.999,
    "max_pod_rank": 16,
    "ridge": 1e-8,
    "max_condition_number": 1e10,
    "max_logm_condition_number": 1e12,
    "max_logm_error_estimate": 1e-8,
    "max_exp_log_reconstruction_error": 1e-8,
    "pss_period_repeats": 20,
    "pss_tolerance_c": 0.01,
    "frequency_tolerance_ghz": 0.01,
    "max_holdout_peak_error_c": 1.0,
    "max_holdout_grid_rmse_c": 0.75,
    "search_grid_points_per_axis": 25,
    "refinement_starts": 5,
}
_REQUIRED_IDENTITY_FIELDS = (
    "canonical_r1_metadata_hash",
    "power_trace",
    "modules_geometry_hash",
    "layout_geometry_hash",
    "configuration_hash",
    "hotspot_hash",
    "grid",
    "stack",
    "cooling",
    "allowed_l2_tiers",
    "calibration_design_hash",
)
_IDENTITY_LABELS = {
    "canonical_r1_metadata_hash": "canonical R1 metadata hash identity",
    "power_trace": "power trace identity",
    "modules_geometry_hash": "modules geometry hash identity",
    "layout_geometry_hash": "layout geometry hash identity",
    "configuration_hash": "configuration hash identity",
    "hotspot_hash": "HotSpot hash identity",
    "grid": "grid identity",
    "stack": "stack identity",
    "cooling": "cooling identity",
    "allowed_l2_tiers": "allowed tiers identity",
    "calibration_design_hash": "calibration design hash identity",
}
_CLASSIFICATION = ROM_CLASSIFICATION


def write_rom_artifact_manifest(output_dir: Path) -> dict:
    """Bind every ROM-owned file to its bytes and non-formal classification."""
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "rom_artifact_manifest.json"
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("ROM artifact manifest cannot bind symlinks")
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


def require_rom_artifact_manifest(package_dir: Path) -> dict:
    """Require a complete, current, symlink-free ROM artifact inventory."""
    package_dir = Path(package_dir).resolve()
    try:
        manifest_path = require_regular_descendant(
            package_dir, "rom_artifact_manifest.json", "reusable ROM package manifest"
        )
    except ValueError as error:
        raise ValueError("reusable ROM package lacks rom_artifact_manifest.json")
    manifest = read_json(manifest_path)
    require_rom_classification(manifest, "reusable ROM package manifest")
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
            raise ValueError(
                "reusable ROM package artifact classification is invalid"
            )
        recorded[relative] = record

    actual: dict[str, Path] = {}
    for path in sorted(package_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("reusable ROM package manifest cannot bind symlinks")
        if path == manifest_path or not path.is_file():
            continue
        actual[path.relative_to(package_dir).as_posix()] = path
    if set(recorded) != set(actual):
        raise ValueError(
            "reusable ROM package manifest inventory is incomplete or stale"
        )
    for relative, path in actual.items():
        if recorded[relative].get("sha256") != sha256_file(path):
            raise ValueError(
                f"reusable ROM package manifest hash differs for {relative}"
            )
    return manifest


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def parse_settings(config: dict) -> ROMSettings:
    """Parse bounded ROM settings, enforcing the fixed eight-plus-two design."""
    if not isinstance(config, dict):
        raise ValueError("configuration must be a dictionary")
    values = config.get("transient_rom", {})
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("transient_rom must be a dictionary")

    calibration_runs = values.get("calibration_runs", 8)
    if isinstance(calibration_runs, bool) or not isinstance(calibration_runs, int) or calibration_runs != 8:
        raise ValueError("calibration_runs must equal 8")
    validation_runs = values.get("validation_runs", 2)
    if isinstance(validation_runs, bool) or not isinstance(validation_runs, int) or validation_runs != 2:
        raise ValueError("validation_runs must equal 2")

    setting = {name: values.get(name, default) for name, default in _DEFAULTS.items()}
    sample_interval_ms = _positive(setting["sample_interval_ms"], "sample_interval_ms")
    calibration_windows = _integer(setting["calibration_windows"], "calibration_windows")
    prbs_seed = _integer(setting["prbs_seed"], "prbs_seed", minimum=0)
    prbs_fraction = _positive(setting["prbs_fraction"], "prbs_fraction")
    if prbs_fraction >= 1.0:
        raise ValueError("prbs_fraction must be less than 1")
    pod_energy_threshold = _positive(
        setting["pod_energy_threshold"], "pod_energy_threshold"
    )
    if pod_energy_threshold >= 1.0:
        raise ValueError("pod_energy_threshold must be less than 1")
    max_pod_rank = _integer(setting["max_pod_rank"], "max_pod_rank")
    ridge = _positive(setting["ridge"], "ridge")
    max_condition_number = _positive(
        setting["max_condition_number"], "max_condition_number"
    )
    if max_condition_number < 1.0:
        raise ValueError("max_condition_number must be at least 1")
    max_logm_condition_number = _positive(
        setting["max_logm_condition_number"], "max_logm_condition_number"
    )
    if max_logm_condition_number < 1.0:
        raise ValueError("max_logm_condition_number must be at least 1")
    max_logm_error_estimate = _positive(
        setting["max_logm_error_estimate"], "max_logm_error_estimate"
    )
    max_exp_log_reconstruction_error = _positive(
        setting["max_exp_log_reconstruction_error"],
        "max_exp_log_reconstruction_error",
    )
    pss_period_repeats = _integer(setting["pss_period_repeats"], "pss_period_repeats")
    pss_tolerance_c = _positive(setting["pss_tolerance_c"], "pss_tolerance_c")
    frequency_tolerance_ghz = _positive(
        setting["frequency_tolerance_ghz"], "frequency_tolerance_ghz"
    )
    max_holdout_peak_error_c = _positive(
        setting["max_holdout_peak_error_c"], "max_holdout_peak_error_c"
    )
    max_holdout_grid_rmse_c = _positive(
        setting["max_holdout_grid_rmse_c"], "max_holdout_grid_rmse_c"
    )
    search_grid_points_per_axis = _integer(
        setting["search_grid_points_per_axis"], "search_grid_points_per_axis", 2
    )
    refinement_starts = _integer(setting["refinement_starts"], "refinement_starts")
    return ROMSettings(
        sample_interval_ms, calibration_windows, prbs_seed, prbs_fraction,
        pod_energy_threshold, max_pod_rank, ridge, max_condition_number,
        max_logm_condition_number, max_logm_error_estimate,
        max_exp_log_reconstruction_error,
        pss_period_repeats, pss_tolerance_c, frequency_tolerance_ghz,
        max_holdout_peak_error_c, max_holdout_grid_rmse_c,
        search_grid_points_per_axis, refinement_starts,
    )


def _json_value(value: Any, name: str) -> Any:
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be JSON-serializable finite provenance") from error
    return json.loads(encoded)


def rom_input_identity(
    *,
    canonical_r1_metadata_hash: str,
    power_trace: str,
    modules_geometry_hash: str,
    layout_geometry_hash: str,
    configuration_hash: str,
    hotspot_hash: str,
    grid: Any,
    stack: Any,
    cooling: Any,
    allowed_l2_tiers: list[int] | tuple[int, ...],
    calibration_design_hash: str,
) -> dict:
    """Return the complete scientific provenance required to reuse a ROM."""
    hashes = {
        "canonical_r1_metadata_hash": canonical_r1_metadata_hash,
        "power_trace": power_trace,
        "modules_geometry_hash": modules_geometry_hash,
        "layout_geometry_hash": layout_geometry_hash,
        "configuration_hash": configuration_hash,
        "hotspot_hash": hotspot_hash,
        "calibration_design_hash": calibration_design_hash,
    }
    for name, value in hashes.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty hash identity")
    tiers = list(allowed_l2_tiers) if isinstance(allowed_l2_tiers, tuple) else allowed_l2_tiers
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("allowed_l2_tiers must be a non-empty list")
    if any(isinstance(tier, bool) or not isinstance(tier, int) or tier < 0 for tier in tiers):
        raise ValueError("allowed_l2_tiers must contain non-negative integer tiers")
    if len(set(tiers)) != len(tiers):
        raise ValueError("allowed_l2_tiers must not contain duplicates")
    return {
        **hashes,
        "grid": _json_value(grid, "grid"),
        "stack": _json_value(stack, "stack"),
        "cooling": _json_value(cooling, "cooling"),
        "allowed_l2_tiers": tiers,
    }


def require_accepted_package(package_dir: Path, identity: dict) -> dict:
    """Load a ROM acceptance record only when every provenance field matches."""
    package_dir = Path(package_dir)
    try:
        path = require_regular_descendant(
            package_dir, "rom_acceptance.json", "ROM package acceptance"
        )
        acceptance = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("ROM package is not accepted: missing valid rom_acceptance.json") from error
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise ValueError("ROM package is not accepted")
    require_rom_classification(acceptance, "ROM package acceptance")
    package_identity = acceptance.get("identity")
    package_identity = _normalized_identity(package_identity, "ROM package")
    identity = _normalized_identity(identity, "requested ROM")
    if package_identity != identity:
        for name in _REQUIRED_IDENTITY_FIELDS:
            if package_identity[name] != identity[name]:
                raise ValueError(_IDENTITY_LABELS[name])
        raise ValueError("ROM package identity differs")
    return acceptance


def require_package_calibration_evidence(
    package_dir: Path, identity: dict, settings: ROMSettings,
) -> dict:
    """Require the complete fitted model and 8+2 evidence used for acceptance."""
    package_dir = Path(package_dir).resolve()
    acceptance = require_accepted_package(package_dir, identity)
    from workflow.transient.rom.calibration_design import (  # avoid import cycle
        calibration_design_hash,
        layout_for_point,
        load_calibration_design,
    )

    design = load_calibration_design(package_dir)
    design_identity = calibration_design_hash(design)
    values = {}
    for name in (
        "calibration_manifest.json", "calibration_cases.json",
        "fit_report.json", "validation_report.json",
    ):
        try:
            path = require_regular_descendant(
                package_dir, name, f"reusable ROM package {name}"
            )
        except ValueError as error:
            raise ValueError(f"reusable ROM package lacks {name}") from error
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"reusable ROM package has invalid {name}")
        values[name] = value

    manifest = values["calibration_manifest.json"]
    cases = values["calibration_cases.json"]
    fit_report = values["fit_report.json"]
    validation = values["validation_report.json"]
    expected_settings = {
        **asdict(settings),
        "calibration_runs": 8,
        "validation_runs": 2,
    }
    if manifest.get("settings") != expected_settings:
        raise ValueError("reusable ROM calibration manifest resolved settings differ")
    try:
        require_rom_classification(manifest, "reusable ROM calibration manifest")
    except ValueError as error:
        raise ValueError("reusable ROM calibration manifest classification differs") from error
    require_rom_classification(cases, "reusable ROM calibration cases report")
    require_rom_classification(fit_report, "reusable ROM fit report")
    if (manifest.get("identity") != acceptance["identity"]
            or manifest.get("calibration_design_hash") != design_identity
            or manifest.get("calibration_runs") != 8
            or manifest.get("validation_runs") != 2):
        raise ValueError("reusable ROM calibration manifest evidence differs")
    sources = manifest.get("sources")
    expected_source_identities = {
        "modules_sha256": acceptance["identity"]["modules_geometry_hash"],
        "power_trace_identity": acceptance["identity"]["power_trace"],
        "config_sha256": acceptance["identity"]["configuration_hash"],
        "hotspot_sha256": acceptance["identity"]["hotspot_hash"],
    }
    if (not isinstance(sources, dict) or any(
        sources.get(field) != value
        for field, value in expected_source_identities.items()
    )):
        raise ValueError("reusable ROM calibration manifest source identities differ")
    from workflow.transient.validation import power_trace_identity

    packaged_sources: dict[str, Path] = {}
    for field, context in (
        ("modules", "modules"),
        ("power_windows", "source power"),
        ("config", "config"),
    ):
        relative = sources.get(field) if isinstance(sources, dict) else None
        if not isinstance(relative, str):
            raise ValueError(
                f"reusable ROM calibration manifest lacks packaged {context}"
            )
        try:
            packaged_sources[field] = require_regular_descendant(
                package_dir, relative, f"reusable ROM calibration {context}"
            )
        except ValueError as error:
            raise ValueError(
                f"reusable ROM calibration {context} evidence differs"
            ) from error
    if sha256_identity(packaged_sources["modules"]) != expected_source_identities["modules_sha256"]:
        raise ValueError("reusable ROM calibration modules evidence differs")
    packaged_source_power = read_json(packaged_sources["power_windows"])
    if (sha256_identity(packaged_sources["power_windows"])
            != sources["power_windows_sha256"]
            or power_trace_identity(packaged_source_power)
            != expected_source_identities["power_trace_identity"]):
        raise ValueError("reusable ROM calibration source power evidence differs")
    packaged_config_path = packaged_sources["config"]
    if sources["config_sha256"] != sha256_identity(packaged_config_path):
        raise ValueError("reusable ROM calibration config evidence differs")
    packaged_config = read_json(packaged_config_path)
    if not isinstance(packaged_config, dict):
        raise ValueError("reusable ROM packaged config must be a dictionary")
    expected_modules_hash = expected_source_identities["modules_sha256"]
    if acceptance.get("validation_report") != "validation_report.json":
        raise ValueError("reusable ROM acceptance validation evidence path differs")

    training, holdouts = cases.get("training_cases"), cases.get("holdout_cases")
    expected_training = [point["id"] for point in design["training"]]
    expected_holdouts = [point["id"] for point in design["holdout"]]
    if (cases.get("training_hotspot_calls") != 8
            or cases.get("holdout_hotspot_calls") != 2
            or not isinstance(training, list) or len(training) != 8
            or not isinstance(holdouts, list) or len(holdouts) != 2
            or [case.get("id") for case in training] != expected_training
            or [case.get("id") for case in holdouts] != expected_holdouts):
        raise ValueError("reusable ROM calibration case evidence must contain exact 8+2")
    if (manifest.get("training_ids") != expected_training
            or manifest.get("holdout_ids") != expected_holdouts):
        raise ValueError("reusable ROM calibration manifest case ids differ")

    expected_points = {
        point["id"]: point for point in [*design["training"], *design["holdout"]]
    }
    artifact_hashes: dict[str, dict[str, str]] = {}
    case_power_windows: dict[str, dict] = {}
    for expected_kind, case_set in (("training", training), ("holdout", holdouts)):
        for case in case_set:
            identifier = case["id"]
            if (case.get("kind") != expected_kind
                    or case.get("point") != expected_points[identifier]
                    or case.get("initial_temperature") != "ambient"
                    or not isinstance(case.get("hotspot"), dict)):
                raise ValueError(
                    f"reusable ROM {expected_kind} case {identifier} evidence differs"
                )
            artifacts = case.get("artifacts")
            hashes = artifacts.get("sha256") if isinstance(artifacts, dict) else None
            artifact_hashes[identifier] = {}
            resolved_artifacts: dict[str, Path] = {}
            for artifact in (
                "modules", "layout", "power_windows", "power_trace",
                "temperature_trace",
            ):
                value = artifacts.get(artifact) if isinstance(artifacts, dict) else None
                recorded = hashes.get(artifact) if isinstance(hashes, dict) else None
                if not isinstance(value, str) or not isinstance(recorded, str):
                    raise ValueError(
                        f"reusable ROM {expected_kind} case {identifier} "
                        f"lacks {artifact} hash"
                    )
                try:
                    path = require_regular_descendant(
                        package_dir, value,
                        f"reusable ROM {expected_kind} case {identifier} {artifact}",
                    )
                except ValueError as error:
                    raise ValueError(
                        f"reusable ROM {expected_kind} case {identifier} "
                        f"{artifact} evidence differs: {error}"
                    ) from error
                if recorded != sha256_identity(path):
                    raise ValueError(
                        f"reusable ROM {expected_kind} case {identifier} "
                        f"{artifact} hash differs"
                    )
                artifact_hashes[identifier][artifact] = recorded
                resolved_artifacts[artifact] = path
            if artifact_hashes[identifier]["modules"] != expected_modules_hash:
                raise ValueError(
                    f"reusable ROM {expected_kind} case {identifier} "
                    "modules evidence differs"
                )
            expected_layout = layout_for_point(
                design["base_layout"], expected_points[identifier]
            )
            if read_json(resolved_artifacts["layout"]) != expected_layout:
                raise ValueError(
                    f"reusable ROM {expected_kind} case {identifier} "
                    "layout evidence differs"
                )
            power_windows = read_json(resolved_artifacts["power_windows"])
            case_power_windows[identifier] = power_windows
            if expected_kind == "training":
                excitation = (
                    power_windows.get("prbs_excitation")
                    if isinstance(power_windows, dict) else None
                )
                if (not isinstance(excitation, dict)
                        or excitation.get("source_power_trace_identity")
                        != acceptance["identity"]["power_trace"]
                        or excitation.get("window_count")
                        != settings.calibration_windows
                        or excitation.get("multipliers")
                        != design["prbs"]["multipliers"]):
                    raise ValueError(
                        f"reusable ROM training case {identifier} "
                        "source power identity differs"
                    )
            elif power_trace_identity(power_windows) != acceptance["identity"]["power_trace"]:
                raise ValueError(
                    f"reusable ROM holdout case {identifier} "
                    "source power identity differs"
                )

    holdout_power_hashes = {
        artifact_hashes[identifier]["power_windows"]
        for identifier in expected_holdouts
    }
    if (len(holdout_power_hashes) != 1
            or sources.get("power_windows_sha256") not in holdout_power_hashes):
        raise ValueError("reusable ROM calibration source power bytes differ")
    from workflow.transient.rom.materialize_calibration import (  # avoid import cycle
        build_prbs_power_windows,
    )

    expected_prbs_power = build_prbs_power_windows(
        case_power_windows[expected_holdouts[0]],
        design["prbs"]["multipliers"],
        settings.calibration_windows,
    )
    for identifier in expected_training:
        if case_power_windows[identifier] != expected_prbs_power:
            raise ValueError(
                f"reusable ROM training case {identifier} "
                "PRBS power evidence differs"
            )

    expected_case_order = sorted(expected_training)
    expected_temperature_hashes = {
        identifier: artifact_hashes[identifier]["temperature_trace"]
        for identifier in expected_case_order
    }
    expected_power_hashes = {
        identifier: artifact_hashes[identifier]["power_windows"]
        for identifier in expected_case_order
    }
    if (fit_report.get("calibration_design_hash") != design_identity
            or fit_report.get("case_order") != expected_case_order
            or fit_report.get("temperature_trace_sha256")
            != expected_temperature_hashes
            or fit_report.get("power_windows_sha256") != expected_power_hashes):
        raise ValueError("reusable ROM fit report training evidence differs")
    identification = fit_report.get("identification")
    if (not isinstance(identification, dict)
            or identification.get("ridge") != settings.ridge
            or identification.get("max_condition_number")
            != settings.max_condition_number):
        raise ValueError("reusable ROM fit report identification settings differ")
    gram_condition = identification.get("gram_condition_number")
    if (isinstance(gram_condition, bool) or not isinstance(gram_condition, Real)
            or not math.isfinite(float(gram_condition))
            or float(gram_condition) > settings.max_condition_number):
        raise ValueError("reusable ROM fit report condition evidence differs")
    pod = fit_report.get("pod")
    if (not isinstance(pod, dict)
            or pod.get("energy_threshold") != settings.pod_energy_threshold):
        raise ValueError("reusable ROM fit report POD settings differ")
    rank = pod.get("rank")
    if (isinstance(rank, bool) or not isinstance(rank, int)
            or rank < 1 or rank > settings.max_pod_rank):
        raise ValueError("reusable ROM fit report POD rank differs")
    conversion = fit_report.get("continuous_conversion")
    conversion_gates = (
        ("logm_input_condition_number", "max_logm_condition_number",
         settings.max_logm_condition_number),
        ("logm_error_estimate", "max_logm_error_estimate",
         settings.max_logm_error_estimate),
        ("exp_log_reconstruction_error", "max_exp_log_reconstruction_error",
         settings.max_exp_log_reconstruction_error),
    )
    if not isinstance(conversion, dict):
        raise ValueError("reusable ROM fit report lacks continuous conversion evidence")
    for value_name, limit_name, expected_limit in conversion_gates:
        value, limit = conversion.get(value_name), conversion.get(limit_name)
        if (limit != expected_limit or isinstance(value, bool)
                or not isinstance(value, Real) or not math.isfinite(float(value))
                or float(value) > expected_limit):
            raise ValueError(
                f"reusable ROM fit report {value_name} evidence differs"
            )

    from workflow.transient.rom.pod_state_space import load_model  # avoid cycle

    model, model_metadata = load_model(package_dir / "pod_model.npz")
    if model_metadata != fit_report:
        raise ValueError("reusable ROM POD model metadata differs from fit report")
    if sorted(model.b_l2_anchors) != expected_case_order:
        raise ValueError("saved calibration design anchors differ from POD B_L2 columns")
    from workflow.transient.rom.calibrate_rom import validate_calibration_holdouts

    recomputed_validation = validate_calibration_holdouts(
        model, design, holdouts, settings, packaged_config,
        output_dir=package_dir, identity=None, publish=False,
    )
    if recomputed_validation != validation:
        raise ValueError(
            "reusable ROM recomputed holdout validation differs from persisted report"
        )
    if recomputed_validation.get("accepted") is not True:
        raise ValueError("reusable ROM recomputed holdout validation is not accepted")
    artifact_manifest = require_rom_artifact_manifest(package_dir)
    return {
        "acceptance": acceptance,
        "design": design,
        "manifest": manifest,
        "cases": cases,
        "fit_report": fit_report,
        "model": model,
        "model_metadata": model_metadata,
        "validation": validation,
        "artifact_manifest": artifact_manifest,
    }


def _normalized_identity(value: Any, context: str) -> dict:
    """Require a JSON-normalized identity with exactly the scientific fields."""
    normalized = _json_value(value, f"{context} identity")
    if not isinstance(normalized, dict):
        raise ValueError(f"{context} identity must be a dictionary")
    for name in _REQUIRED_IDENTITY_FIELDS:
        if name not in normalized:
            raise ValueError(_IDENTITY_LABELS[name])
    extras = set(normalized) - set(_REQUIRED_IDENTITY_FIELDS)
    if extras:
        raise ValueError(f"{context} identity contains unsupported fields")
    return normalized
