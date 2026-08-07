"""Strict settings and provenance contracts for reusable transient ROM packages."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from workflow.common import read_json


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
}


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
) -> dict:
    """Return the complete scientific provenance required to reuse a ROM."""
    hashes = {
        "canonical_r1_metadata_hash": canonical_r1_metadata_hash,
        "power_trace": power_trace,
        "modules_geometry_hash": modules_geometry_hash,
        "layout_geometry_hash": layout_geometry_hash,
        "configuration_hash": configuration_hash,
        "hotspot_hash": hotspot_hash,
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
    path = Path(package_dir) / "rom_acceptance.json"
    try:
        acceptance = read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("ROM package is not accepted: missing valid rom_acceptance.json") from error
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise ValueError("ROM package is not accepted")
    package_identity = acceptance.get("identity")
    package_identity = _normalized_identity(package_identity, "ROM package")
    identity = _normalized_identity(identity, "requested ROM")
    if package_identity != identity:
        for name in _REQUIRED_IDENTITY_FIELDS:
            if package_identity[name] != identity[name]:
                raise ValueError(_IDENTITY_LABELS[name])
        raise ValueError("ROM package identity differs")
    return acceptance


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
