"""POD reduction and stable continuous-time identification for transient ROMs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy
from scipy.linalg import expm, logm

from workflow.common import read_json
from workflow.transient.rom.contracts import ROMSettings
from workflow.transient.run_hotspot_transient import parse_ttrace_grid


_IMAGINARY_RELATIVE_TOLERANCE = 1e-9
_STABILITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class StateSpaceModel:
    temperature_basis: numpy.ndarray
    a_continuous: numpy.ndarray
    b_fixed: numpy.ndarray
    b_l2_anchors: dict[str, numpy.ndarray]
    module_names: tuple[str, ...]


def _finite_real_array(value: Any, name: str, ndim: int) -> numpy.ndarray:
    array = numpy.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array")
    if numpy.iscomplexobj(array) and numpy.any(numpy.imag(array) != 0.0):
        raise ValueError(f"{name} must be real")
    result = numpy.asarray(numpy.real(array), dtype=float)
    if not numpy.all(numpy.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validated_model(model: StateSpaceModel) -> tuple[int, int]:
    if not isinstance(model, StateSpaceModel):
        raise ValueError("model must be a StateSpaceModel")
    basis = _finite_real_array(model.temperature_basis, "temperature_basis", 2)
    a_continuous = _finite_real_array(model.a_continuous, "a_continuous", 2)
    b_fixed = _finite_real_array(model.b_fixed, "b_fixed", 2)
    rank = basis.shape[1]
    if rank < 1 or basis.shape[0] < rank:
        raise ValueError("temperature_basis has invalid dimensions")
    if a_continuous.shape != (rank, rank):
        raise ValueError("a_continuous dimensions do not match temperature_basis")
    if float(numpy.max(numpy.real(numpy.linalg.eigvals(a_continuous)))) > _STABILITY_TOLERANCE:
        raise ValueError("unstable continuous-time state matrix")
    if b_fixed.shape != (rank, len(model.module_names)):
        raise ValueError("b_fixed dimensions do not match module_names")
    if any(not isinstance(name, str) or not name for name in model.module_names):
        raise ValueError("module_names must contain non-empty strings")
    if len(set(model.module_names)) != len(model.module_names):
        raise ValueError("module_names must not contain duplicates")
    if not isinstance(model.b_l2_anchors, dict) or not model.b_l2_anchors:
        raise ValueError("b_l2_anchors must be a non-empty dictionary")
    for anchor, column in model.b_l2_anchors.items():
        if not isinstance(anchor, str) or not anchor:
            raise ValueError("L2 anchor names must be non-empty strings")
        if _finite_real_array(column, f"b_l2_anchors[{anchor!r}]", 2).shape != (rank, 1):
            raise ValueError(f"b_l2_anchors[{anchor!r}] must have shape ({rank}, 1)")
    return rank, len(model.module_names)


def _case_id(case: dict) -> str:
    point = case.get("point")
    identifier = case.get("id")
    if identifier is None and isinstance(point, dict):
        identifier = point.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("every training case must have a non-empty anchor id")
    if isinstance(point, dict) and point.get("id") not in (None, identifier):
        raise ValueError(f"training case {identifier} has inconsistent point id")
    return identifier


def _artifact(case: dict, name: str, identifier: str) -> Path:
    artifacts = case.get("artifacts")
    value = artifacts.get(name) if isinstance(artifacts, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"training case {identifier} lacks {name} artifact")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonicalize_basis_signs(basis: numpy.ndarray) -> numpy.ndarray:
    result = basis.copy()
    for column in range(result.shape[1]):
        pivot = int(numpy.argmax(numpy.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _load_training_case(case: dict, expected_dt: float,
                        expected_windows: int) -> dict:
    identifier = _case_id(case)
    if case.get("initial_temperature") != "ambient":
        raise ValueError(
            f"training case {identifier} must record ambient initial temperature"
        )
    temperature_path = _artifact(case, "temperature_trace", identifier)
    power_path = _artifact(case, "power_windows", identifier)
    names, temperatures_k = parse_ttrace_grid(temperature_path)
    power_windows = read_json(power_path)
    windows = power_windows.get("windows") if isinstance(power_windows, dict) else None
    if not isinstance(windows, list):
        raise ValueError(f"training case {identifier} has invalid power windows")
    if len(windows) != expected_windows or len(temperatures_k) != expected_windows:
        raise ValueError(
            f"training case {identifier} must contain settings.calibration_windows "
            f"({expected_windows}) temperature and power samples"
        )
    manifest_path = temperature_path.parent / "hotspot_manifest.json"
    manifest = read_json(manifest_path)
    ambient_c = manifest.get("ambient_c") if isinstance(manifest, dict) else None
    if isinstance(ambient_c, bool) or not isinstance(ambient_c, (int, float)):
        raise ValueError(f"training case {identifier} lacks finite ambient_c")
    ambient_k = float(ambient_c) + 273.15
    if not math.isfinite(ambient_k):
        raise ValueError(f"training case {identifier} lacks finite ambient_c")
    temperatures = numpy.asarray(temperatures_k, dtype=float) - ambient_k
    if temperatures.ndim != 2 or not numpy.all(numpy.isfinite(temperatures)):
        raise ValueError(f"training case {identifier} has invalid temperatures")

    module_kinds: dict[str, str] | None = None
    powers: list[dict[str, float]] = []
    for position, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"training case {identifier} has invalid power window")
        duration = window.get("duration_s", expected_dt)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(f"training case {identifier} has invalid window duration")
        if not math.isfinite(float(duration)) or not math.isclose(
            float(duration), expected_dt, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(
                f"training case {identifier} window duration differs from sample interval"
            )
        modules = window.get("modules")
        if not isinstance(modules, list) or not modules:
            raise ValueError(f"training case {identifier} has invalid module powers")
        current_kinds: dict[str, str] = {}
        current_powers: dict[str, float] = {}
        for module in modules:
            if not isinstance(module, dict):
                raise ValueError(f"training case {identifier} has invalid module powers")
            name, kind = module.get("name"), module.get("kind")
            power = module.get("total_power_w")
            if not isinstance(name, str) or not name or name in current_kinds:
                raise ValueError(f"training case {identifier} has invalid module names")
            if not isinstance(kind, str) or not kind:
                raise ValueError(f"training case {identifier} has invalid module kind")
            if isinstance(power, bool) or not isinstance(power, (int, float)):
                raise ValueError(f"training case {identifier} has invalid module power")
            value = float(power)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"training case {identifier} has invalid module power")
            current_kinds[name] = kind
            current_powers[name] = value
        if module_kinds is None:
            module_kinds = current_kinds
        elif current_kinds != module_kinds:
            raise ValueError(
                f"training case {identifier} module set changed at window {position}"
            )
        powers.append(current_powers)
    assert module_kinds is not None
    l2_names = sorted(name for name, kind in module_kinds.items() if kind == "l2")
    if len(l2_names) != 1:
        raise ValueError(f"training case {identifier} must contain exactly one L2 module")
    return {
        "id": identifier,
        "grid_names": tuple(names),
        "temperatures": temperatures,
        "module_kinds": module_kinds,
        "powers": powers,
        "l2_name": l2_names[0],
        "temperature_sha256": _sha256(temperature_path),
        "power_sha256": _sha256(power_path),
    }


def fit_state_space(training_cases: list[dict], settings: ROMSettings) -> tuple[StateSpaceModel, dict]:
    """Fit a shared stable continuous model from eight full-grid anchor traces."""
    if not isinstance(settings, ROMSettings):
        raise ValueError("settings must be ROMSettings")
    if not isinstance(training_cases, list) or len(training_cases) != 8:
        raise ValueError("training_cases must contain exactly eight cases")
    if any(not isinstance(case, dict) for case in training_cases):
        raise ValueError("training_cases must contain dictionaries")
    dt = settings.sample_interval_ms / 1000.0
    ordered_cases = sorted(training_cases, key=_case_id)
    identifiers = [_case_id(case) for case in ordered_cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("training case anchor ids must be unique")
    loaded = [
        _load_training_case(case, dt, settings.calibration_windows)
        for case in ordered_cases
    ]

    grid_names = loaded[0]["grid_names"]
    module_kinds = loaded[0]["module_kinds"]
    l2_name = loaded[0]["l2_name"]
    for case in loaded[1:]:
        if case["grid_names"] != grid_names:
            raise ValueError("training cases have inconsistent full-grid cell order")
        if case["module_kinds"] != module_kinds or case["l2_name"] != l2_name:
            raise ValueError("training cases have inconsistent module inputs")
    fixed_names = tuple(sorted(name for name in module_kinds if name != l2_name))

    snapshots = numpy.concatenate(
        [case["temperatures"].T for case in loaded], axis=1
    )
    u_full, singular_values, _ = numpy.linalg.svd(snapshots, full_matrices=False)
    squared = singular_values ** 2
    total_energy = float(numpy.sum(squared))
    if not math.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("training temperature snapshots have zero POD energy")
    cumulative = numpy.cumsum(squared) / total_energy
    rank = int(numpy.searchsorted(cumulative, settings.pod_energy_threshold) + 1)
    if rank > settings.max_pod_rank:
        raise ValueError(
            "POD rank required by pod_energy_threshold exceeds max_pod_rank"
        )
    basis = _canonicalize_basis_signs(u_full[:, :rank])
    reconstructed = basis @ (basis.T @ snapshots)
    reconstruction_residual = float(numpy.linalg.norm(snapshots - reconstructed))
    reconstruction_relative = reconstruction_residual / float(numpy.linalg.norm(snapshots))

    feature_columns: list[numpy.ndarray] = []
    target_columns: list[numpy.ndarray] = []
    for anchor_index, case in enumerate(loaded):
        reduced_rows = (basis.T @ case["temperatures"].T).T
        states = numpy.vstack((numpy.zeros((1, rank)), reduced_rows))
        for window_index, powers in enumerate(case["powers"]):
            l2_features = numpy.zeros(len(loaded), dtype=float)
            l2_features[anchor_index] = powers[l2_name]
            feature_columns.append(numpy.concatenate((
                states[window_index],
                numpy.asarray([powers[name] for name in fixed_names], dtype=float),
                l2_features,
            )))
            target_columns.append(states[window_index + 1])
    features = numpy.stack(feature_columns, axis=1)
    targets = numpy.stack(target_columns, axis=1)
    gram = features @ features.T
    regularized_gram = gram + settings.ridge * numpy.eye(gram.shape[0])
    gram_condition = float(numpy.linalg.cond(regularized_gram))
    if not math.isfinite(gram_condition) or gram_condition > settings.max_condition_number:
        raise ValueError(
            "ridge Gram condition number exceeds max_condition_number"
        )
    coefficients = numpy.linalg.solve(
        regularized_gram, features @ targets.T
    ).T
    fitted = coefficients @ features
    fit_residual = float(numpy.linalg.norm(targets - fitted))
    target_norm = float(numpy.linalg.norm(targets))
    fit_relative = fit_residual / target_norm if target_norm else fit_residual

    a_discrete = coefficients[:, :rank]
    b_discrete = coefficients[:, rank:]
    input_count = b_discrete.shape[1]
    discrete_augmented = numpy.zeros((rank + input_count, rank + input_count))
    discrete_augmented[:rank, :rank] = a_discrete
    discrete_augmented[:rank, rank:] = b_discrete
    discrete_augmented[rank:, rank:] = numpy.eye(input_count)
    continuous_complex, logm_error = logm(discrete_augmented, disp=False)
    if not numpy.all(numpy.isfinite(continuous_complex)):
        raise ValueError("discrete-to-continuous matrix logarithm is not finite")
    continuous_complex = continuous_complex / dt
    imaginary_max = float(numpy.max(numpy.abs(numpy.imag(continuous_complex))))
    real_scale = max(1.0, float(numpy.max(numpy.abs(numpy.real(continuous_complex)))))
    if imaginary_max > _IMAGINARY_RELATIVE_TOLERANCE * real_scale:
        raise ValueError("matrix logarithm has material imaginary components")
    continuous_augmented = numpy.asarray(numpy.real(continuous_complex), dtype=float)
    a_continuous = continuous_augmented[:rank, :rank]
    eigenvalues = numpy.linalg.eigvals(a_continuous)
    maximum_real_pole = float(numpy.max(numpy.real(eigenvalues)))
    if maximum_real_pole > _STABILITY_TOLERANCE:
        raise ValueError("unstable continuous-time state matrix")
    b_continuous = continuous_augmented[:rank, rank:]
    b_fixed = b_continuous[:, :len(fixed_names)]
    b_l2 = {
        identifier: b_continuous[:, len(fixed_names) + index:index + len(fixed_names) + 1]
        for index, identifier in enumerate(identifiers)
    }
    model = StateSpaceModel(
        temperature_basis=basis,
        a_continuous=a_continuous,
        b_fixed=b_fixed,
        b_l2_anchors=b_l2,
        module_names=fixed_names,
    )
    _validated_model(model)
    report = {
        "schema_version": 1,
        "case_order": identifiers,
        "grid_unit_names": list(grid_names),
        "sample_interval_s": dt,
        "snapshot_count": int(snapshots.shape[1]),
        "transition_count": int(features.shape[1]),
        "temperature_trace_sha256": {
            case["id"]: case["temperature_sha256"] for case in loaded
        },
        "power_windows_sha256": {
            case["id"]: case["power_sha256"] for case in loaded
        },
        "pod": {
            "rank": rank,
            "energy_threshold": settings.pod_energy_threshold,
            "retained_energy": float(cumulative[rank - 1]),
            "singular_values": [float(value) for value in singular_values],
            "reconstruction_residual_frobenius_c": reconstruction_residual,
            "reconstruction_relative_frobenius": reconstruction_relative,
            "temperature_basis_sha256": "sha256:" + hashlib.sha256(
                numpy.ascontiguousarray(basis).tobytes()
            ).hexdigest(),
        },
        "identification": {
            "ridge": settings.ridge,
            "gram_condition_number": gram_condition,
            "max_condition_number": settings.max_condition_number,
            "residual_frobenius": fit_residual,
            "relative_residual_frobenius": fit_relative,
        },
        "continuous_conversion": {
            "logm_error_estimate": float(logm_error) / dt,
            "maximum_imaginary_component": imaginary_max,
            "imaginary_relative_tolerance": _IMAGINARY_RELATIVE_TOLERANCE,
            "stability_tolerance": _STABILITY_TOLERANCE,
            "maximum_real_pole": maximum_real_pole,
            "eigenvalues": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in eigenvalues
            ],
        },
    }
    return model, report


def discretize(model: StateSpaceModel, duration_s: float,
               b_l2: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return the exact zero-order-hold discretization using an augmented expm."""
    rank, _ = _validated_model(model)
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise ValueError("duration_s must be finite and positive")
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    l2 = _finite_real_array(b_l2, "b_l2", 2)
    if l2.shape != (rank, 1):
        raise ValueError(f"b_l2 must have shape ({rank}, 1)")
    inputs = numpy.concatenate((model.b_fixed, l2), axis=1)
    input_count = inputs.shape[1]
    augmented = numpy.zeros((rank + input_count, rank + input_count))
    augmented[:rank, :rank] = model.a_continuous
    augmented[:rank, rank:] = inputs
    discrete = expm(augmented * duration)
    if not numpy.all(numpy.isfinite(discrete)):
        raise ValueError("continuous-time discretization is not finite")
    return discrete[:rank, :rank], discrete[:rank, rank:]


def save_model(path: Path, model: StateSpaceModel, metadata: dict) -> None:
    """Persist model arrays and finite JSON metadata without pickle objects."""
    _validated_model(model)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    try:
        metadata_json = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be finite and JSON-serializable") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    anchors = sorted(model.b_l2_anchors)
    with destination.open("wb") as stream:
        numpy.savez_compressed(
            stream,
            temperature_basis=numpy.asarray(model.temperature_basis, dtype=float),
            a_continuous=numpy.asarray(model.a_continuous, dtype=float),
            b_fixed=numpy.asarray(model.b_fixed, dtype=float),
            b_l2_anchor_ids=numpy.asarray(anchors, dtype=str),
            b_l2_anchor_values=numpy.stack(
                [numpy.asarray(model.b_l2_anchors[name], dtype=float)[:, 0] for name in anchors]
            ),
            module_names=numpy.asarray(model.module_names, dtype=str),
            metadata_json=numpy.asarray(metadata_json),
        )


def load_model(path: Path) -> tuple[StateSpaceModel, dict]:
    """Load a model written by :func:`save_model`, rejecting malformed arrays."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    required = {
        "temperature_basis", "a_continuous", "b_fixed", "b_l2_anchor_ids",
        "b_l2_anchor_values", "module_names", "metadata_json",
    }
    try:
        with numpy.load(source, allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise ValueError("POD model archive is incomplete")
            anchor_id_array = archive["b_l2_anchor_ids"]
            if anchor_id_array.ndim != 1:
                raise ValueError("POD model anchor ids must be one-dimensional")
            anchor_ids = [str(value) for value in anchor_id_array.tolist()]
            if any(not name for name in anchor_ids) or len(set(anchor_ids)) != len(anchor_ids):
                raise ValueError("POD model anchor ids must be non-empty and unique")
            anchor_values = _finite_real_array(
                archive["b_l2_anchor_values"], "b_l2_anchor_values", 2
            )
            if anchor_values.shape[0] != len(anchor_ids):
                raise ValueError("POD model anchor arrays have inconsistent dimensions")
            metadata_raw = archive["metadata_json"]
            if metadata_raw.ndim != 0:
                raise ValueError("POD model metadata is malformed")
            metadata = json.loads(str(metadata_raw.item()))
            module_name_array = archive["module_names"]
            if module_name_array.ndim != 1:
                raise ValueError("POD model module_names must be one-dimensional")
            model = StateSpaceModel(
                temperature_basis=numpy.array(_finite_real_array(
                    archive["temperature_basis"], "temperature_basis", 2
                ), copy=True),
                a_continuous=numpy.array(_finite_real_array(
                    archive["a_continuous"], "a_continuous", 2
                ), copy=True),
                b_fixed=numpy.array(_finite_real_array(
                    archive["b_fixed"], "b_fixed", 2
                ), copy=True),
                b_l2_anchors={
                    name: anchor_values[index, :, numpy.newaxis]
                    for index, name in enumerate(anchor_ids)
                },
                module_names=tuple(str(value) for value in module_name_array.tolist()),
            )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("POD model archive is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("POD model metadata must be a dictionary")
    _validated_model(model)
    return model, metadata
