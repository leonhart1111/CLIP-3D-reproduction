#!/usr/bin/env python3
"""Linear steady-state thermal response and sensitivity primitives.

The first non-ML stage models a fixed HotSpot contract as

    T(P) = intercept + H @ P

where ``H`` is obtained from central finite differences.  The class is kept
independent of HotSpot process management so it can be tested with synthetic
linear systems and reused by the architecture search.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from workflow.common import read_json, sha256_file, write_json


def _vector(value: Iterable[float] | np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(list(value) if not isinstance(value, np.ndarray) else value,
                        dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains non-finite values")
    return result


def _matrix(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{label} must be a non-empty matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains non-finite values")
    return result


def softmax(values: Iterable[float] | np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Numerically stable softmax for the smooth peak-temperature metric."""
    values = _vector(values, "values")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    shifted = (values - np.max(values)) / tau
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("softmax normalization is non-positive")
    return weights / total


def soft_peak(values: Iterable[float] | np.ndarray, tau: float = 1.0) -> float:
    """Return a stable smooth maximum with the same units as ``values``."""
    values = _vector(values, "values")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    maximum = float(np.max(values))
    result = maximum + tau * math.log(
        float(np.sum(np.exp((values - maximum) / tau)))
    )
    if not math.isfinite(result):
        raise ValueError("soft peak is non-finite")
    return result


@dataclass(frozen=True)
class LinearThermalResponse:
    """Affine thermal response under one immutable physical contract."""

    source_names: tuple[str, ...]
    temperature_names: tuple[str, ...]
    baseline_power_w: np.ndarray
    baseline_temperature_c: np.ndarray
    response_k_per_w: np.ndarray
    intercept_c: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        power = _vector(self.baseline_power_w, "baseline_power_w")
        temperature = _vector(self.baseline_temperature_c,
                              "baseline_temperature_c")
        response = _matrix(self.response_k_per_w, "response_k_per_w")
        intercept = _vector(self.intercept_c, "intercept_c")
        if response.shape != (temperature.size, power.size):
            raise ValueError(
                "response shape must be (temperature_count, source_count)"
            )
        if intercept.size != temperature.size:
            raise ValueError("intercept length must match temperature_count")
        if len(self.source_names) != power.size:
            raise ValueError("source_names length must match source_count")
        if len(self.temperature_names) != temperature.size:
            raise ValueError(
                "temperature_names length must match temperature_count"
            )
        if len(set(self.source_names)) != len(self.source_names):
            raise ValueError("source_names must be unique")
        if len(set(self.temperature_names)) != len(self.temperature_names):
            raise ValueError("temperature_names must be unique")
        if not np.all(np.isfinite(power)) or np.any(power < -1.0e-12):
            raise ValueError("baseline_power_w must be finite and non-negative")
        reconstructed = intercept + response @ power
        if not np.allclose(reconstructed, temperature, rtol=1.0e-10,
                           atol=1.0e-8):
            raise ValueError(
                "intercept and response do not reconstruct baseline temperature"
            )
        object.__setattr__(self, "baseline_power_w", power.copy())
        object.__setattr__(self, "baseline_temperature_c", temperature.copy())
        object.__setattr__(self, "response_k_per_w", response.copy())
        object.__setattr__(self, "intercept_c", intercept.copy())

    @property
    def source_count(self) -> int:
        return int(self.baseline_power_w.size)

    @property
    def temperature_count(self) -> int:
        return int(self.baseline_temperature_c.size)

    def predict(self, power_w: Iterable[float] | np.ndarray) -> np.ndarray:
        power = _vector(power_w, "power_w")
        if power.size != self.source_count:
            raise ValueError("power vector has the wrong source count")
        if np.any(power < -1.0e-12):
            raise ValueError("power_w must be non-negative")
        result = self.intercept_c + self.response_k_per_w @ power
        if not np.all(np.isfinite(result)):
            raise ValueError("predicted temperature is non-finite")
        return result

    def predict_delta(self, delta_power_w: Iterable[float] | np.ndarray) -> np.ndarray:
        delta = _vector(delta_power_w, "delta_power_w")
        if delta.size != self.source_count:
            raise ValueError("delta power vector has the wrong source count")
        result = self.response_k_per_w @ delta
        if not np.all(np.isfinite(result)):
            raise ValueError("predicted temperature delta is non-finite")
        return result

    def sensitivity(self, power_w: Iterable[float] | np.ndarray | None = None,
                    tau: float = 1.0) -> np.ndarray:
        """Return d(T_soft)/dP in K/W for the supplied operating point."""
        power = self.baseline_power_w if power_w is None else _vector(power_w, "power_w")
        temperatures = self.predict(power)
        weights = softmax(temperatures, tau)
        result = weights @ self.response_k_per_w
        if not np.all(np.isfinite(result)):
            raise ValueError("thermal sensitivity is non-finite")
        return result

    def sensitivity_map(self, power_w: Iterable[float] | np.ndarray | None = None,
                        tau: float = 1.0) -> dict[str, np.ndarray]:
        """Return the full spatial sensitivity contract for one operating point.

        ``temperature_gradient`` is ``d(T_soft)/dT`` for every active HotSpot
        grid sample. ``response_k_per_w`` is ``dT/dP`` for every sample and
        source, and ``source_gradient_k_per_w`` is their chain-rule product
        ``d(T_soft)/dP``. Keeping all three arrays explicit prevents the
        source-level scalar used for pruning from being mistaken for a spatial
        sensitivity map.
        """
        power = self.baseline_power_w if power_w is None else _vector(power_w, "power_w")
        temperatures = self.predict(power)
        temperature_gradient = softmax(temperatures, tau)
        source_gradient = temperature_gradient @ self.response_k_per_w
        return {
            "temperature_gradient": temperature_gradient,
            "response_k_per_w": self.response_k_per_w.copy(),
            "source_gradient_k_per_w": source_gradient,
        }

    def score_power_action(self, power_w: Iterable[float] | np.ndarray,
                           tau: float = 1.0) -> dict[str, Any]:
        """Predict a candidate action relative to this response baseline."""
        candidate = _vector(power_w, "power_w")
        if candidate.size != self.source_count:
            raise ValueError("candidate power has the wrong source count")
        baseline_soft = soft_peak(self.baseline_temperature_c, tau)
        candidate_temperature = self.predict(candidate)
        candidate_soft = soft_peak(candidate_temperature, tau)
        delta_power = candidate - self.baseline_power_w
        sensitivity = self.sensitivity(self.baseline_power_w, tau)
        predicted_delta = float(sensitivity @ delta_power)
        return {
            "predicted_temperature_c": candidate_temperature.tolist(),
            "predicted_tmax_c": float(np.max(candidate_temperature)),
            "predicted_tsoft_c": candidate_soft,
            "baseline_tsoft_c": baseline_soft,
            "predicted_delta_tsoft_c": predicted_delta,
            "delta_power_w": delta_power.tolist(),
            "linearization_residual_c": candidate_soft - baseline_soft - predicted_delta,
        }

    def save(self, output_prefix: Path | str) -> tuple[Path, Path]:
        """Write ``.npz`` arrays plus a human-readable JSON manifest."""
        prefix = Path(output_prefix)
        if prefix.suffix:
            prefix = prefix.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        arrays_path = prefix.with_suffix(".npz")
        manifest_path = prefix.with_suffix(".json")
        np.savez_compressed(
            arrays_path,
            baseline_power_w=self.baseline_power_w,
            baseline_temperature_c=self.baseline_temperature_c,
            response_k_per_w=self.response_k_per_w,
            intercept_c=self.intercept_c,
        )
        manifest = {
            "schema_version": 1,
            "source_names": list(self.source_names),
            "temperature_names": list(self.temperature_names),
            "source_count": self.source_count,
            "temperature_count": self.temperature_count,
            "arrays_file": str(arrays_path.resolve()),
            "arrays_sha256": sha256_file(arrays_path),
            "metadata": self.metadata,
        }
        write_json(manifest_path, manifest)
        return arrays_path, manifest_path

    @classmethod
    def load(cls, manifest_path: Path | str) -> "LinearThermalResponse":
        manifest_path = Path(manifest_path)
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported linear response schema")
        arrays_path = Path(manifest["arrays_file"])
        if not arrays_path.is_absolute():
            arrays_path = manifest_path.parent / arrays_path
        if sha256_file(arrays_path) != manifest.get("arrays_sha256"):
            raise ValueError("linear response array hash mismatch")
        with np.load(arrays_path) as arrays:
            return cls(
                tuple(manifest["source_names"]),
                tuple(manifest["temperature_names"]),
                arrays["baseline_power_w"],
                arrays["baseline_temperature_c"],
                arrays["response_k_per_w"],
                arrays["intercept_c"],
                dict(manifest.get("metadata", {})),
            )


def fit_central_difference(
    source_names: Iterable[str],
    temperature_names: Iterable[str],
    baseline_power_w: Iterable[float] | np.ndarray,
    baseline_temperature_c: Iterable[float] | np.ndarray,
    perturbations: Iterable[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> LinearThermalResponse:
    """Fit an affine response from central-difference HotSpot probes.

    Each perturbation record must contain ``source_index``, ``delta_w``,
    ``positive_temperature_c`` and ``negative_temperature_c``.  Every source
    must be probed exactly once; this prevents silently creating a response
    matrix with missing columns.
    """
    source_names = tuple(str(name) for name in source_names)
    temperature_names = tuple(str(name) for name in temperature_names)
    power = _vector(baseline_power_w, "baseline_power_w")
    temperature = _vector(baseline_temperature_c, "baseline_temperature_c")
    if len(source_names) != power.size:
        raise ValueError("source_names length must match baseline power")
    if len(temperature_names) != temperature.size:
        raise ValueError("temperature_names length must match baseline temperature")
    response = np.zeros((temperature.size, power.size), dtype=np.float64)
    seen: set[int] = set()
    records = list(perturbations)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each perturbation must be an object")
        index = record.get("source_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("source_index must be an integer")
        if index < 0 or index >= power.size:
            raise ValueError("source_index is outside the source vector")
        if index in seen:
            raise ValueError(f"duplicate perturbation for source {index}")
        seen.add(index)
        delta = record.get("delta_w")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise ValueError("delta_w must be numeric")
        delta = float(delta)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta_w must be finite and positive")
        positive = _vector(record.get("positive_temperature_c", []),
                           "positive_temperature_c")
        negative = _vector(record.get("negative_temperature_c", []),
                           "negative_temperature_c")
        if positive.size != temperature.size or negative.size != temperature.size:
            raise ValueError("perturbation temperature length mismatch")
        response[:, index] = (positive - negative) / (2.0 * delta)
    missing = sorted(set(range(power.size)).difference(seen))
    if missing:
        raise ValueError(f"missing perturbation source indices: {missing}")
    intercept = temperature - response @ power
    result = LinearThermalResponse(
        source_names,
        temperature_names,
        power,
        temperature,
        response,
        intercept,
        {
            **(metadata or {}),
            "fit_method": "central-finite-difference",
            "perturbation_count": len(records),
        },
    )
    plus_residuals = []
    minus_residuals = []
    for record in records:
        index = int(record["source_index"])
        delta = float(record["delta_w"])
        plus_power = power.copy()
        minus_power = power.copy()
        plus_power[index] += delta
        minus_power[index] -= delta
        if minus_power[index] < -1.0e-12:
            raise ValueError(
                f"negative perturbed power for source {index}; reduce delta_w"
            )
        plus_residuals.append(
            np.max(np.abs(result.predict(plus_power)
                          - _vector(record["positive_temperature_c"], "positive")))
        )
        minus_residuals.append(
            np.max(np.abs(result.predict(minus_power)
                          - _vector(record["negative_temperature_c"], "negative")))
        )
    result.metadata["positive_reconstruction_max_error_c"] = float(max(plus_residuals))
    result.metadata["negative_reconstruction_max_error_c"] = float(max(minus_residuals))
    return result


def parse_ptrace(path: Path | str) -> tuple[list[str], np.ndarray]:
    """Read CLIP's two-line power trace into deterministic source vectors."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(lines) != 2:
        raise ValueError("power trace must contain exactly two non-empty lines")
    names = lines[0].split()
    values = _vector([float(value) for value in lines[1].split()], "power trace")
    if len(names) != values.size:
        raise ValueError("power trace names and values have different lengths")
    if len(set(names)) != len(names):
        raise ValueError("power trace names must be unique")
    if np.any(values < -1.0e-12):
        raise ValueError("power trace contains negative power")
    return names, values
