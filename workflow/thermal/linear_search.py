#!/usr/bin/env python3
"""Non-ML candidate scoring and Pareto pruning for LogicFolding DSE."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from workflow.thermal.linear_response import LinearThermalResponse


def evaluate_candidates(
    response: LinearThermalResponse,
    candidates: Iterable[dict[str, Any]],
    thermal_limit_c: float,
    tau: float = 1.0,
) -> list[dict[str, Any]]:
    """Add first-order thermal estimates to architecture/floorplan candidates.

    Candidate records must contain ``id``, ``power_w`` and numeric performance,
    energy, area, and communication fields.  The function never runs HotSpot;
    it is the cheap ranking/pruning stage.
    """
    if not math.isfinite(thermal_limit_c):
        raise ValueError("thermal_limit_c must be finite")
    result = []
    baseline = response.baseline_power_w
    sensitivity = response.sensitivity(baseline, tau)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be an object")
        identifier = candidate.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("each candidate needs a non-empty id")
        power = np.asarray(candidate.get("power_w", []), dtype=np.float64)
        if power.ndim != 1 or power.size != response.source_count:
            raise ValueError(f"candidate {identifier} has the wrong power vector")
        if not np.all(np.isfinite(power)) or np.any(power < -1.0e-12):
            raise ValueError(f"candidate {identifier} has invalid power")
        values = {}
        for field in ("performance", "energy", "area", "communication"):
            value = candidate.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"candidate {identifier}.{field} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"candidate {identifier}.{field} must be finite")
            values[field] = float(value)
        estimate = response.score_power_action(power, tau)
        enriched = {
            **candidate,
            **values,
            "power_w": power.tolist(),
            "thermal": {
                **estimate,
                "thermal_limit_c": float(thermal_limit_c),
                "predicted_thermal_feasible": (
                    estimate["predicted_tmax_c"] <= thermal_limit_c
                ),
                "sensitivity_k_per_w": sensitivity.tolist(),
            },
        }
        result.append(enriched)
    return result


def dominates(left: dict[str, Any], right: dict[str, Any],
              maximize: tuple[str, ...] = ("performance",),
              minimize: tuple[str, ...] = (
                  "energy", "area", "communication", "predicted_tsoft_c"
              )) -> bool:
    """Return whether left Pareto-dominates right."""
    def value(record: dict[str, Any], field: str) -> float:
        if field == "predicted_tsoft_c":
            return float(record["thermal"][field])
        return float(record[field])

    no_worse = all(value(left, field) >= value(right, field) for field in maximize)
    no_worse &= all(value(left, field) <= value(right, field) for field in minimize)
    strictly_better = any(value(left, field) > value(right, field) for field in maximize)
    strictly_better |= any(value(left, field) < value(right, field) for field in minimize)
    return no_worse and strictly_better


def pareto_front(records: Iterable[dict[str, Any]],
                 maximize: tuple[str, ...] = ("performance",),
                 minimize: tuple[str, ...] = (
                     "energy", "area", "communication", "predicted_tsoft_c"
                 )) -> list[dict[str, Any]]:
    """Return feasible/non-dominated records in deterministic id order."""
    records = list(records)
    if not records:
        return []
    feasible = [
        record for record in records
        if record.get("thermal", {}).get("predicted_thermal_feasible") is True
    ]
    pool = feasible if feasible else records
    front = [
        record for record in pool
        if not any(
            other is not record and dominates(other, record, maximize, minimize)
            for other in pool
        )
    ]
    return sorted(front, key=lambda record: str(record["id"]))


def rank_for_pruning(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic cheap ranking: feasible first, then thermal headroom/perf."""
    records = list(records)
    return sorted(
        records,
        key=lambda record: (
            not record["thermal"]["predicted_thermal_feasible"],
            record["thermal"]["predicted_tsoft_c"],
            -record["performance"],
            record["energy"],
            str(record["id"]),
        ),
    )
