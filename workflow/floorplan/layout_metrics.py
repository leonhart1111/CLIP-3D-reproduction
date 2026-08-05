#!/usr/bin/env python3
"""Derive layout-dependent TSV and on-die wire delays for gem5 R2."""

from __future__ import annotations

import math


WIRE_R_OHM_PER_MM = 50.0
WIRE_C_F_PER_MM = 200e-15


def smooth_abs(value: float, epsilon: float = 1e-6) -> float:
    return math.sqrt(value * value + epsilon * epsilon)


def core_centers(modules: list[dict]) -> list[tuple[float, float, int]]:
    """Return area-weighted (x, y, tier) centers for all represented cores."""
    core_ids = sorted({int(module["core"]) for module in modules
                       if module.get("core") is not None})
    if not core_ids:
        raise ValueError("layout contains no core modules")
    centers = []
    for core in core_ids:
        group = [module for module in modules if module.get("core") == core]
        area = sum(float(module["area_mm2"]) for module in group)
        if area <= 0:
            raise ValueError(f"core {core} has no positive module area")
        tiers = {int(module["tier"]) for module in group}
        if len(tiers) != 1:
            raise ValueError(f"core {core} spans multiple tiers: {sorted(tiers)}")
        centers.append((
            sum((module["x_mm"] + module["width_mm"] / 2.0) * module["area_mm2"]
                for module in group) / area,
            sum((module["y_mm"] + module["height_mm"] / 2.0) * module["area_mm2"]
                for module in group) / area,
            tiers.pop(),
        ))
    return centers


def l2_module(modules: list[dict]) -> dict:
    matches = [module for module in modules if module.get("kind") == "l2"]
    if len(matches) != 1:
        raise ValueError(f"layout must contain exactly one L2, found {len(matches)}")
    return matches[0]


def mean_wire_cycles(modules: list[dict], f0_ghz: float,
                     wire_r_ohm_per_mm: float = WIRE_R_OHM_PER_MM,
                     wire_c_f_per_mm: float = WIRE_C_F_PER_MM) -> tuple[float, list[dict]]:
    """Evaluate the paper's 0.69*R*C*L^2 delay at mean core-to-L2 distance."""
    l2 = l2_module(modules)
    lx = l2["x_mm"] + l2["width_mm"] / 2.0
    ly = l2["y_mm"] + l2["height_mm"] / 2.0
    cycle_seconds = 1.0 / (f0_ghz * 1e9)
    per_core = []
    for core, (x, y, tier) in enumerate(core_centers(modules)):
        length_mm = smooth_abs(lx - x) + smooth_abs(ly - y)
        delay_seconds = 0.69 * wire_r_ohm_per_mm * wire_c_f_per_mm * length_mm * length_mm
        per_core.append({
            "core": core,
            "core_tier": tier,
            "l2_tier": int(l2["tier"]),
            "manhattan_length_mm": length_mm,
            "delay_seconds": delay_seconds,
            "delay_cycles": delay_seconds / cycle_seconds,
        })
    return sum(item["delay_cycles"] for item in per_core) / len(per_core), per_core


def communication_weights_from_model(model: dict,
                                     required: bool = False) -> dict[int, float] | None:
    """Return recorded per-core traffic weights without inventing a fallback."""
    profile = model.get("communication_profile")
    if not isinstance(profile, dict) or profile.get("status") != "available":
        if required:
            diagnostics = profile.get("diagnostics", []) if isinstance(profile, dict) else []
            detail = "; ".join(str(item) for item in diagnostics)
            suffix = f": {detail}" if detail else ""
            raise ValueError(f"communication profile unavailable{suffix}")
        return None
    per_core = profile.get("per_core")
    if not isinstance(per_core, dict):
        raise ValueError("available communication profile has no per-core records")
    weights: dict[int, float] = {}
    for core, record in per_core.items():
        if not isinstance(record, dict) or "normalized_weight" not in record:
            raise ValueError(f"communication profile core {core} has no normalized weight")
        try:
            core_id = int(core)
            weight = float(record["normalized_weight"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"communication profile core {core} has an invalid weight") from error
        if core_id in weights:
            raise ValueError(f"communication profile repeats core {core_id}")
        weights[core_id] = weight
    return weights


def aggregate_wire_cycles(per_core: list[dict], aggregation: str = "mean",
                          communication_weights: dict[int | str, float] | None = None
                          ) -> float:
    """Aggregate represented core-to-L2 delays with one validated policy."""
    if not per_core:
        raise ValueError("wire aggregation requires at least one represented core")
    delays: dict[int, float] = {}
    for item in per_core:
        core = int(item["core"])
        delay = float(item["delay_cycles"])
        if core in delays:
            raise ValueError(f"wire delays repeat represented core {core}")
        if not math.isfinite(delay) or delay < 0:
            raise ValueError(f"core {core} wire delay must be finite and non-negative")
        delays[core] = delay
    if aggregation == "mean":
        return sum(delays.values()) / len(delays)
    if aggregation == "maximum":
        return max(delays.values())
    if aggregation != "traffic-weighted":
        raise ValueError(f"unknown wire-cycle aggregation: {aggregation}")
    if communication_weights is None:
        raise ValueError("traffic-weighted wire aggregation requires communication weights")

    weights: dict[int, float] = {}
    for key, value in communication_weights.items():
        try:
            core = int(key)
            weight = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("communication weights must use integer core IDs and numbers") from error
        if core in weights:
            raise ValueError(f"communication weights repeat core {core}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("communication weights must be finite and non-negative")
        weights[core] = weight
    if set(weights) != set(delays):
        raise ValueError(
            "communication weight keys must exactly match represented core IDs: "
            f"weights={sorted(weights)}, represented={sorted(delays)}"
        )
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("communication weights must sum to one")
    return sum(weights[core] * delays[core] for core in sorted(delays))


def round_wire_cycles(value: float, policy: str = "nearest") -> int:
    if value < 0:
        raise ValueError("wire delay cannot be negative")
    if policy == "nearest":
        return int(math.floor(value + 0.5))
    if policy == "ceil":
        return int(math.ceil(value))
    if policy == "floor":
        return int(math.floor(value))
    raise ValueError(f"unknown wire-cycle rounding policy: {policy}")


def derive_layout_delays(layout: dict, f0_ghz: float = 2.0,
                         wire_rounding: str = "nearest",
                         communication_weights: dict[int | str, float] | None = None
                         ) -> dict:
    modules = layout["modules"]
    l2 = l2_module(modules)
    centers = core_centers(modules)
    hops = [abs(tier - int(l2["tier"])) for _, _, tier in centers]
    if len(set(hops)) != 1:
        raise ValueError(f"one shared xbar latency cannot represent TSV hops {hops}")
    mean_cycles, per_core = mean_wire_cycles(modules, f0_ghz)
    minimum_cycles = min(item["delay_cycles"] for item in per_core)
    maximum_cycles = max(item["delay_cycles"] for item in per_core)
    result = {
        "tsv_hops": hops[0],
        "wire_cycles_unrounded": mean_cycles,
        "wire_cycles": round_wire_cycles(mean_cycles, wire_rounding),
        "wire_cycle_aggregation": "mean across represented core-to-L2 paths",
        "minimum_wire_cycles_unrounded": minimum_cycles,
        "maximum_wire_cycles_unrounded": maximum_cycles,
        "maximum_wire_cycles": round_wire_cycles(maximum_cycles, wire_rounding),
        "wire_rounding": wire_rounding,
        "per_core": per_core,
        "wire_model": {
            "equation": "0.69*R*C*L^2",
            "r_ohm_per_mm": WIRE_R_OHM_PER_MM,
            "c_f_per_mm": WIRE_C_F_PER_MM,
            "distance": "smoothed Manhattan core-cluster center to L2 center",
        },
    }
    if communication_weights is not None:
        weighted = aggregate_wire_cycles(
            per_core, "traffic-weighted", communication_weights
        )
        normalized = {int(key): float(value)
                      for key, value in communication_weights.items()}
        for item in per_core:
            weight = normalized[int(item["core"])]
            item["communication_weight"] = weight
            item["weighted_delay_cycles_contribution"] = (
                weight * item["delay_cycles"]
            )
        result.update({
            "traffic_weighted_wire_cycles_unrounded": weighted,
            "traffic_weighted_wire_cycles": round_wire_cycles(
                weighted, wire_rounding
            ),
            "traffic_weighted_wire_cycle_aggregation": (
                "shared-L2 demand-access weighted represented core-to-L2 paths"
            ),
        })
    return result
