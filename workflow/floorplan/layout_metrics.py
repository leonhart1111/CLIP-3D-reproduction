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
                         wire_rounding: str = "nearest") -> dict:
    modules = layout["modules"]
    l2 = l2_module(modules)
    centers = core_centers(modules)
    hops = [abs(tier - int(l2["tier"])) for _, _, tier in centers]
    if len(set(hops)) != 1:
        raise ValueError(f"one shared xbar latency cannot represent TSV hops {hops}")
    mean_cycles, per_core = mean_wire_cycles(modules, f0_ghz)
    minimum_cycles = min(item["delay_cycles"] for item in per_core)
    maximum_cycles = max(item["delay_cycles"] for item in per_core)
    return {
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
