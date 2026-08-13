#!/usr/bin/env python3
"""Identify an unscaled strict-P1 thermal proxy from local HotSpot evidence."""

from __future__ import annotations

import math
from pathlib import Path

from workflow.common import read_json
from workflow.floorplan.generate_hotspot_inputs import baseline_layout, check_geometry
from workflow.floorplan.optimize_layout import collision_area


PLACEMENT_LABELS = (
    "center",
    "corner_ll", "corner_lr", "corner_ul", "corner_ur",
    "edge_bottom", "edge_top", "edge_left", "edge_right",
    "near_core0", "near_core1", "near_core2", "near_core3",
)


def _normalized_lattice(steps: int = 41) -> list[tuple[float, float]]:
    values = [index / (steps - 1) for index in range(steps)]
    return [(x, y) for y in values for x in values]


def placement_design(model_path: Path, utilization: float,
                     requested: int = 13) -> list[dict]:
    """Return deterministic distinct legal L2 placements for spatial fitting."""
    if requested != len(PLACEMENT_LABELS):
        raise ValueError(f"the formal placement design requires {len(PLACEMENT_LABELS)} cases")
    base = baseline_layout(read_json(model_path), utilization)
    l2 = next(module for module in base["modules"] if module["kind"] == "l2")
    fixed = [dict(module) for module in base["modules"] if module["kind"] != "l2"]
    side = float(base["die_width_mm"])
    upper_x = side - float(l2["width_mm"])
    upper_y = side - float(l2["height_mm"])
    if upper_x <= 0 or upper_y <= 0:
        raise RuntimeError("geometry cannot provide at least 11 distinct L2 positions")

    targets: list[tuple[str, float, float]] = [
        ("center", 0.5, 0.5),
        ("corner_ll", 0.0, 0.0), ("corner_lr", 1.0, 0.0),
        ("corner_ul", 0.0, 1.0), ("corner_ur", 1.0, 1.0),
        ("edge_bottom", 0.5, 0.0), ("edge_top", 0.5, 1.0),
        ("edge_left", 0.0, 0.5), ("edge_right", 1.0, 0.5),
    ]
    cores = sorted(
        (module for module in fixed if module.get("kind") == "core"),
        key=lambda module: int(module.get("core", -1)),
    )
    if len(cores) != 4:
        raise ValueError("formal placement design requires exactly four core modules")
    for core_index, core in enumerate(cores):
        center_x = float(core["x_mm"]) + float(core["width_mm"]) / 2.0
        center_y = float(core["y_mm"]) + float(core["height_mm"]) / 2.0
        targets.append((
            f"near_core{core_index}",
            min(max((center_x - float(l2["width_mm"]) / 2.0) / upper_x, 0.0), 1.0),
            min(max((center_y - float(l2["height_mm"]) / 2.0) / upper_y, 0.0), 1.0),
        ))

    lattice = _normalized_lattice()
    selected: list[dict] = []
    used: list[tuple[float, float]] = []
    for label, target_x, target_y in targets:
        candidates = sorted(
            lattice,
            key=lambda point: (
                (point[0] - target_x) ** 2 + (point[1] - target_y) ** 2,
                point[1], point[0],
            ),
        )
        chosen = None
        for fx, fy in candidates:
            x_mm, y_mm = fx * upper_x, fy * upper_y
            if any(math.isclose(x_mm, x, abs_tol=1e-9)
                   and math.isclose(y_mm, y, abs_tol=1e-9) for x, y in used):
                continue
            candidate = dict(l2, x_mm=x_mm, y_mm=y_mm)
            if collision_area(candidate, fixed) > 1e-9:
                continue
            modules = fixed + [candidate]
            check_geometry(modules, side)
            layout = dict(base, modules=modules,
                          policy="unscaled alpha-Lc deterministic spatial design")
            chosen = {
                "label": label, "layout": layout,
                "x_mm": x_mm, "y_mm": y_mm, "fx": fx, "fy": fy,
            }
            break
        if chosen is None:
            break
        selected.append(chosen)
        used.append((chosen["x_mm"], chosen["y_mm"]))
    if len(selected) < 11:
        raise RuntimeError(
            f"geometry provides {len(selected)} placements; at least 11 distinct are required"
        )
    if len(selected) != requested:
        raise RuntimeError(
            f"geometry provides only {len(selected)} of {requested} requested placements"
        )
    return selected
