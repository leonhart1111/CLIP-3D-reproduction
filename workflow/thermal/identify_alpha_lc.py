#!/usr/bin/env python3
"""Identify an unscaled strict-P1 thermal proxy from local HotSpot evidence."""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import (
    baseline_layout,
    check_geometry,
    materialize,
)
from workflow.floorplan.optimize_layout import collision_area
from workflow.thermal.run_hotspot import DEFAULT_HOTSPOT, run_hotspot


PLACEMENT_LABELS = (
    "center",
    "corner_ll", "corner_lr", "corner_ul", "corner_ur",
    "edge_bottom", "edge_top", "edge_left", "edge_right",
    "near_core0", "near_core1", "near_core2", "near_core3",
)


def case_identity(model: dict, layout: dict, physical: dict,
                  stimulus: dict) -> str:
    """Hash every value that can change a HotSpot calibration result."""
    payload = {
        "schema_version": 1,
        "model": model,
        "layout": layout,
        "physical": physical,
        "stimulus": stimulus,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_grid_convergence(records: list[dict],
                              grids: tuple[int, ...] = (32, 64, 128),
                              tolerance_c: float = 0.10) -> dict:
    """Select the smallest grid matching the finest grid in Tmax and rank."""
    if not grids or sorted(set(grids)) != list(grids):
        raise ValueError("grids must be unique and strictly increasing")
    reference_grid = grids[-1]
    by_grid: dict[int, dict[str, float]] = {grid: {} for grid in grids}
    for record in records:
        grid = int(record["grid_size"])
        if grid not in by_grid:
            raise ValueError(f"unexpected grid size: {grid}")
        label = str(record["label"])
        if label in by_grid[grid]:
            raise ValueError(f"duplicate grid/layout record: {grid}/{label}")
        value = float(record["tmax_c"])
        if not math.isfinite(value):
            raise ValueError("grid convergence temperatures must be finite")
        by_grid[grid][label] = value
    reference_labels = set(by_grid[reference_grid])
    if not reference_labels or any(set(values) != reference_labels
                                   for values in by_grid.values()):
        raise ValueError("every grid must contain the same layout labels")

    reference_order = sorted(
        reference_labels,
        key=lambda label: (by_grid[reference_grid][label], label),
    )
    reports = {}
    selected = reference_grid
    for grid in grids:
        differences = {
            label: abs(by_grid[grid][label] - by_grid[reference_grid][label])
            for label in sorted(reference_labels)
        }
        order = sorted(
            reference_labels, key=lambda label: (by_grid[grid][label], label),
        )
        order_matches = order == reference_order
        accepted = max(differences.values()) <= tolerance_c and order_matches
        reports[str(grid)] = {
            "max_abs_difference_c": max(differences.values()),
            "differences_c": differences,
            "rank_order": order,
            "rank_order_matches_reference": order_matches,
            "accepted": accepted,
        }
        if accepted and selected == reference_grid:
            selected = grid
    return {
        "schema_version": 1,
        "reference_grid_size": reference_grid,
        "tolerance_c": tolerance_c,
        "selected_grid_size": selected,
        "accepted": reports[str(reference_grid)]["accepted"],
        "grids": reports,
    }


def run_hotspot_case(model_path: Path, layout: dict, physical: dict,
                     stimulus: dict, cases_root: Path,
                     hotspot: Path = DEFAULT_HOTSPOT) -> dict:
    """Materialize and run one immutable, resumable HotSpot evidence case."""
    model_path = Path(model_path).resolve()
    model = read_json(model_path)
    identity = case_identity(model, layout, physical, stimulus)
    case_dir = Path(cases_root).resolve() / identity
    manifest_path = case_dir / "case_manifest.json"
    result_path = case_dir / "thermal_result.json"
    expected = {
        "schema_version": 1,
        "case_identity": identity,
        "model": str(model_path),
        "physical": physical,
        "stimulus": stimulus,
    }
    if manifest_path.is_file() and result_path.is_file():
        if read_json(manifest_path) == expected:
            return {
                **read_json(result_path), "case_identity": identity,
                "case_dir": str(case_dir), "reused": True,
            }
    case_dir.mkdir(parents=True, exist_ok=True)
    layout_path = case_dir / "candidate_layout.json"
    write_json(layout_path, layout)
    stack = physical.get("thermal_stack") or {}
    materialize(
        model_path, case_dir, int(physical["grid_size"]),
        float(physical["utilization"]), float(physical["ambient_c"]),
        float(physical["r_convec_k_per_w"]), layout_path, stack,
    )
    write_json(manifest_path, expected)
    result = run_hotspot(case_dir, hotspot)
    return {
        **result, "case_identity": identity,
        "case_dir": str(case_dir), "reused": False,
    }


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
