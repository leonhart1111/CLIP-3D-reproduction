"""Shared deterministic grid search over integer R2 wire-cycle partitions."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Callable, Sequence

from workflow.floorplan.generate_hotspot_inputs import check_geometry


CandidateEvaluator = Callable[[dict, str, str], dict | None]


def validate_partition_controls(
    grid_steps: int, include_fixed_baseline: bool,
) -> None:
    """Validate the reviewed discrete search controls before evaluation."""
    if (
        isinstance(grid_steps, bool)
        or not isinstance(grid_steps, int)
        or grid_steps < 3
        or grid_steps % 2 == 0
    ):
        raise ValueError("partition grid steps must be an odd integer >= 3")
    if include_fixed_baseline is not True:
        raise ValueError("discrete partition requires the fixed-bin baseline")


def place_l2(
    base_layout: dict, l2_name: str, tier: int, x_mm: float, y_mm: float,
) -> dict:
    """Return a copied legal layout with exactly one named L2 repositioned."""
    if not isinstance(base_layout, dict):
        raise ValueError("base layout must be a dictionary")
    modules = base_layout.get("modules")
    if not isinstance(modules, list):
        raise ValueError("base layout must contain a module list")
    if isinstance(tier, bool) or not isinstance(tier, int) or tier not in (0, 1):
        raise ValueError("L2 tier must be 0 or 1")
    coordinates = (float(x_mm), float(y_mm))
    if any(not math.isfinite(value) for value in coordinates):
        raise ValueError("L2 coordinates must be finite")

    copied = []
    matches = 0
    for module in modules:
        if module.get("kind") == "l2" and module.get("name") == l2_name:
            copied.append(
                dict(module, tier=tier, x_mm=coordinates[0], y_mm=coordinates[1])
            )
            matches += 1
        else:
            copied.append(dict(module))
    if matches != 1:
        raise ValueError("base layout must contain exactly one named L2")

    die_width = float(base_layout.get("die_width_mm"))
    die_height = float(base_layout.get("die_height_mm"))
    if (
        not math.isfinite(die_width)
        or not math.isfinite(die_height)
        or die_width <= 0.0
        or die_height <= 0.0
        or not math.isclose(die_width, die_height, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("discrete partition requires a positive square die")
    check_geometry(copied, die_width)
    return {**base_layout, "modules": copied}


def candidate_key(candidate: dict) -> tuple[float, float, int, float, float]:
    """Return the shared deterministic objective and geometry tie-break key."""
    required = (
        "objective_loss",
        "continuous_selected_wire_cycles",
        "r2_wire_cycles",
        "tier",
        "x_mm",
        "y_mm",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ValueError(f"discrete candidate lacks: {', '.join(missing)}")
    loss = candidate["objective_loss"]
    continuous = candidate["continuous_selected_wire_cycles"]
    cycle = candidate["r2_wire_cycles"]
    tier = candidate["tier"]
    if (
        isinstance(loss, bool)
        or not isinstance(loss, Real)
        or not math.isfinite(float(loss))
        or isinstance(continuous, bool)
        or not isinstance(continuous, Real)
        or not math.isfinite(float(continuous))
        or isinstance(cycle, bool)
        or not isinstance(cycle, Integral)
        or int(cycle) < 0
        or isinstance(tier, bool)
        or not isinstance(tier, Integral)
    ):
        raise ValueError("discrete candidate objective and cycles must be finite")
    x_mm = float(candidate["x_mm"])
    y_mm = float(candidate["y_mm"])
    if not math.isfinite(x_mm) or not math.isfinite(y_mm):
        raise ValueError("discrete candidate coordinates must be finite")
    return (
        float(loss),
        float(continuous),
        int(tier),
        y_mm,
        x_mm,
    )


def _evaluated(
    evaluator: CandidateEvaluator, layout: dict, origin: str, start: str,
) -> dict | None:
    candidate = evaluator(layout, origin, start)
    if candidate is None:
        return None
    if not isinstance(candidate, dict):
        raise ValueError("discrete candidate evaluator must return a dictionary or None")
    normalized = dict(candidate)
    normalized.setdefault("origin", origin)
    normalized.setdefault("start", start)
    candidate_key(normalized)
    return normalized


def search_discrete_partitions(
    base_layout: dict,
    l2_name: str,
    allowed_tiers: Sequence[int],
    grid_steps: int,
    include_fixed_baseline: bool,
    evaluate: CandidateEvaluator,
) -> dict:
    """Enumerate legal layouts and select one best candidate per R2 cycle."""
    validate_partition_controls(grid_steps, include_fixed_baseline)
    tiers = tuple(dict.fromkeys(allowed_tiers))
    if not tiers or any(
        isinstance(tier, bool) or not isinstance(tier, int) or tier not in (0, 1)
        for tier in tiers
    ):
        raise ValueError("allowed L2 tiers must contain tier 0 and/or tier 1")
    modules = base_layout.get("modules")
    if not isinstance(modules, list):
        raise ValueError("base layout must contain a module list")
    matches = [
        module for module in modules
        if module.get("kind") == "l2" and module.get("name") == l2_name
    ]
    if len(matches) != 1:
        raise ValueError("base layout must contain exactly one named L2")
    original = matches[0]
    die_width = float(base_layout.get("die_width_mm"))
    die_height = float(base_layout.get("die_height_mm"))
    upper_x = die_width - float(original["width_mm"])
    upper_y = die_height - float(original["height_mm"])
    if min(upper_x, upper_y) < 0.0:
        raise ValueError("L2 geometry does not fit inside the die")

    fixed_layout = place_l2(
        base_layout,
        l2_name,
        int(original["tier"]),
        float(original["x_mm"]),
        float(original["y_mm"]),
    )
    fixed = _evaluated(evaluate, fixed_layout, "fixed-bin", "FIXED")
    if fixed is None:
        raise RuntimeError("fixed-bin baseline is not selectable")

    legal_grid = []
    geometry_rejections = []
    evaluator_rejections = []
    partitions: dict[int, dict] = {}
    for tier in tiers:
        for y_index in range(grid_steps):
            y_mm = upper_y * y_index / (grid_steps - 1)
            for x_index in range(grid_steps):
                x_mm = upper_x * x_index / (grid_steps - 1)
                start = f"GRID-{y_index}-{x_index}"
                point = {
                    "origin": "partition-grid",
                    "start": start,
                    "tier": tier,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                }
                try:
                    layout = place_l2(base_layout, l2_name, tier, x_mm, y_mm)
                except ValueError as error:
                    geometry_rejections.append({**point, "reason": str(error)})
                    continue
                candidate = _evaluated(
                    evaluate, layout, "partition-grid", start
                )
                if candidate is None:
                    evaluator_rejections.append(point)
                    continue
                legal_grid.append(candidate)
                cycle = int(candidate["r2_wire_cycles"])
                incumbent = partitions.get(cycle)
                if incumbent is None or candidate_key(candidate) < candidate_key(
                    incumbent
                ):
                    partitions[cycle] = candidate

    if not legal_grid:
        raise RuntimeError("discrete partition found no selectable legal grid placement")
    partition_records = [partitions[cycle] for cycle in sorted(partitions)]
    candidates = [fixed, *partition_records]
    selected = min(candidates, key=candidate_key)
    return {
        "grid_steps": grid_steps,
        "total_grid_candidates": len(tiers) * grid_steps * grid_steps,
        "legal_grid_candidates": len(legal_grid),
        "legal_grid_candidates_detail": legal_grid,
        "rejected_grid_candidates": (
            len(geometry_rejections) + len(evaluator_rejections)
        ),
        "geometry_rejections": geometry_rejections,
        "evaluator_rejections": evaluator_rejections,
        "fixed_baseline_included": True,
        "fixed_baseline": fixed,
        "partitions": partition_records,
        "candidates": candidates,
        "selected": selected,
    }
