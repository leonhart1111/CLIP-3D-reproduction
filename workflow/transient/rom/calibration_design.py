"""Deterministic legal layouts and PRBS excitations for transient-ROM fitting."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import random
from typing import Iterable

from workflow.floorplan.generate_hotspot_inputs import baseline_layout, check_geometry
from workflow.transient.rom.contracts import ROMSettings


_TOLERANCE_MM = 1e-9
_BISECTION_STEPS = 64
_SEARCH_GRID = 64


def build_design(model: dict, allowed_l2_tiers: list[int], settings: ROMSettings) -> dict:
    """Build the fixed eight-anchor, two-holdout calibration design.

    The supplied tier list is copied without sorting or broadening it: it is
    part of the scientific provenance of a ROM package.
    """
    if not isinstance(settings, ROMSettings):
        raise ValueError("settings must be ROMSettings")
    if not isinstance(allowed_l2_tiers, list) or not allowed_l2_tiers:
        raise ValueError("allowed_l2_tiers must be a non-empty list")
    if any(isinstance(tier, bool) or tier not in (0, 1) for tier in allowed_l2_tiers):
        raise ValueError("allowed_l2_tiers must contain only physical tiers 0 and 1")
    if len(set(allowed_l2_tiers)) != len(allowed_l2_tiers):
        raise ValueError("allowed_l2_tiers must not contain duplicates")
    if len(allowed_l2_tiers) not in (1, 2):
        raise ValueError("ROM calibration supports one tier or tiers [0, 1]")

    base_layout = baseline_layout(model)
    l2 = _exactly_one_l2(base_layout)
    allowed = list(allowed_l2_tiers)
    training: list[dict] = []
    holdout: list[dict] = []

    if len(allowed) == 2:
        for tier in allowed:
            anchors = _rectangle_anchors(base_layout, l2, tier, training)
            training.extend(anchors)
            holdout.append(_interior_holdout(
                base_layout, l2, tier, anchors, (0.37, 0.61), training + holdout,
                f"h{len(holdout)}",
            ))
    else:
        tier = allowed[0]
        training.extend(_single_tier_anchors(base_layout, l2, tier, training))
        anchors = list(training)
        for fractions in ((0.37, 0.61), (0.63, 0.39)):
            holdout.append(_interior_holdout(
                base_layout, l2, tier, anchors, fractions, training + holdout,
                f"h{len(holdout)}",
            ))

    if len(training) != 8 or len(holdout) != 2:
        raise ValueError("ROM calibration requires exactly eight anchors and two holdouts")
    _reject_duplicates(training + holdout)
    domains = {
        str(tier): _domain_for_tier(training, tier, two_tier=len(allowed) == 2)
        for tier in allowed
    }
    return {
        "base_layout": base_layout,
        "l2_name": l2["name"],
        "allowed_l2_tiers": allowed,
        "training": training,
        "holdout": holdout,
        "domains": domains,
        "prbs": make_prbs_input(
            [module["name"] for module in base_layout["modules"]], l2["name"], settings
        ),
    }


def layout_for_point(base_layout: dict, point: dict) -> dict:
    """Return a copied layout with its sole L2 moved to ``point`` and checked."""
    if not isinstance(point, dict):
        raise ValueError("point must be a dictionary")
    try:
        tier = point["tier"]
        x_mm = float(point["x_mm"])
        y_mm = float(point["y_mm"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("point must contain finite tier, x_mm, and y_mm") from error
    if isinstance(tier, bool) or tier not in (0, 1):
        raise ValueError("point tier must be 0 or 1")
    if not math.isfinite(x_mm) or not math.isfinite(y_mm):
        raise ValueError("point coordinates must be finite")

    layout = deepcopy(base_layout)
    l2 = _exactly_one_l2(layout)
    l2["tier"] = tier
    l2["x_mm"] = x_mm
    l2["y_mm"] = y_mm
    check_geometry(layout["modules"], layout["die_width_mm"])
    return layout


def make_prbs_input(module_names: list[str], l2_name: str, settings: ROMSettings) -> dict:
    """Generate independent, bounded, positive module-wise PRBS multipliers."""
    if not isinstance(settings, ROMSettings):
        raise ValueError("settings must be ROMSettings")
    if not isinstance(module_names, list) or not module_names:
        raise ValueError("module_names must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in module_names):
        raise ValueError("module_names must contain non-empty strings")
    if len(set(module_names)) != len(module_names):
        raise ValueError("module_names must not contain duplicates")
    if l2_name not in module_names:
        raise ValueError("l2_name must be present in module_names")
    if not 0.0 < settings.prbs_fraction < 1.0:
        raise ValueError("settings must produce strictly positive PRBS multipliers")

    multipliers = {}
    for name in module_names:
        seed = _module_seed(settings.prbs_seed, name)
        generator = random.Random(seed)
        signs = [1.0] * (settings.calibration_windows // 2)
        signs.extend([-1.0] * (settings.calibration_windows // 2))
        if settings.calibration_windows % 2:
            signs.append(0.0)
        generator.shuffle(signs)
        multipliers[name] = [1.0 + settings.prbs_fraction * sign for sign in signs]
    return {
        "seed": settings.prbs_seed,
        "fraction": settings.prbs_fraction,
        "window_count": settings.calibration_windows,
        "l2_name": l2_name,
        "multipliers": multipliers,
    }


def interpolation_domain(design: dict, tier: int) -> dict:
    """Return the persisted interpolation domain for one permitted tier."""
    if not isinstance(design, dict) or not isinstance(design.get("domains"), dict):
        raise ValueError("design does not contain interpolation domains")
    if isinstance(tier, bool) or tier not in (0, 1):
        raise ValueError("tier must be 0 or 1")
    try:
        return deepcopy(design["domains"][str(tier)])
    except KeyError as error:
        raise ValueError(f"tier {tier} is not in the calibration design") from error


def _exactly_one_l2(layout: dict) -> dict:
    l2s = [module for module in layout.get("modules", []) if module.get("kind") == "l2"]
    if len(l2s) != 1:
        raise ValueError("ROM calibration requires exactly one L2 module")
    return l2s[0]


def _rectangle_anchors(base_layout: dict, l2: dict, tier: int,
                       existing: list[dict]) -> list[dict]:
    maximum_x, maximum_y = _coordinate_limits(base_layout, l2)
    if _rectangle_is_legal(base_layout, tier, _rectangle_coordinates(maximum_x, maximum_y, 0.0), existing):
        scale = 0.0
    else:
        scale = _find_rectangle_shrink(base_layout, tier, maximum_x, maximum_y, existing)
    rules = ("lower_left", "lower_right", "upper_left", "upper_right")
    return [
        {
            "id": f"a{len(existing) + index}",
            "tier": tier,
            "x_mm": coordinates[0],
            "y_mm": coordinates[1],
            "rule": rule,
        }
        for index, (rule, coordinates) in enumerate(
            zip(rules, _rectangle_coordinates(maximum_x, maximum_y, scale))
        )
    ]


def _find_rectangle_shrink(base_layout: dict, tier: int, maximum_x: float,
                           maximum_y: float, existing: Iterable[dict]) -> float:
    """Shrink all four extremes together until they form one legal rectangle."""
    upper_step = _SEARCH_GRID // 2
    for step in range(1, upper_step + 1):
        scale = step / _SEARCH_GRID
        coordinates = _rectangle_coordinates(maximum_x, maximum_y, scale)
        if not _rectangle_is_legal(base_layout, tier, coordinates, existing):
            continue
        lower, upper = (step - 1) / _SEARCH_GRID, scale
        for _ in range(_BISECTION_STEPS):
            middle = (lower + upper) / 2.0
            coordinates = _rectangle_coordinates(maximum_x, maximum_y, middle)
            if _rectangle_is_legal(base_layout, tier, coordinates, existing):
                upper = middle
            else:
                lower = middle
        return upper
    raise ValueError(
        f"cannot construct a legal axis-aligned bilinear rectangle on tier {tier}"
    )


def _rectangle_coordinates(maximum_x: float, maximum_y: float,
                           scale: float) -> tuple[tuple[float, float], ...]:
    minimum_x = scale * maximum_x / 2.0
    maximum_x = maximum_x - minimum_x
    minimum_y = scale * maximum_y / 2.0
    maximum_y = maximum_y - minimum_y
    return (
        (minimum_x, minimum_y),
        (maximum_x, minimum_y),
        (minimum_x, maximum_y),
        (maximum_x, maximum_y),
    )


def _rectangle_is_legal(base_layout: dict, tier: int,
                        coordinates: Iterable[tuple[float, float]],
                        existing: Iterable[dict]) -> bool:
    candidates = list(coordinates)
    if len({x_mm for x_mm, _ in candidates}) != 2 or len({y_mm for _, y_mm in candidates}) != 2:
        return False
    return all(_is_legal_unique(base_layout, tier, candidate, existing) for candidate in candidates)


def _single_tier_anchors(base_layout: dict, l2: dict, tier: int,
                         existing: list[dict]) -> list[dict]:
    maximum_x, maximum_y = _coordinate_limits(base_layout, l2)
    candidates = (
        ("lower_left", 0.0, 0.0),
        ("lower_right", maximum_x, 0.0),
        ("upper_left", 0.0, maximum_y),
        ("upper_right", maximum_x, maximum_y),
        ("lower_midpoint", maximum_x / 2.0, 0.0),
        ("right_midpoint", maximum_x, maximum_y / 2.0),
        ("upper_midpoint", maximum_x / 2.0, maximum_y),
        ("left_midpoint", 0.0, maximum_y / 2.0),
    )
    anchors = []
    for rule, x_mm, y_mm in candidates:
        anchors.append(_legal_point(
            base_layout, l2, tier, x_mm, y_mm, existing + anchors,
            rule=rule, identifier=f"a{len(existing) + len(anchors)}",
        ))
    return anchors


def _interior_holdout(base_layout: dict, l2: dict, tier: int, anchors: list[dict],
                      fractions: tuple[float, float], existing: list[dict],
                      identifier: str) -> dict:
    xs = [point["x_mm"] for point in anchors]
    ys = [point["y_mm"] for point in anchors]
    x_mm = min(xs) + fractions[0] * (max(xs) - min(xs))
    y_mm = min(ys) + fractions[1] * (max(ys) - min(ys))
    return _legal_point(
        base_layout, l2, tier, x_mm, y_mm, existing,
        rule="interior_holdout", identifier=identifier, fractions=fractions,
    )


def _legal_point(base_layout: dict, l2: dict, tier: int, x_mm: float, y_mm: float,
                 existing: Iterable[dict], *, rule: str, identifier: str,
                 fractions: tuple[float, float] | None = None) -> dict:
    maximum_x, maximum_y = _coordinate_limits(base_layout, l2)
    start = (min(max(x_mm, 0.0), maximum_x), min(max(y_mm, 0.0), maximum_y))
    center = (maximum_x / 2.0, maximum_y / 2.0)
    candidate = _toward_center_by_bisection(
        base_layout, tier, start, center, existing
    )
    if candidate is None:
        candidate = _nearest_legal_grid_point(base_layout, tier, start, existing)
    if candidate is None:
        raise ValueError(f"no legal unique L2 placement for {rule} on tier {tier}")
    point = {
        "id": identifier,
        "tier": tier,
        "x_mm": candidate[0],
        "y_mm": candidate[1],
        "rule": rule,
    }
    if fractions is not None:
        point["fractions"] = list(fractions)
    layout_for_point(base_layout, point)
    _reject_duplicates(list(existing) + [point])
    return point


def _toward_center_by_bisection(base_layout: dict, tier: int,
                                start: tuple[float, float], center: tuple[float, float],
                                existing: Iterable[dict]) -> tuple[float, float] | None:
    """Keep a legal extreme, or shrink an illegal one toward die center."""
    if _is_legal_unique(base_layout, tier, start, existing):
        return start
    previous = 0.0
    for step in range(1, _SEARCH_GRID + 1):
        fraction = step / _SEARCH_GRID
        candidate = _interpolate(start, center, fraction)
        if not _is_legal_unique(base_layout, tier, candidate, existing):
            previous = fraction
            continue
        low, high = previous, fraction
        for _ in range(_BISECTION_STEPS):
            middle = (low + high) / 2.0
            if _is_legal_unique(base_layout, tier, _interpolate(start, center, middle), existing):
                high = middle
            else:
                low = middle
        return _interpolate(start, center, high)
    return None


def _nearest_legal_grid_point(base_layout: dict, tier: int, target: tuple[float, float],
                              existing: Iterable[dict]) -> tuple[float, float] | None:
    """Find a deterministic legal fallback when the center ray is obstructed."""
    l2 = _exactly_one_l2(base_layout)
    maximum_x, maximum_y = _coordinate_limits(base_layout, l2)
    best = None
    best_distance = math.inf
    for row in range(_SEARCH_GRID + 1):
        y_mm = maximum_y * row / _SEARCH_GRID
        for column in range(_SEARCH_GRID + 1):
            x_mm = maximum_x * column / _SEARCH_GRID
            candidate = (x_mm, y_mm)
            if not _is_legal_unique(base_layout, tier, candidate, existing):
                continue
            distance = (x_mm - target[0]) ** 2 + (y_mm - target[1]) ** 2
            if distance < best_distance - _TOLERANCE_MM:
                best, best_distance = candidate, distance
    return best


def _is_legal_unique(base_layout: dict, tier: int, candidate: tuple[float, float],
                     existing: Iterable[dict]) -> bool:
    point = {"tier": tier, "x_mm": candidate[0], "y_mm": candidate[1]}
    try:
        layout_for_point(base_layout, point)
    except ValueError:
        return False
    return not any(
        point["tier"] == other["tier"]
        and math.hypot(point["x_mm"] - other["x_mm"], point["y_mm"] - other["y_mm"])
        <= _TOLERANCE_MM
        for other in existing
    )


def _coordinate_limits(layout: dict, l2: dict) -> tuple[float, float]:
    maximum_x = layout["die_width_mm"] - l2["width_mm"]
    maximum_y = layout["die_height_mm"] - l2["height_mm"]
    if maximum_x < -_TOLERANCE_MM or maximum_y < -_TOLERANCE_MM:
        raise ValueError("L2 does not fit inside the die")
    return max(maximum_x, 0.0), max(maximum_y, 0.0)


def _interpolate(start: tuple[float, float], end: tuple[float, float],
                 fraction: float) -> tuple[float, float]:
    return (
        start[0] + fraction * (end[0] - start[0]),
        start[1] + fraction * (end[1] - start[1]),
    )


def _reject_duplicates(points: Iterable[dict]) -> None:
    seen = []
    for point in points:
        for other in seen:
            if point["tier"] == other["tier"] and math.hypot(
                point["x_mm"] - other["x_mm"], point["y_mm"] - other["y_mm"]
            ) <= _TOLERANCE_MM:
                raise ValueError("duplicate L2 calibration point")
        seen.append(point)


def _domain_for_tier(training: list[dict], tier: int, *, two_tier: bool) -> dict:
    points = [point for point in training if point["tier"] == tier]
    coordinates = [[point["x_mm"], point["y_mm"]] for point in points]
    if two_tier:
        if len(points) != 4:
            raise ValueError("bilinear interpolation requires four anchors per tier")
        x_values = {point["x_mm"] for point in points}
        y_values = {point["y_mm"] for point in points}
        if len(x_values) != 2 or len(y_values) != 2 or {
            (point["x_mm"], point["y_mm"]) for point in points
        } != {(x_mm, y_mm) for x_mm in x_values for y_mm in y_values}:
            raise ValueError("bilinear interpolation requires an axis-aligned legal rectangle")
        return {"kind": "bilinear", "anchor_ids": [point["id"] for point in points],
                "corners": coordinates}
    if len(points) != 8:
        raise ValueError("Delaunay interpolation requires eight anchors")
    return {
        "kind": "delaunay",
        "anchor_ids": [point["id"] for point in points],
        "points": coordinates,
        "simplices": _delaunay_simplices(coordinates),
    }


def _delaunay_simplices(points: list[list[float]]) -> list[list[int]]:
    """Persist SciPy's deterministic Delaunay triangulation."""
    try:
        from scipy.spatial import Delaunay  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise ValueError(
            "SciPy is required for single-tier Delaunay interpolation but is not installed"
        ) from error
    triangulation = Delaunay(points)
    return [list(map(int, simplex)) for simplex in triangulation.simplices.tolist()]


def _module_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")
