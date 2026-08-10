"""Pure numerical layout evaluation and validation gates for transient ROMs."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy

from workflow.common import write_json
from workflow.transient.rom.calibration_design import interpolation_domain
from workflow.transient.rom.contracts import ROMSettings, _normalized_identity
from workflow.transient.rom.pod_state_space import StateSpaceModel, discretize
from workflow.transient.validation import validate_power_triplet
from workflow.transient.verify_sustainable_frequency import find_sustainable_frequency


_NEGATIVE_WEIGHT_TOLERANCE = -1e-12
_DEGENERATE_RELATIVE_TOLERANCE = 1e-14
_STEADY_SOLVE_RESIDUAL_LIMIT = 1e-10


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _anchor_columns(model: StateSpaceModel, anchor_ids: list[str]) -> list[numpy.ndarray]:
    if not isinstance(model, StateSpaceModel):
        raise ValueError("model must be a StateSpaceModel")
    if not isinstance(anchor_ids, list) or not anchor_ids:
        raise ValueError("ROM interpolation domain has invalid anchor_ids")
    if len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("ROM interpolation domain has duplicate anchor_ids")
    rank = numpy.asarray(model.temperature_basis).shape[1]
    result = []
    for identifier in anchor_ids:
        if not isinstance(identifier, str) or identifier not in model.b_l2_anchors:
            raise ValueError(f"ROM interpolation anchor {identifier!r} is absent from model")
        column = numpy.asarray(model.b_l2_anchors[identifier], dtype=float)
        if column.shape != (rank, 1) or not numpy.all(numpy.isfinite(column)):
            raise ValueError(f"ROM interpolation anchor {identifier!r} is malformed")
        result.append(column)
    return result


def _coordinates(value: Any, expected: int, name: str) -> numpy.ndarray:
    array = numpy.asarray(value, dtype=float)
    if array.shape != (expected, 2) or not numpy.all(numpy.isfinite(array)):
        raise ValueError(f"ROM interpolation domain has invalid {name}")
    return array


def _weighted_column(columns: list[numpy.ndarray], weights: numpy.ndarray) -> numpy.ndarray:
    if numpy.any(weights < _NEGATIVE_WEIGHT_TOLERANCE):
        raise ValueError("outside ROM interpolation domain: negative barycentric weight")
    clipped = numpy.maximum(weights, 0.0)
    total = float(numpy.sum(clipped))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("ROM interpolation domain produced invalid weights")
    clipped /= total
    exact = numpy.flatnonzero(clipped == 1.0)
    if len(exact) == 1:
        return columns[int(exact[0])].copy()
    return sum(
        (weight * column for weight, column in zip(clipped, columns)),
        start=numpy.zeros_like(columns[0]),
    )


def _bilinear(column_by_coordinate: dict[tuple[float, float], numpy.ndarray],
              x_mm: float, y_mm: float) -> numpy.ndarray:
    x_values = sorted({coordinate[0] for coordinate in column_by_coordinate})
    y_values = sorted({coordinate[1] for coordinate in column_by_coordinate})
    if len(x_values) != 2 or len(y_values) != 2:
        raise ValueError("bilinear ROM interpolation domain is not a rectangle")
    expected = {(x, y) for x in x_values for y in y_values}
    if set(column_by_coordinate) != expected:
        raise ValueError("bilinear ROM interpolation domain is not a rectangle")
    x0, x1 = x_values
    y0, y1 = y_values
    if not x0 <= x_mm <= x1 or not y0 <= y_mm <= y1:
        raise ValueError("outside ROM interpolation domain")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bilinear ROM interpolation domain is degenerate")
    for coordinate, column in column_by_coordinate.items():
        if x_mm == coordinate[0] and y_mm == coordinate[1]:
            return column.copy()
    tx = (x_mm - x0) / (x1 - x0)
    ty = (y_mm - y0) / (y1 - y0)
    columns = [
        column_by_coordinate[(x0, y0)], column_by_coordinate[(x1, y0)],
        column_by_coordinate[(x0, y1)], column_by_coordinate[(x1, y1)],
    ]
    weights = numpy.asarray(
        [(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty),
         (1.0 - tx) * ty, tx * ty],
        dtype=float,
    )
    return _weighted_column(columns, weights)


def _barycentric_weights(triangle: numpy.ndarray, point: numpy.ndarray) -> numpy.ndarray:
    edges = numpy.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    determinant = float(numpy.linalg.det(edges))
    scale = max(1.0, float(numpy.max(numpy.abs(edges))))
    if abs(determinant) <= _DEGENERATE_RELATIVE_TOLERANCE * scale * scale:
        raise ValueError("degenerate simplex in ROM interpolation domain")
    tail = numpy.linalg.solve(edges, point - triangle[0])
    return numpy.asarray([1.0 - tail[0] - tail[1], tail[0], tail[1]], dtype=float)


def _delaunay(columns: list[numpy.ndarray], points: numpy.ndarray,
              simplices: Any, x_mm: float, y_mm: float) -> numpy.ndarray:
    if not isinstance(simplices, list) or not simplices:
        raise ValueError("Delaunay ROM interpolation domain has invalid simplices")
    parsed = []
    for simplex in simplices:
        if (not isinstance(simplex, list) or len(simplex) != 3
                or any(isinstance(index, bool) or not isinstance(index, int)
                       or index < 0 or index >= len(points) for index in simplex)
                or len(set(simplex)) != 3):
            raise ValueError("Delaunay ROM interpolation domain has invalid simplex")
        parsed.append(simplex)
    query = numpy.asarray([x_mm, y_mm], dtype=float)
    for index, coordinate in enumerate(points):
        if numpy.array_equal(query, coordinate):
            return columns[index].copy()
    candidates = []
    for simplex in parsed:
        weights = _barycentric_weights(points[simplex], query)
        if numpy.all(weights >= _NEGATIVE_WEIGHT_TOLERANCE):
            candidates.append((simplex, weights))
    if not candidates:
        raise ValueError("outside ROM interpolation domain")
    # A boundary can belong to two persisted simplices.  Their barycentric
    # predictions must agree at the shared edge; deterministic ordering keeps
    # floating-point output repeatable.
    simplex, weights = min(candidates, key=lambda value: tuple(value[0]))
    return _weighted_column([columns[index] for index in simplex], weights)


def interpolate_l2_input(model: StateSpaceModel, design: dict,
                         tier: int, x_mm: float, y_mm: float) -> numpy.ndarray:
    """Interpolate the L2 input column within one persisted tier domain."""
    if isinstance(tier, bool) or not isinstance(tier, int) or tier not in (0, 1):
        raise ValueError("tier must be 0 or 1")
    x = _finite(x_mm, "x_mm")
    y = _finite(y_mm, "y_mm")
    domain = interpolation_domain(design, tier)
    anchor_ids = domain.get("anchor_ids")
    columns = _anchor_columns(model, anchor_ids)
    kind = domain.get("kind")
    if kind == "bilinear":
        corners = _coordinates(domain.get("corners"), len(columns), "corners")
        if len(columns) != 4:
            raise ValueError("bilinear ROM interpolation requires four anchors")
        mapping = {
            (float(coordinate[0]), float(coordinate[1])): column
            for coordinate, column in zip(corners, columns)
        }
        if len(mapping) != len(columns):
            raise ValueError("bilinear ROM interpolation domain has duplicate corners")
        return _bilinear(mapping, x, y)
    if kind == "delaunay":
        points = _coordinates(domain.get("points"), len(columns), "points")
        return _delaunay(columns, points, domain.get("simplices"), x, y)
    raise ValueError(f"unsupported ROM interpolation kind: {kind!r}")


def _layout_l2(layout: dict, design: dict) -> dict:
    if not isinstance(layout, dict) or not isinstance(layout.get("modules"), list):
        raise ValueError("layout must contain a module list")
    expected_name = design.get("l2_name")
    if not isinstance(expected_name, str) or not expected_name:
        raise ValueError("design lacks l2_name")
    l2s = [module for module in layout["modules"]
           if isinstance(module, dict) and module.get("kind") == "l2"]
    if len(l2s) != 1 or l2s[0].get("name") != expected_name:
        raise ValueError("layout must contain exactly the designed L2 module")
    l2 = l2s[0]
    tier = l2.get("tier")
    if isinstance(tier, bool) or not isinstance(tier, int) or tier not in (0, 1):
        raise ValueError("layout L2 tier must be 0 or 1")
    _finite(l2.get("x_mm"), "layout L2 x_mm")
    _finite(l2.get("y_mm"), "layout L2 y_mm")
    return l2


def _period_inputs(model: StateSpaceModel, design: dict,
                   power_windows: dict, scale: float) -> list[tuple[float, numpy.ndarray]]:
    if not isinstance(power_windows, dict) or not isinstance(power_windows.get("windows"), list):
        raise ValueError("power_windows must contain a window list")
    windows = power_windows["windows"]
    if not windows:
        raise ValueError("power_windows must contain at least one window")
    nominal_duration = _positive(
        power_windows.get("nominal_sample_interval_ms"),
        "power_windows nominal_sample_interval_ms",
    ) / 1000.0
    l2_name = design.get("l2_name")
    expected_order = (*model.module_names, l2_name)
    if (not isinstance(l2_name, str) or not l2_name
            or len(set(expected_order)) != len(expected_order)):
        raise ValueError("model and design contain an invalid module order")
    expected = set(expected_order)
    result = []
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(f"window {index} must be a dictionary")
        duration = _positive(window.get("duration_s"), f"window {index} duration_s")
        if duration > nominal_duration and not math.isclose(
            duration, nominal_duration, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(
                f"window {index} duration_s exceeds the HotSpot sampling interval"
            )
        if index < len(windows) - 1 and not math.isclose(
            duration, nominal_duration, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(
                f"non-final window {index} duration_s differs from the HotSpot sampling interval"
            )
        modules = window.get("modules")
        if not isinstance(modules, list):
            raise ValueError(f"window {index} modules must be a list")
        by_name = {}
        for module in modules:
            if not isinstance(module, dict):
                raise ValueError(f"window {index} contains a malformed module")
            name = module.get("name")
            if not isinstance(name, str) or not name or name in by_name:
                raise ValueError(f"window {index} contains invalid or duplicate module names")
            validate_power_triplet(module, f"window {index} module {name}")
            by_name[name] = module
        if set(by_name) != expected:
            raise ValueError(f"window {index} module set does not match ROM module order")
        l2_kind = by_name[l2_name].get("kind")
        if l2_kind != "l2":
            raise ValueError(f"window {index} designed L2 module has kind {l2_kind!r}")
        powers = numpy.asarray([
            float(by_name[name]["leakage_power_w"])
            + scale * float(by_name[name]["dynamic_power_w"])
            for name in expected_order
        ], dtype=float)
        if not numpy.all(numpy.isfinite(powers)) or numpy.any(powers < 0.0):
            raise ValueError(f"window {index} has invalid frequency-scaled module power")
        # HotSpot accepts one power row per fixed sampling interval.  Its last
        # partial gem5 window is therefore held through the remaining padding,
        # so the ROM must integrate that row for the same fixed interval.
        result.append((nominal_duration / scale, powers))
    return result


def _temperature(model: StateSpaceModel, state: numpy.ndarray,
                 ambient_c: float) -> numpy.ndarray:
    grid = numpy.asarray(model.temperature_basis, dtype=float) @ state
    if grid.ndim != 1 or not numpy.all(numpy.isfinite(grid)):
        raise ValueError("ROM reconstructed a malformed full-grid temperature")
    return grid + ambient_c


def _average_power_steady_state(
    model: StateSpaceModel,
    b_l2: numpy.ndarray,
    period_inputs: list[tuple[float, numpy.ndarray]],
    max_condition_number: float,
) -> tuple[numpy.ndarray, dict]:
    """Solve ``A x + B(l) mean(u) = 0`` with fail-closed diagnostics."""
    if not isinstance(model, StateSpaceModel):
        raise ValueError("model must be a StateSpaceModel")
    a_continuous = numpy.asarray(model.a_continuous, dtype=float)
    b_fixed = numpy.asarray(model.b_fixed, dtype=float)
    b_l2 = numpy.asarray(b_l2, dtype=float)
    rank = numpy.asarray(model.temperature_basis, dtype=float).shape[1]
    if (a_continuous.shape != (rank, rank)
            or b_fixed.shape != (rank, len(model.module_names))
            or b_l2.shape != (rank, 1)
            or not numpy.all(numpy.isfinite(a_continuous))
            or not numpy.all(numpy.isfinite(b_fixed))
            or not numpy.all(numpy.isfinite(b_l2))):
        raise ValueError("ROM steady initialization matrices are malformed")
    limit = _positive(max_condition_number, "max_condition_number")
    condition_number = float(numpy.linalg.cond(a_continuous))
    if (not math.isfinite(condition_number)
            or condition_number > limit):
        raise ValueError(
            "ROM steady initialization state matrix condition number "
            f"{condition_number} exceeds {limit}"
        )
    if not isinstance(period_inputs, list) or not period_inputs:
        raise ValueError("ROM steady initialization requires period inputs")
    input_count = len(model.module_names) + 1
    weighted_power = numpy.zeros(input_count, dtype=float)
    total_duration = 0.0
    for index, item in enumerate(period_inputs):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                f"ROM steady initialization period input {index} is malformed"
            )
        duration = _positive(item[0], f"period input {index} duration")
        powers = numpy.asarray(item[1], dtype=float)
        if (powers.shape != (input_count,)
                or not numpy.all(numpy.isfinite(powers))
                or numpy.any(powers < 0.0)):
            raise ValueError(
                f"ROM steady initialization period input {index} power is malformed"
            )
        weighted_power += duration * powers
        total_duration += duration
    mean_power = weighted_power / total_duration
    b_layout = numpy.concatenate((b_fixed, b_l2), axis=1)
    forcing = b_layout @ mean_power
    try:
        state = numpy.linalg.solve(a_continuous, -forcing)
    except numpy.linalg.LinAlgError as error:
        raise ValueError(
            "ROM steady initialization state matrix solve failed"
        ) from error
    if state.shape != (rank,) or not numpy.all(numpy.isfinite(state)):
        raise ValueError("ROM steady initialization produced a malformed state")
    residual = a_continuous @ state + forcing
    denominator = max(
        1.0,
        float(numpy.linalg.norm(a_continuous) * numpy.linalg.norm(state)
              + numpy.linalg.norm(forcing)),
    )
    normalized_residual = float(numpy.linalg.norm(residual) / denominator)
    if (not math.isfinite(normalized_residual)
            or normalized_residual > _STEADY_SOLVE_RESIDUAL_LIMIT):
        raise ValueError(
            "ROM steady initialization normalized residual "
            f"{normalized_residual} exceeds {_STEADY_SOLVE_RESIDUAL_LIMIT}"
        )
    return state, {
        "method": "average_power_continuous_equilibrium",
        "condition_number": condition_number,
        "max_condition_number": limit,
        "normalized_residual": normalized_residual,
        "max_normalized_residual": _STEADY_SOLVE_RESIDUAL_LIMIT,
        "mean_power_w": mean_power.tolist(),
        "period_duration_s": total_duration,
    }


def evaluate_layout_rom(model: StateSpaceModel, design: dict, power_windows: dict,
                        layout: dict, frequency_ghz: float, settings: ROMSettings,
                        config: dict) -> dict:
    """Evaluate a layout/frequency from matched average-power steady state."""
    if not isinstance(settings, ROMSettings):
        raise ValueError("settings must be ROMSettings")
    if settings.pss_period_repeats < 2:
        raise ValueError("pss_period_repeats must be at least two")
    if not isinstance(config, dict) or not isinstance(config.get("frequency"), dict):
        raise ValueError("config lacks frequency settings")
    frequency_settings = config["frequency"]
    frequency = _positive(frequency_ghz, "frequency_ghz")
    f0 = _positive(frequency_settings.get("f0_ghz"), "frequency.f0_ghz")
    ambient = _finite(frequency_settings.get("ambient_c"), "frequency.ambient_c")
    tsafe = _finite(frequency_settings.get("tsafe_c"), "frequency.tsafe_c")
    scale = frequency / f0
    l2 = _layout_l2(layout, design)
    b_l2 = interpolate_l2_input(
        model, design, l2["tier"], float(l2["x_mm"]), float(l2["y_mm"])
    )
    period_inputs = _period_inputs(model, design, power_windows, scale)
    discretizations = [
        (*discretize(model, duration, b_l2), powers)
        for duration, powers in period_inputs
    ]
    state, initialization = _average_power_steady_state(
        model, b_l2, period_inputs, settings.max_condition_number,
    )
    previous_end = _temperature(model, state, ambient)
    period_end_deltas = []
    converged = False
    final_start = previous_end.copy()
    final_rows: list[numpy.ndarray] = []
    periods = 0
    for period in range(settings.pss_period_repeats):
        final_start = _temperature(model, state, ambient)
        final_rows = []
        for a_discrete, b_discrete, powers in discretizations:
            state = a_discrete @ state + b_discrete @ powers
            final_rows.append(_temperature(model, state, ambient))
        current_end = final_rows[-1]
        if period > 0:
            delta = numpy.abs(current_end - previous_end)
            period_end_deltas.append(float(numpy.max(delta)))
        previous_end = current_end.copy()
        periods = period + 1
    converged = period_end_deltas[-1] <= settings.pss_tolerance_c
    inclusive = [final_start, *final_rows]
    peak_sample, peak_unit = max(
        ((sample, unit) for sample in range(len(inclusive))
         for unit in range(len(inclusive[sample]))),
        key=lambda item: inclusive[item[0]][item[1]],
    )
    peak_c = float(inclusive[peak_sample][peak_unit])
    return {
        "schema_version": 1,
        "thermal_mode": "transient-rom",
        "non_formal": True,
        "paper_equivalent": False,
        "frequency_ghz": frequency,
        "frequency_scale": scale,
        "module_order": [*model.module_names, design["l2_name"]],
        "l2_position": {
            "tier": l2["tier"], "x_mm": float(l2["x_mm"]),
            "y_mm": float(l2["y_mm"]),
        },
        "thermal_initialization": initialization,
        "periods_evaluated": periods,
        "period_end_deltas_max_c": period_end_deltas,
        "pss_tolerance_c": settings.pss_tolerance_c,
        "converged": converged,
        "period_start_grid_c": final_start.tolist(),
        "final_period_grid_c": [row.tolist() for row in final_rows],
        "last_period_peak_c": peak_c,
        "last_period_peak": {
            "tmax_c": peak_c,
            "sample_index": peak_sample,
            "unit_index": peak_unit,
            "includes_period_initial_state": True,
            "phase": "period_initial_state" if peak_sample == 0 else "window_end",
            "window_index_in_period": peak_sample - 1,
        },
        "safe": converged and peak_c <= tsafe,
        "tsafe_c": tsafe,
    }


def find_rom_sustainable_frequency(model: StateSpaceModel, design: dict,
                                   power_windows: dict, layout: dict,
                                   frequencies_ghz: list[float],
                                   settings: ROMSettings, config: dict) -> dict:
    """Search ROM frequency using the real verifier's local bracket policy."""
    if not isinstance(settings, ROMSettings):
        raise ValueError("settings must be ROMSettings")
    if not isinstance(config, dict) or not isinstance(config.get("frequency"), dict):
        raise ValueError("config lacks frequency settings")
    tsafe = _finite(config["frequency"].get("tsafe_c"), "frequency.tsafe_c")

    def evaluate(frequency: float) -> dict:
        return evaluate_layout_rom(
            model, design, power_windows, layout, frequency, settings, config
        )

    return find_sustainable_frequency(
        evaluate, frequencies_ghz, tsafe, settings.frequency_tolerance_ghz
    )


def _finite_grid(value: Any, name: str) -> numpy.ndarray:
    grid = numpy.asarray(value, dtype=float)
    if grid.ndim != 2 or not grid.size or not numpy.all(numpy.isfinite(grid)):
        raise ValueError(f"{name} must be a finite non-empty two-dimensional grid")
    return grid


def validate_holdouts(holdouts: list[dict], settings: ROMSettings, *,
                      output_dir: Path | None = None,
                      identity: dict | None = None) -> dict:
    """Apply all two-holdout gates and conditionally publish acceptance."""
    if not isinstance(settings, ROMSettings):
        raise ValueError("settings must be ROMSettings")
    if not isinstance(holdouts, list) or len(holdouts) != 2:
        raise ValueError("holdouts must contain exactly two comparisons")
    cases = []
    failures: list[str] = []
    for index, comparison in enumerate(holdouts):
        if not isinstance(comparison, dict):
            raise ValueError(f"holdout {index} comparison must be a dictionary")
        identifier = comparison.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"holdout {index} lacks an id")
        rom, hotspot = comparison.get("rom"), comparison.get("hotspot")
        if not isinstance(rom, dict) or not isinstance(hotspot, dict):
            raise ValueError(f"holdout {identifier} lacks ROM/HotSpot results")
        gate = {
            "geometry_identity": comparison.get("geometry_match") is True,
            "input_identity": comparison.get("input_identity_match") is True,
            "frequency_identity": comparison.get("frequency_match") is True,
            "hotspot_trace_identity": (
                comparison.get("hotspot_trace_identity_match") is True
            ),
            "temperature_grid_identity": (
                comparison.get("temperature_grid_identity_match") is True
            ),
            "periodic_steady_state": (
                rom.get("converged") is True and hotspot.get("converged") is True
            ),
        }
        rmse = None
        if gate["temperature_grid_identity"]:
            rom_grid = _finite_grid(
                rom.get("final_period_grid_c"), f"holdout {identifier} ROM grid"
            )
            hotspot_grid = _finite_grid(
                hotspot.get("final_period_grid_c"), f"holdout {identifier} HotSpot grid"
            )
            if rom_grid.shape != hotspot_grid.shape:
                raise ValueError(f"holdout {identifier} ROM/HotSpot grids differ in shape")
            rmse = float(numpy.sqrt(numpy.mean((rom_grid - hotspot_grid) ** 2)))
        rom_peak = _finite(
            rom.get("last_period_peak_c"), f"holdout {identifier} ROM peak"
        )
        hotspot_peak = _finite(
            hotspot.get("last_period_peak_c"), f"holdout {identifier} HotSpot peak"
        )
        peak_error = abs(rom_peak - hotspot_peak)
        rom_safe, hotspot_safe = rom.get("safe"), hotspot.get("safe")
        if not isinstance(rom_safe, bool) or not isinstance(hotspot_safe, bool):
            raise ValueError(f"holdout {identifier} lacks boolean safety classes")
        gate.update({
            "grid_rmse": (
                None if rmse is None
                else rmse <= settings.max_holdout_grid_rmse_c
            ),
            "peak_temperature_error": peak_error <= settings.max_holdout_peak_error_c,
            "safety_classification": rom_safe == hotspot_safe,
        })
        for name, passed in gate.items():
            if passed is False and name not in failures:
                failures.append(name)
        cases.append({
            "id": identifier,
            "frequency_ghz": comparison.get("frequency_ghz"),
            "grid_rmse_c": rmse,
            "peak_temperature_error_c": peak_error,
            "rom_peak_c": rom_peak,
            "hotspot_peak_c": hotspot_peak,
            "rom_safe": rom_safe,
            "hotspot_safe": hotspot_safe,
            "gates": gate,
            "accepted": all(gate.values()),
        })
    accepted = not failures and all(case["accepted"] for case in cases)
    report = {
        "schema_version": 1,
        "thermal_mode": "transient-rom",
        "non_formal": True,
        "paper_equivalent": False,
        "thresholds": {
            "pss_tolerance_c": settings.pss_tolerance_c,
            "max_holdout_grid_rmse_c": settings.max_holdout_grid_rmse_c,
            "max_holdout_peak_error_c": settings.max_holdout_peak_error_c,
        },
        "holdouts": cases,
        "failure_reasons": failures,
        "accepted": accepted,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "validation_report.json", report)
        acceptance_path = destination / "rom_acceptance.json"
        if accepted:
            if not isinstance(identity, dict):
                raise ValueError("accepted ROM publication requires a complete identity")
            try:
                normalized_identity = _normalized_identity(identity, "ROM acceptance")
            except ValueError:
                acceptance_path.unlink(missing_ok=True)
                raise
            acceptance = {
                "schema_version": 1,
                "thermal_mode": "transient-rom",
                "non_formal": True,
                "paper_equivalent": False,
                "accepted": True,
                "identity": normalized_identity,
                "validation_report": "validation_report.json",
            }
            write_json(acceptance_path, acceptance)
        elif acceptance_path.exists():
            acceptance_path.unlink()
    return report
