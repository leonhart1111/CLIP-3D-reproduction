#!/usr/bin/env python3
"""Optimize L2 tier/coordinates with equations (13)--(15), then emit a layout.

SciPy's L-BFGS-B is used when available.  A deterministic bounded pattern
search is provided for machines such as this EDA node where SciPy is absent;
the result records which solver was actually used.  All reported final
temperatures must still come from workflow.thermal.run_hotspot.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import baseline_layout, check_geometry, overlap
from workflow.floorplan.layout_metrics import (
    aggregate_wire_cycles,
    communication_weights_from_model,
    derive_layout_delays,
    mean_wire_cycles as layout_mean_wire_cycles,
    round_wire_cycles,
)
from workflow.thermal.sustainable_frequency import closed_form_frequency


def quadrature_points(module: dict, order: int) -> list[tuple[float, float]]:
    if order == 1:
        fractions = (0.5,)
    elif order == 2:
        # Two-point Gauss-Legendre nodes mapped from [-1,1] to [0,1].
        offset = 0.5 / math.sqrt(3.0)
        fractions = (0.5 - offset, 0.5 + offset)
    elif order == 3:
        offset = 0.5 * math.sqrt(3.0 / 5.0)
        fractions = (0.5 - offset, 0.5, 0.5 + offset)
    else:
        raise ValueError("proxy quadrature order must be 1, 2, or 3")
    return [
        (module["x_mm"] + fx * module["width_mm"],
         module["y_mm"] + fy * module["height_mm"])
        for fy in fractions for fx in fractions
    ]


def proxy_temperature(modules: list[dict], side: float, ambient: float,
                      r_convec: float, alpha: float, beta: float,
                      cross_tier_weight: float,
                      spatial_model: str = "center",
                      quadrature_order: int = 2) -> float:
    total = sum(module["total_power_w"] for module in modules)
    if spatial_model == "center":
        samples = [(
            [(m["x_mm"] + m["width_mm"] / 2,
              m["y_mm"] + m["height_mm"] / 2)], m
        ) for m in modules]
    elif spatial_model == "area-quadrature":
        samples = [(quadrature_points(m, quadrature_order), m) for m in modules]
    else:
        raise ValueError("thermal proxy spatial_model must be center or area-quadrature")
    length = side / 2.0
    hotspot = 0.0
    for receiver_points, receiver in samples:
        for xi, yi in receiver_points:
            coupled = 0.0
            for source_points, source in samples:
                point_power = source["total_power_w"] / len(source_points)
                weight = (
                    1.0 if receiver["tier"] == source["tier"] else cross_tier_weight
                )
                for xj, yj in source_points:
                    distance = math.hypot(xi - xj, yi - yj)
                    kernel = 1.0 / math.sqrt(1.0 + (distance / length) ** 2)
                    coupled += point_power * kernel * weight
            hotspot = max(hotspot, coupled)
    bottom_power = sum(m["total_power_w"] for m in modules if m["tier"] == 0)
    return ambient + r_convec * total + alpha * hotspot + beta * bottom_power


def collision_area(candidate: dict, others: list[dict]) -> float:
    return sum(overlap(candidate, m["x_mm"], m["y_mm"],
                       m["x_mm"] + m["width_mm"], m["y_mm"] + m["height_mm"])
               for m in others if m["tier"] == candidate["tier"])


def pattern_search(function, start: tuple[float, float],
                   upper: tuple[float, float]) -> tuple[list[float], float, int]:
    point = [min(max(start[index], 0.0), upper[index]) for index in range(2)]
    value = function(point)
    step = max(max(upper) / 4.0, 1e-4)
    evaluations = 1
    while step > max(max(upper) * 1e-5, 1e-6) and evaluations < 1000:
        improved = False
        for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step),
                       (step, step), (step, -step), (-step, step), (-step, -step)):
            trial = [min(max(point[0] + dx, 0.0), upper[0]),
                     min(max(point[1] + dy, 0.0), upper[1])]
            trial_value = function(trial)
            evaluations += 1
            if trial_value < value:
                point, value, improved = trial, trial_value, True
        if not improved:
            step *= 0.5
    return point, value, evaluations


def optimize(model_path: Path, output_layout: Path, report_path: Path,
             utilization: float = 0.70, ambient: float = 25.0,
             r_convec: float = 5.0, alpha: float = 0.3, beta: float = 0.1,
             cross_tier_weight: float = 0.65, f0_ghz: float = 2.0,
             fmin_ghz: float = 0.4, tsafe: float = 95.0,
             lambda_wire: float = 0.02, require_scipy: bool = False,
             allowed_l2_tiers: tuple[int, ...] | list[int] | None = None,
             proxy_spatial_model: str = "center",
             proxy_quadrature_order: int = 2,
             wire_objective: str = "continuous",
             wire_rounding: str = "nearest",
             wire_aggregation: str = "mean") -> dict:
    model = read_json(model_path)
    base = baseline_layout(model, utilization)
    side = base["die_width_mm"]
    original = next(m for m in base["modules"] if m["kind"] == "l2")
    fixed = [dict(m) for m in base["modules"] if m["kind"] != "l2"]
    upper = (side - original["width_mm"], side - original["height_mm"])
    if min(upper) < 0:
        raise ValueError("L2 geometry does not fit inside the die")
    if wire_objective not in ("continuous", "r2-quantized"):
        raise ValueError("wire_objective must be continuous or r2-quantized")
    if wire_aggregation not in ("mean", "traffic-weighted"):
        raise ValueError(
            "optimizer wire_aggregation must be mean or traffic-weighted; "
            "maximum is only a conservative R2 sensitivity mode"
        )
    communication_weights = communication_weights_from_model(
        model, required=wire_aggregation == "traffic-weighted"
    )
    starts = ((0.0, 0.0), (upper[0] / 2.0, upper[1] / 2.0), upper)
    candidates = []
    try:
        from scipy.optimize import minimize  # type: ignore
        solver = "scipy-L-BFGS-B"
    except ImportError:
        if require_scipy:
            raise RuntimeError(
                "SciPy is required for the paper L-BFGS-B solver but is not installed"
            )
        minimize = None
        solver = "dependency-free bounded pattern search"

    tiers = tuple(dict.fromkeys(
        int(tier) for tier in (allowed_l2_tiers if allowed_l2_tiers is not None else (0, 1))
    ))
    if not tiers or any(tier not in (0, 1) for tier in tiers):
        raise ValueError("allowed_l2_tiers must contain tier 0 and/or tier 1")

    # Keep the analytic objective and the quantities that can actually change
    # gem5 R2 side by side.  A continuous wire improvement can disappear after
    # cycle discretization, while a low-power movable L2 may have too little
    # thermal leverage to move the real HotSpot result.  Recording both here
    # prevents a lower proxy loss from being mistaken for a measured gain.
    baseline_proxy = proxy_temperature(
        base["modules"], side, ambient, r_convec, alpha, beta,
        cross_tier_weight, proxy_spatial_model, proxy_quadrature_order,
    )
    baseline_frequency = closed_form_frequency(
        baseline_proxy, model["gamma"], f0_ghz, fmin_ghz, tsafe, ambient
    )[0]
    baseline_delays = derive_layout_delays(
        base, f0_ghz, wire_rounding, communication_weights
    )

    def selected_wire_cycles(modules: list[dict]) -> tuple[float, float]:
        mean_cycles, per_core = layout_mean_wire_cycles(modules, f0_ghz)
        selected_cycles = aggregate_wire_cycles(
            per_core, wire_aggregation, communication_weights
        )
        return mean_cycles, selected_cycles

    for tier in tiers:
        def objective(point):
            l2 = dict(original, x_mm=float(point[0]), y_mm=float(point[1]), tier=tier)
            modules = fixed + [l2]
            collision = collision_area(l2, fixed)
            proxy = proxy_temperature(modules, side, ambient, r_convec, alpha, beta,
                                      cross_tier_weight, proxy_spatial_model,
                                      proxy_quadrature_order)
            frequency = closed_form_frequency(proxy, model["gamma"], f0_ghz,
                                               fmin_ghz, tsafe, ambient)[0]
            _, wire = selected_wire_cycles(modules)
            objective_wire = (
                wire if wire_objective == "continuous"
                else round_wire_cycles(wire, wire_rounding)
            )
            # A large dimensional penalty makes illegal overlaps unattractive.
            return (-model["ipc1"] * frequency
                    + lambda_wire * model["ipc1"] * objective_wire
                    + 1e4 * collision)

        for start_name, start in zip(("BL", "CENTER", "TR"), starts):
            if minimize is not None:
                result = minimize(objective, start, method="L-BFGS-B",
                                  bounds=((0.0, upper[0]), (0.0, upper[1])))
                point, value, evaluations = list(map(float, result.x)), float(result.fun), int(result.nfev)
            else:
                point, value, evaluations = pattern_search(objective, start, upper)
            l2 = dict(original, x_mm=point[0], y_mm=point[1], tier=tier)
            modules = fixed + [l2]
            collision = collision_area(l2, fixed)
            proxy = proxy_temperature(modules, side, ambient, r_convec, alpha, beta,
                                      cross_tier_weight, proxy_spatial_model,
                                      proxy_quadrature_order)
            frequency, frequency_state, raw_frequency = closed_form_frequency(
                proxy, model["gamma"], f0_ghz, fmin_ghz, tsafe, ambient
            )
            floor_scale = model["gamma"] + (
                1.0 - model["gamma"]
            ) * (fmin_ghz / f0_ghz)
            tmax_at_floor = ambient + (proxy - ambient) * floor_scale
            mean_wire, wire = selected_wire_cycles(modules)
            objective_wire = (
                wire if wire_objective == "continuous"
                else round_wire_cycles(wire, wire_rounding)
            )
            candidate = {"tier": tier, "start": start_name, "x_mm": point[0],
                         "y_mm": point[1], "loss": value, "evaluations": evaluations,
                         "collision_mm2": collision, "proxy_tmax_c": proxy,
                         "proxy_frequency_ghz": frequency,
                         "proxy_unclamped_frequency_ghz": raw_frequency,
                         "proxy_frequency_state": frequency_state,
                         "proxy_tmax_at_fmin_c": tmax_at_floor,
                         "proxy_thermal_feasible_at_fmin": tmax_at_floor <= tsafe,
                         "mean_wire_cycles": mean_wire,
                         "wire_aggregation": wire_aggregation,
                         "wire_objective_cycles": objective_wire}
            if wire_aggregation == "traffic-weighted":
                candidate["traffic_weighted_wire_cycles"] = wire
            candidates.append(candidate)
    legal = [candidate for candidate in candidates if candidate["collision_mm2"] <= 1e-8]
    if not legal:
        raise RuntimeError("layout optimizer found no non-overlapping L2 placement")
    best = min(legal, key=lambda item: item["loss"])
    optimized = dict(base)
    optimized["policy"] = "CLIP-3D equation (15) optimized"
    optimized["modules"] = fixed + [dict(original, tier=best["tier"],
                                          x_mm=best["x_mm"], y_mm=best["y_mm"])]
    check_geometry(optimized["modules"], side)
    optimized_delays = derive_layout_delays(
        optimized, f0_ghz, wire_rounding, communication_weights
    )
    write_json(output_layout, optimized)
    total_power = float(model["totals"]["total_power_w"])
    movable_power = float(original["total_power_w"])
    movable_fraction = movable_power / total_power if total_power > 0 else 0.0
    mean_cycle_changed = (
        baseline_delays["wire_cycles"] != optimized_delays["wire_cycles"]
    )
    maximum_cycle_changed = (
        baseline_delays["maximum_wire_cycles"]
        != optimized_delays["maximum_wire_cycles"]
    )
    selected_cycle_field = (
        "traffic_weighted_wire_cycles"
        if wire_aggregation == "traffic-weighted" else "wire_cycles"
    )
    selected_cycle_changed = (
        baseline_delays[selected_cycle_field] != optimized_delays[selected_cycle_field]
    )
    cautions = []
    if movable_fraction < 0.05:
        cautions.append(
            "Movable L2 power is below 5% of total power; large thermal gains "
            "are not supported unless the fixed high-power blocks or inputs change."
        )
    if not selected_cycle_changed:
        cautions.append(
            f"Continuous {wire_aggregation} wire delay changed without changing "
            "the rounded cycle used by gem5 R2."
        )
    if maximum_cycle_changed and not mean_cycle_changed:
        cautions.append(
            "The rounded worst path changed although the rounded mean did not; "
            "maximum-path R2 is a separate conservative experiment, not paper mean mode."
        )
    report = {
        "schema_version": 1, "solver": solver, "selected": best,
        "candidates": candidates,
        "parameters": {"ambient_c": ambient, "r_convec_k_per_w": r_convec,
                       "alpha": alpha, "beta": beta,
                       "cross_tier_weight": cross_tier_weight,
                       "lambda_wire": lambda_wire, "f0_ghz": f0_ghz,
                       "fmin_ghz": fmin_ghz, "tsafe_c": tsafe,
                       "allowed_l2_tiers": list(tiers),
                       "proxy_spatial_model": proxy_spatial_model,
                       "proxy_quadrature_order": proxy_quadrature_order,
                       "wire_objective": wire_objective,
                       "wire_aggregation": wire_aggregation,
                       "wire_rounding": wire_rounding,
                       "wire_r_ohm_per_mm": 50.0, "wire_c_f_per_mm": 200e-15},
        "baseline": {
            "policy": base.get("policy", "fixed-bin"),
            "proxy_tmax_c": baseline_proxy,
            "proxy_frequency_ghz": baseline_frequency,
            "layout_delays": baseline_delays,
        },
        "predicted_deltas": {
            "proxy_tmax_c": best["proxy_tmax_c"] - baseline_proxy,
            "proxy_frequency_ghz": best["proxy_frequency_ghz"] - baseline_frequency,
            "mean_wire_cycles_unrounded": (
                optimized_delays["wire_cycles_unrounded"]
                - baseline_delays["wire_cycles_unrounded"]
            ),
            "mean_wire_cycles_rounded": (
                optimized_delays["wire_cycles"] - baseline_delays["wire_cycles"]
            ),
            "maximum_wire_cycles_rounded": (
                optimized_delays["maximum_wire_cycles"]
                - baseline_delays["maximum_wire_cycles"]
            ),
        },
        "observability_diagnostics": {
            "movable_l2_power_w": movable_power,
            "total_power_w": total_power,
            "movable_l2_power_fraction": movable_fraction,
            "paper_mean_r2_cycle_changed": mean_cycle_changed,
            "conservative_maximum_r2_cycle_changed": maximum_cycle_changed,
            "selected_r2_cycle_changed": selected_cycle_changed,
            "optimized_layout_delays": optimized_delays,
            "cautions": cautions,
        },
        "warning": "proxy_tmax_c guides search only; run HotSpot on output_layout for reportable Tmax.",
    }
    if wire_aggregation == "traffic-weighted":
        report["predicted_deltas"].update({
            "traffic_weighted_wire_cycles_unrounded": (
                optimized_delays["traffic_weighted_wire_cycles_unrounded"]
                - baseline_delays["traffic_weighted_wire_cycles_unrounded"]
            ),
            "traffic_weighted_wire_cycles_rounded": (
                optimized_delays["traffic_weighted_wire_cycles"]
                - baseline_delays["traffic_weighted_wire_cycles"]
            ),
        })
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--output-layout", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--utilization", type=float, default=0.70)
    parser.add_argument("--ambient-c", type=float, default=25.0)
    parser.add_argument("--r-convec", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--cross-tier-weight", type=float, default=0.65)
    parser.add_argument("--lambda-wire", type=float, default=0.02)
    parser.add_argument("--require-scipy", action="store_true")
    parser.add_argument(
        "--allowed-l2-tier", action="append", type=int, choices=(0, 1),
        help="repeatable; omit to enumerate both tiers",
    )
    parser.add_argument(
        "--proxy-spatial-model", choices=("center", "area-quadrature"),
        default="center",
    )
    parser.add_argument("--proxy-quadrature-order", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        "--wire-objective", choices=("continuous", "r2-quantized"),
        default="continuous",
    )
    parser.add_argument("--wire-rounding", choices=("nearest", "ceil", "floor"), default="nearest")
    parser.add_argument(
        "--wire-aggregation", choices=("mean", "traffic-weighted"), default="mean"
    )
    args = parser.parse_args()
    result = optimize(args.modules, args.output_layout, args.report, args.utilization,
                      args.ambient_c, args.r_convec, args.alpha, args.beta,
                      args.cross_tier_weight, lambda_wire=args.lambda_wire,
                      require_scipy=args.require_scipy,
                      allowed_l2_tiers=args.allowed_l2_tier,
                      proxy_spatial_model=args.proxy_spatial_model,
                      proxy_quadrature_order=args.proxy_quadrature_order,
                      wire_objective=args.wire_objective,
                      wire_rounding=args.wire_rounding,
                      wire_aggregation=args.wire_aggregation)
    selected = result["selected"]
    print(f"Selected tier={selected['tier']} x={selected['x_mm']:.4f} y={selected['y_mm']:.4f} mm")


if __name__ == "__main__":
    main()
