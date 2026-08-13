#!/usr/bin/env python3
"""Identify an unscaled strict-P1 thermal proxy from local HotSpot evidence."""

from __future__ import annotations

import math
import hashlib
import json
import copy
import random
import statistics
import argparse
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import (
    baseline_layout,
    check_geometry,
    materialize,
)
from workflow.floorplan.optimize_layout import collision_area
from workflow.floorplan.optimize_layout import spatial_coupling
from workflow.thermal.run_hotspot import DEFAULT_HOTSPOT, run_hotspot


PLACEMENT_LABELS = (
    "center",
    "corner_ll", "corner_lr", "corner_ul", "corner_ur",
    "edge_bottom", "edge_top", "edge_left", "edge_right",
    "near_core0", "near_core1", "near_core2", "near_core3",
)


def validate_identification_config(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported alpha/Lc identification config schema")
    physical = config.get("physical") or {}
    stack = physical.get("thermal_stack") or {}
    if float(stack.get("local_resistance_scale", math.nan)) != 1.0:
        raise ValueError("alpha/Lc identification requires local_resistance_scale == 1.0")
    if int(config.get("placements_per_point", 0)) != len(PLACEMENT_LABELS):
        raise ValueError(f"formal campaign requires {len(PLACEMENT_LABELS)} placements per point")
    if not 0 < float(physical.get("utilization", 0)) <= 1:
        raise ValueError("physical utilization must be in (0, 1]")
    if int(physical.get("grid_size", 0)) < 2:
        raise ValueError("physical grid_size must be at least 2")
    expected_workloads = {"fft", "cholesky", "matmul", "stencil", "stream"}
    if set(config.get("workloads", [])) != expected_workloads:
        raise ValueError("formal campaign requires all five declared workloads")


def validate_model_contract(model: dict) -> None:
    if (model.get("area_provenance") or {}).get("global_scaling") != "none":
        raise ValueError("alpha/Lc evidence must use an unscaled physical model")
    provenance = model.get("power_provenance") or {}
    expected = {
        "dynamic": "McPAT Runtime Dynamic",
        "leakage": "McPAT Subthreshold Leakage + Gate Leakage",
        "postprocessing": "none",
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("alpha/Lc evidence requires raw McPAT power with no postprocessing")
    if not model.get("cacti_characterization_id"):
        raise ValueError("model lacks local CACTI characterization identity")


def identification_plan(r1_root: Path, config: dict) -> dict:
    """Inventory existing R1 without starting or modifying any simulation."""
    validate_identification_config(config)
    root = Path(r1_root).resolve()
    work_points = []
    missing = []
    for workload in config["workloads"]:
        for l1d in config["l1d_sizes"]:
            for l2 in config["l2_sizes"]:
                point = root / workload / f"l1d_{l1d}" / f"l2_{l2}"
                required = (point / "r1_metadata.json", point / "stats.txt")
                if not all(path.is_file() for path in required):
                    missing.append(str(point))
                work_points.append({
                    "workload": workload, "l1d_size": l1d, "l2_size": l2,
                    "r1_dir": str(point),
                })
    if missing:
        raise FileNotFoundError(
            f"formal campaign requires completed R1 for all 45 points; missing {len(missing)}: "
            + ", ".join(missing[:3])
        )
    return {
        "schema_version": 1,
        "r1_root": str(root),
        "work_point_count": len(work_points),
        "placements_per_point": int(config["placements_per_point"]),
        "intended_real_power_cases": (
            len(work_points) * int(config["placements_per_point"])
        ),
        "work_points": work_points,
        "r1_policy": "read-only reuse; no R1 command exists in this workflow",
    }


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


def unit_power_model(model: dict, source_name: str,
                     source_power_w: float = 1.0) -> dict:
    """Return an isolated model with exactly one unit-power source."""
    source_power_w = float(source_power_w)
    if not math.isfinite(source_power_w) or source_power_w <= 0:
        raise ValueError("unit source power must be finite and positive")
    matches = [module for module in model.get("modules", [])
               if module.get("name") == source_name]
    if len(matches) != 1:
        raise ValueError("unit-power stimulus requires exactly one named source")
    result = copy.deepcopy(model)
    for module in result["modules"]:
        active = module["name"] == source_name
        module["dynamic_power_w"] = source_power_w if active else 0.0
        module["leakage_power_w"] = 0.0
        module["total_power_w"] = source_power_w if active else 0.0
        if "area_mm2" in module and float(module["area_mm2"]) > 0:
            module["power_density_w_per_mm2"] = (
                module["total_power_w"] / float(module["area_mm2"])
            )
    result["unit_power_stimulus"] = {
        "source_name": source_name,
        "source_power_w": source_power_w,
        "purpose": "HotSpot spatial system identification only",
    }
    return result


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires non-empty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def estimate_cross_tier_weight(matched_responses: list[dict],
                               bootstrap_samples: int = 1000,
                               seed: int = 20260813,
                               max_relative_interval_width: float = 0.50) -> dict:
    """Estimate scalar cross-tier coupling from matched one-watt rises."""
    if not matched_responses:
        raise ValueError("cross-tier estimation requires matched responses")
    if bootstrap_samples < 20:
        raise ValueError("cross-tier bootstrap requires at least 20 samples")
    ratios = []
    records = []
    for case in matched_responses:
        ambient = float(case["ambient_c"])
        same_rise = float(case["same_tier_tmax_c"]) - ambient
        cross_rise = float(case["cross_tier_tmax_c"]) - ambient
        if not math.isfinite(same_rise) or same_rise <= 0:
            raise ValueError("same-tier temperature rise must be finite and positive")
        if not math.isfinite(cross_rise) or cross_rise < 0:
            raise ValueError("cross-tier temperature rise must be finite and nonnegative")
        ratio = cross_rise / same_rise
        ratios.append(ratio)
        records.append({
            "label": str(case["label"]), "same_tier_rise_c": same_rise,
            "cross_tier_rise_c": cross_rise, "ratio": ratio,
        })
    estimate = statistics.median(ratios)
    rng = random.Random(seed)
    bootstrap = [
        statistics.median(rng.choices(ratios, k=len(ratios)))
        for _ in range(bootstrap_samples)
    ]
    low, high = _percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)
    relative_width = (high - low) / estimate if estimate > 0 else math.inf
    reasons = []
    if not all(math.isfinite(value) for value in (estimate, low, high)):
        reasons.append("bootstrap interval is not finite")
    if relative_width > max_relative_interval_width:
        reasons.append(
            "relative interval width exceeds the predeclared stability ceiling"
        )
    return {
        "schema_version": 1,
        "method": "median matched unit-power rise ratio with whole-case bootstrap",
        "estimate": estimate,
        "confidence_interval_95": {"low": low, "high": high},
        "relative_interval_width": relative_width,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "per_case_ratios": records,
        "accepted": not reasons,
        "rejection_reasons": reasons,
    }


def _feature_rows(samples: list[dict], cross_tier_weight: float,
                  lc_ratio: float) -> list[tuple[dict, float, float]]:
    reference: dict[str, float] = {}
    raw = []
    for sample in samples:
        group = str(sample["group"])
        side = float(sample["die_side_mm"])
        feature = spatial_coupling(
            sample["modules"], side, cross_tier_weight,
            spatial_model="area-quadrature", quadrature_order=2,
            lc_mm=lc_ratio * side,
        )
        raw.append((sample, feature))
        if bool(sample.get("is_reference")):
            if group in reference:
                raise ValueError(f"multiple reference layouts in group {group}")
            reference[group] = feature
    groups = {str(sample["group"]) for sample in samples}
    if set(reference) != groups:
        raise ValueError("every work-point group requires exactly one reference layout")
    return [
        (sample, feature - reference[str(sample["group"])],
         float(sample["delta_t_c"]))
        for sample, feature in raw
    ]


def _huber_loss(residual: float, delta: float) -> float:
    magnitude = abs(residual)
    if magnitude <= delta:
        return 0.5 * residual * residual
    return delta * (magnitude - 0.5 * delta)


def _robust_nonnegative_slope(rows: list[tuple[dict, float, float]],
                              huber_delta_c: float) -> tuple[float, float]:
    denominator = sum(feature * feature for _, feature, _ in rows)
    if denominator <= 1e-24:
        raise ValueError("thermal samples are spatially degenerate")
    alpha = max(0.0, sum(feature * target for _, feature, target in rows) / denominator)
    for _ in range(100):
        numerator = 0.0
        weighted_denominator = 0.0
        for _, feature, target in rows:
            residual = alpha * feature - target
            weight = 1.0 if abs(residual) <= huber_delta_c else (
                huber_delta_c / abs(residual)
            )
            numerator += weight * feature * target
            weighted_denominator += weight * feature * feature
        if weighted_denominator <= 1e-24:
            raise ValueError("thermal samples are spatially degenerate")
        updated = max(0.0, numerator / weighted_denominator)
        if abs(updated - alpha) <= 1e-12 * max(1.0, alpha):
            alpha = updated
            break
        alpha = updated
    objective = sum(
        _huber_loss(alpha * feature - target, huber_delta_c)
        for _, feature, target in rows
    )
    return alpha, objective


def _fit_core(samples: list[dict], cross_tier_weight: float,
              lc_bounds_ratio: tuple[float, float],
              huber_delta_c: float) -> dict:
    low, high = map(float, lc_bounds_ratio)
    if not 0 < low < high:
        raise ValueError("Lc ratio bounds must be finite, positive, and increasing")

    def evaluate(log_ratio: float) -> tuple[float, float, list]:
        ratio = math.exp(log_ratio)
        rows = _feature_rows(samples, cross_tier_weight, ratio)
        alpha, objective = _robust_nonnegative_slope(rows, huber_delta_c)
        return objective, alpha, rows

    log_low, log_high = math.log(low), math.log(high)
    scan = []
    for index in range(161):
        log_ratio = log_low + (log_high - log_low) * index / 160
        scan.append((evaluate(log_ratio)[0], log_ratio))
    _, best_log = min(scan)
    step = (log_high - log_low) / 160
    left, right = max(log_low, best_log - step), min(log_high, best_log + step)
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1, f2 = evaluate(x1)[0], evaluate(x2)[0]
    for _ in range(80):
        if right - left < 1e-10:
            break
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = evaluate(x1)[0]
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = evaluate(x2)[0]
    best_log = (left + right) / 2.0
    objective, alpha, rows = evaluate(best_log)
    ratio = math.exp(best_log)
    on_boundary = (
        ratio <= low * (1.0 + 1e-5) or ratio >= high * (1.0 - 1e-5)
    )
    return {
        "alpha": alpha, "lc_die_side_ratio": ratio,
        "objective": objective, "rows": rows, "on_boundary": on_boundary,
    }


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[order[offset]] = rank
        position = end
    return ranks


def _spearman(actual: list[float], predicted: list[float]) -> float | None:
    if len(actual) < 2:
        return None
    left, right = _average_ranks(actual), _average_ranks(predicted)
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator > 0 else None


def _prediction_metrics(rows: list[tuple[dict, float, float]],
                        alpha: float) -> dict:
    actual = [target for _, _, target in rows]
    predicted = [alpha * feature for _, feature, _ in rows]
    errors = [prediction - target for prediction, target in zip(predicted, actual)]
    groups: dict[str, list[int]] = {}
    for index, (sample, _, _) in enumerate(rows):
        groups.setdefault(str(sample["group"]), []).append(index)
    regrets = []
    per_group_spearman = {}
    for group, indices in groups.items():
        group_actual = [actual[index] for index in indices]
        group_predicted = [predicted[index] for index in indices]
        per_group_spearman[group] = _spearman(group_actual, group_predicted)
        selected = min(indices, key=lambda index: (predicted[index], index))
        regrets.append(actual[selected] - min(group_actual))
    return {
        "count": len(rows),
        "mae_c": statistics.mean(abs(error) for error in errors),
        "rmse_c": math.sqrt(statistics.mean(error * error for error in errors)),
        "spearman": _spearman(actual, predicted),
        "per_group_spearman": per_group_spearman,
        "median_selection_regret_c": statistics.median(regrets),
        "p95_selection_regret_c": _percentile(regrets, 0.95),
    }


def _jacobian_diagnostics(samples: list[dict], cross_tier_weight: float,
                          alpha: float, ratio: float) -> dict:
    rows = _feature_rows(samples, cross_tier_weight, ratio)
    epsilon = 1e-4
    plus = _feature_rows(samples, cross_tier_weight, ratio * math.exp(epsilon))
    minus = _feature_rows(samples, cross_tier_weight, ratio * math.exp(-epsilon))
    columns = [
        (row[1], alpha * (p[1] - m[1]) / (2.0 * epsilon))
        for row, p, m in zip(rows, plus, minus)
    ]
    aa = sum(a * a for a, _ in columns)
    ab = sum(a * b for a, b in columns)
    bb = sum(b * b for _, b in columns)
    trace, determinant = aa + bb, aa * bb - ab * ab
    discriminant = max(trace * trace - 4.0 * determinant, 0.0)
    eigen_high = (trace + math.sqrt(discriminant)) / 2.0
    eigen_low = (trace - math.sqrt(discriminant)) / 2.0
    rank = int(eigen_high > 1e-16) + int(eigen_low > eigen_high * 1e-12)
    condition = math.sqrt(eigen_high / eigen_low) if eigen_low > 0 else math.inf
    return {"jacobian_rank": rank, "jacobian_condition": condition}


def fit_alpha_lc(samples: list[dict], cross_tier_weight: float,
                 lc_bounds_ratio: tuple[float, float] = (0.02, 4.0),
                 huber_delta_c: float = 0.5,
                 bootstrap_samples: int = 1000,
                 seed: int = 20260813) -> dict:
    """Fit alpha and Lc/die-side with whole-work-point validation evidence."""
    if bootstrap_samples < 20:
        raise ValueError("alpha/Lc bootstrap requires at least 20 samples")
    fit = _fit_core(samples, cross_tier_weight, lc_bounds_ratio, huber_delta_c)
    metrics = _prediction_metrics(fit["rows"], fit["alpha"])
    diagnostics = _jacobian_diagnostics(
        samples, cross_tier_weight, fit["alpha"], fit["lc_die_side_ratio"]
    )
    if diagnostics["jacobian_rank"] < 2:
        raise ValueError("thermal samples are spatially degenerate")

    workloads = sorted({str(sample["workload"]) for sample in samples})
    leave_one_out = {}
    for workload in workloads:
        train = [sample for sample in samples if str(sample["workload"]) != workload]
        validation = [sample for sample in samples if str(sample["workload"]) == workload]
        local = _fit_core(train, cross_tier_weight, lc_bounds_ratio, huber_delta_c)
        rows = _feature_rows(
            validation, cross_tier_weight, local["lc_die_side_ratio"]
        )
        leave_one_out[workload] = {
            "train_parameters": {
                "alpha": local["alpha"],
                "lc_die_side_ratio": local["lc_die_side_ratio"],
            },
            "validation_metrics": _prediction_metrics(rows, local["alpha"]),
        }

    groups: dict[str, list[dict]] = {}
    for sample in samples:
        groups.setdefault(str(sample["group"]), []).append(sample)
    group_names = sorted(groups)
    rng = random.Random(seed)
    boot_alpha, boot_ratio = [], []
    for _ in range(bootstrap_samples):
        chosen = rng.choices(group_names, k=len(group_names))
        resampled = []
        for occurrence, group in enumerate(chosen):
            for sample in groups[group]:
                duplicate = dict(sample)
                duplicate["group"] = f"{group}#bootstrap{occurrence}"
                resampled.append(duplicate)
        local = _fit_core(
            resampled, cross_tier_weight, lc_bounds_ratio, huber_delta_c
        )
        boot_alpha.append(local["alpha"])
        boot_ratio.append(local["lc_die_side_ratio"])
    bootstrap = {
        "samples": bootstrap_samples, "seed": seed,
        "alpha_95": {
            "low": _percentile(boot_alpha, 0.025),
            "high": _percentile(boot_alpha, 0.975),
        },
        "lc_die_side_ratio_95": {
            "low": _percentile(boot_ratio, 0.025),
            "high": _percentile(boot_ratio, 0.975),
        },
    }
    return {
        "schema_version": 1,
        "method": "nested log-Lc search and nonnegative Huber slope",
        "parameters": {
            "alpha": fit["alpha"],
            "lc_die_side_ratio": fit["lc_die_side_ratio"],
            "cross_tier_weight": float(cross_tier_weight),
        },
        "objective": fit["objective"],
        "metrics": metrics,
        "diagnostics": {**diagnostics, "parameter_on_boundary": fit["on_boundary"]},
        "cross_validation": {"leave_one_workload_out": leave_one_out},
        "bootstrap": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser(
        "plan", help="inventory the read-only 45-point R1 campaign"
    )
    plan_parser.add_argument("--r1-root", type=Path, required=True)
    plan_parser.add_argument("--config", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        config = read_json(args.config)
        report = identification_plan(args.r1_root, config)
        write_json(args.output, report)
        print(
            f"planned {report['work_point_count']} work points and "
            f"{report['intended_real_power_cases']} real-power HotSpot cases"
        )


if __name__ == "__main__":
    main()


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
