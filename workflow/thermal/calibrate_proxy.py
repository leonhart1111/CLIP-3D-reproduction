#!/usr/bin/env python3
"""Calibrate the equation-(14) thermal proxy against detailed-3D HotSpot.

This is a reproduction calibration utility, not an implementation detail
published by the CLIP-3D authors.  It moves the shared L2 over a deterministic
normalized grid, runs the same HotSpot stack used by the lifting pipeline, and
fits only the identifiable proxy parameters with a held-out split.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from workflow.common import (
    PROJECT_ROOT,
    format_temperature_csv_row,
    read_json,
    write_json,
)
from workflow.floorplan.generate_hotspot_inputs import (
    baseline_layout,
    check_geometry,
    materialize,
)
from workflow.floorplan.optimize_layout import collision_area, proxy_temperature
from workflow.thermal.run_hotspot import DEFAULT_HOTSPOT, run_hotspot


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/clip3d_constrained_5p0.json"


def parse_model(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    label, raw_path = text.split("=", 1)
    if not label or any(character in label for character in "/\\"):
        raise argparse.ArgumentTypeError("model label must be a non-empty path component")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"model does not exist: {path}")
    return label, path


def parse_external_case(text: str) -> tuple[str, str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "external case must be GROUP:SAMPLE=CASE_DIR"
        )
    label, raw_path = text.split("=", 1)
    if ":" in label:
        group, sample_label = label.split(":", 1)
    else:
        # Backward compatible, but a single case in its own group cannot
        # validate a spatial ordering.  Use GROUP:SAMPLE for new runs.
        group = sample_label = label
    path = Path(raw_path).expanduser().resolve()
    required = ("layout.json", "thermal_result.json", "hotspot_manifest.json")
    missing = [name for name in required if not (path / name).is_file()]
    if not group or not sample_label or missing:
        raise argparse.ArgumentTypeError(
            f"invalid external case {path}; missing: {', '.join(missing) or 'label'}"
        )
    return group, sample_label, path


def parse_l2_tiers(text: str) -> tuple[int, ...]:
    try:
        tiers = tuple(int(value) for value in text.split(",") if value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("tiers must be comma-separated integers") from error
    if not tiers or any(tier not in (0, 1) for tier in tiers) or len(set(tiers)) != len(tiers):
        raise argparse.ArgumentTypeError("tiers must be a non-empty subset of 0,1")
    return tiers


def candidate_layouts(model_path: Path, grid_points: int, utilization: float,
                      allowed_l2_tiers: tuple[int, ...] = (0, 1)) -> list[dict]:
    """Return legal L2 placements restricted to the requested physical tiers."""
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    if not allowed_l2_tiers:
        raise ValueError("allowed_l2_tiers must not be empty")
    if any(type(tier) is not int or tier not in (0, 1) for tier in allowed_l2_tiers):
        raise ValueError("allowed_l2_tiers must contain only 0 and/or 1")
    if len(set(allowed_l2_tiers)) != len(allowed_l2_tiers):
        raise ValueError("allowed_l2_tiers must not contain duplicates")
    model = read_json(model_path)
    base = baseline_layout(model, utilization)
    original = next(module for module in base["modules"] if module["kind"] == "l2")
    fixed = [dict(module) for module in base["modules"] if module["kind"] != "l2"]
    side = float(base["die_width_mm"])
    upper_x = side - float(original["width_mm"])
    upper_y = side - float(original["height_mm"])
    if min(upper_x, upper_y) < 0:
        raise ValueError(f"L2 does not fit in die for {model_path}")

    layouts = []
    for tier in allowed_l2_tiers:
        for row in range(grid_points):
            fy = row / (grid_points - 1)
            for column in range(grid_points):
                fx = column / (grid_points - 1)
                l2 = dict(original, tier=tier, x_mm=fx * upper_x, y_mm=fy * upper_y)
                if collision_area(l2, fixed) > 1e-9:
                    continue
                modules = fixed + [l2]
                check_geometry(modules, side)
                layout = dict(base)
                layout["policy"] = "thermal-proxy calibration grid"
                layout["modules"] = modules
                layouts.append({
                    "tier": tier,
                    "row": row,
                    "column": column,
                    "fx": fx,
                    "fy": fy,
                    "layout": layout,
                })
    if not layouts:
        raise RuntimeError(f"no legal calibration layouts for {model_path}")
    return layouts


def proxy_acceptance_checks(validation: dict, baseline: dict, fitted_rank: int,
                            selected_weight: float, beta_status: str) -> dict:
    """Return the strict-P1 checks without concealing a failed condition."""
    return {
        "lower_validation_rmse": validation["rmse_c"] < baseline["rmse_c"],
        "lower_validation_spatial_rmse": (
            validation["spatial_centered_rmse_c"]
            < baseline["spatial_centered_rmse_c"]
        ),
        "validation_spatial_rank_at_least_0p8": (
            validation["spatial_spearman"] is not None
            and validation["spatial_spearman"] >= 0.8
        ),
        "cross_weight_not_on_bound": 0.0 < selected_weight < 1.0,
        "active_parameter_jacobian_full_rank": fitted_rank == 2,
        "beta_policy_valid": beta_status == "fixed_unidentifiable_under_p1",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_raw_power_provenance(model_path: Path) -> None:
    provenance = read_json(model_path).get("power_provenance")
    required = {
        "dynamic": "McPAT Runtime Dynamic",
        "leakage": "McPAT Subthreshold Leakage + Gate Leakage",
        "postprocessing": "none",
    }
    if not isinstance(provenance, dict) or any(
            provenance.get(key) != value for key, value in required.items()):
        raise ValueError(
            f"model must use raw direct-McPAT power with no postprocessing: {model_path}"
        )


def sample_split(model_index: int, row: int, column: int, tier: int,
                 grid_points: int = 3) -> str:
    # Hold out the lower-left anchor and center.  This guarantees at least two
    # spatially distinct validation points per model for the default 3x3 scan;
    # leave-one-model-out below supplies the stricter cross-model check.
    del model_index, tier
    middle = (grid_points - 1) // 2
    return "validation" if (row, column) in ((0, 0), (middle, middle)) else "train"


def run_one(sample: dict, config: dict, hotspot: Path, force: bool) -> dict:
    case_dir = Path(sample["case_dir"])
    result_path = case_dir / "thermal_result.json"
    metadata_path = case_dir / "calibration_sample.json"
    signature = {
        "schema_version": 1,
        "model": sample["model"],
        "model_label": sample["model_label"],
        "tier": sample["tier"],
        "row": sample["row"],
        "column": sample["column"],
        "fx": sample["fx"],
        "fy": sample["fy"],
        "config": config,
    }
    reusable = False
    if result_path.is_file() and metadata_path.is_file() and not force:
        reusable = read_json(metadata_path) == signature
    if not reusable:
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True)
        layout_path = case_dir / "candidate_layout.json"
        write_json(layout_path, sample["layout"])
        physical = config["physical"]
        frequency = config["frequency"]
        materialize(
            Path(sample["model"]), case_dir,
            int(physical["grid_size"]), float(physical["utilization"]),
            float(frequency["ambient_c"]), float(physical["r_convec_k_per_w"]),
            layout_path, physical.get("thermal_stack"),
        )
        write_json(metadata_path, signature)
        thermal = run_hotspot(case_dir, hotspot)
    else:
        thermal = read_json(result_path)
    result = {key: value for key, value in sample.items() if key != "layout"}
    result.update({
        "tmax_c": float(thermal["tmax_c"]),
        "peak_unit": thermal["peak_unit"],
        "reused": reusable,
    })
    return result


def proxy_prediction(sample: dict, parameters: list[float], config: dict) -> float:
    alpha, beta, cross_tier_weight = map(float, parameters)
    layout = read_json(Path(sample["case_dir"]) / "layout.json")
    physical = config["physical"]
    frequency = config["frequency"]
    return proxy_temperature(
        layout["modules"], float(layout["die_width_mm"]),
        float(frequency["ambient_c"]), float(physical["r_convec_k_per_w"]),
        alpha, beta, cross_tier_weight,
        config.get("layout_optimizer", {}).get("proxy_spatial_model", "center"),
        int(config.get("layout_optimizer", {}).get("proxy_quadrature_order", 2)),
    )


def metrics(samples: list[dict], parameters: list[float], config: dict) -> dict:
    import numpy as np
    from scipy.stats import kendalltau, spearmanr

    actual = np.asarray([sample["tmax_c"] for sample in samples], dtype=float)
    predicted = np.asarray(
        [proxy_prediction(sample, parameters, config) for sample in samples], dtype=float
    )
    errors = predicted - actual
    centered_actual = []
    centered_predicted = []
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample["model_label"], []).append(index)
    per_model = {}
    for label, indices in groups.items():
        a = actual[indices]
        p = predicted[indices]
        centered_actual.extend(a - np.mean(a))
        centered_predicted.extend(p - np.mean(p))
        per_model[label] = {
            "n": len(indices),
            "actual_range_c": float(np.ptp(a)),
            "predicted_range_c": float(np.ptp(p)),
            "rmse_c": float(np.sqrt(np.mean((p - a) ** 2))),
            "mae_c": float(np.mean(np.abs(p - a))),
        }
    centered_actual_array = np.asarray(centered_actual, dtype=float)
    centered_predicted_array = np.asarray(centered_predicted, dtype=float)
    centered_errors = centered_predicted_array - centered_actual_array
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - float(np.sum(errors ** 2)) / denominator if denominator > 0 else 0.0

    def finite_correlation(result) -> float | None:
        value = float(result.statistic)
        return value if math.isfinite(value) else None

    def safe_correlation(function, first, second) -> float | None:
        if len(first) < 2 or float(np.ptp(first)) == 0.0 or float(np.ptp(second)) == 0.0:
            return None
        return finite_correlation(function(first, second))

    return {
        "n": len(samples),
        "rmse_c": float(np.sqrt(np.mean(errors ** 2))),
        "mae_c": float(np.mean(np.abs(errors))),
        "max_abs_error_c": float(np.max(np.abs(errors))),
        "mean_error_c": float(np.mean(errors)),
        "r2": r2,
        "spearman": safe_correlation(spearmanr, actual, predicted),
        "kendall_tau": safe_correlation(kendalltau, actual, predicted),
        "spatial_centered_rmse_c": float(np.sqrt(np.mean(centered_errors ** 2))),
        "spatial_spearman": safe_correlation(
            spearmanr, centered_actual_array, centered_predicted_array
        ),
        "per_model": per_model,
    }


def fit(samples: list[dict], config: dict, starts: list[list[float]] | None = None,
        spatial_weight: float = 0.0,
        fixed_beta: float | None = None,
        fixed_cross_tier_weight: float | None = None) -> dict:
    import numpy as np
    from scipy.optimize import least_squares

    if not samples:
        raise ValueError("cannot fit an empty sample set")
    default = config["layout_optimizer"]
    if fixed_cross_tier_weight is not None and not 0 <= fixed_cross_tier_weight <= 1:
        raise ValueError("fixed_cross_tier_weight must be in [0, 1]")
    if fixed_beta is not None and not 0 <= fixed_beta <= 20:
        raise ValueError("fixed_beta must be in [0, 20]")
    active_parameters = ["alpha"]
    if fixed_beta is None:
        active_parameters.append("beta")
    if fixed_cross_tier_weight is None:
        active_parameters.append("cross_tier_weight")
    if starts is None:
        initial = {
            "alpha": float(default["alpha"]),
            "beta": float(default["beta"]),
            "cross_tier_weight": float(default["cross_tier_weight"]),
        }
        trial_values = {
            "alpha": (0.1, 0.5, 1.0, 2.0),
            "beta": (0.0, 0.1, 0.5, 1.0),
            "cross_tier_weight": (0.0, 0.25, 0.5, 0.75, 1.0),
        }
        starts = [[initial[name] for name in active_parameters]]
        for index in range(max(len(values) for values in trial_values.values())):
            starts.append([
                trial_values[name][index % len(trial_values[name])]
                for name in active_parameters
            ])

    if spatial_weight < 0:
        raise ValueError("spatial_weight must be non-negative")
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample["model_label"], []).append(index)

    def expanded(parameters):
        values = dict(zip(active_parameters, parameters))
        return [
            float(values["alpha"]),
            float(fixed_beta if fixed_beta is not None else values["beta"]),
            float(
                fixed_cross_tier_weight if fixed_cross_tier_weight is not None
                else values["cross_tier_weight"]
            ),
        ]

    def residuals(parameters):
        proxy_parameters = expanded(parameters)
        absolute = np.asarray([
            proxy_prediction(sample, proxy_parameters, config) - float(sample["tmax_c"])
            for sample in samples
        ])
        if spatial_weight == 0:
            return absolute
        centered = np.concatenate([
            absolute[indices] - np.mean(absolute[indices])
            for indices in groups.values() if len(indices) > 1
        ])
        return np.concatenate((absolute, spatial_weight * centered))

    solutions = []
    lower = [0.0 for _ in active_parameters]
    upper = [1.0 if name == "cross_tier_weight" else 20.0
             for name in active_parameters]
    for start in starts:
        if len(start) != len(active_parameters):
            raise ValueError("each fit start must contain one value per active parameter")
        result = least_squares(
            residuals, start, bounds=(lower, upper),
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000,
        )
        solutions.append(result)
    result = min(solutions, key=lambda item: float(np.sum(item.fun ** 2)))
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    threshold = (max(result.jac.shape) * np.finfo(float).eps * singular_values[0]
                 if len(singular_values) else 0.0)
    effective = singular_values[singular_values > threshold]
    condition = (
        float(effective[0] / effective[-1]) if len(effective) == len(singular_values)
        else math.inf
    )
    degrees = max(len(result.fun) - len(result.x), 1)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * (
        float(np.sum(result.fun ** 2)) / degrees
    )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    proxy_parameters = expanded(result.x)
    standard_error_values = {"alpha": None, "beta": None, "cross_tier_weight": None}
    for index, name in enumerate(active_parameters):
        standard_error_values[name] = float(standard_errors[index])
    beta_status = (
        "fixed_unidentifiable_under_p1" if fixed_beta == 0.0
        else "fixed" if fixed_beta is not None else "fitted"
    )
    return {
        "parameters": {
            "alpha": float(proxy_parameters[0]),
            "beta": float(proxy_parameters[1]),
            "cross_tier_weight": float(proxy_parameters[2]),
        },
        "standard_errors_linearized": standard_error_values,
        "jacobian_singular_values": [float(value) for value in singular_values],
        "jacobian_condition_number": condition,
        "rank": int(np.linalg.matrix_rank(result.jac)),
        "active_parameters": active_parameters,
        "cost_sum_squared_error": float(np.sum(result.fun ** 2)),
        "success": bool(result.success),
        "message": result.message,
        "function_evaluations": int(result.nfev),
        "spatial_residual_weight": spatial_weight,
        "fixed_beta": fixed_beta,
        "beta_status": beta_status,
        "fixed_cross_tier_weight": fixed_cross_tier_weight,
    }


def cross_validate_weight(training: list[dict], validation: list[dict],
                          config: dict, spatial_weight: float,
                          step: float = 0.005,
                          fixed_beta: float | None = None) -> dict:
    if not 0 < step <= 1:
        raise ValueError("cross_weight_step must be in (0, 1]")
    count = int(math.floor(1.0 / step + 1e-12))
    weights = [min(index * step, 1.0) for index in range(count + 1)]
    if weights[-1] < 1.0 - 1e-12:
        weights.append(1.0)
    candidates = []
    for weight in weights:
        fitted = fit(
            training, config, spatial_weight=spatial_weight,
            fixed_beta=fixed_beta,
            fixed_cross_tier_weight=weight,
        )
        vector = list(fitted["parameters"].values())
        validation_metrics = metrics(validation, vector, config)
        score = math.sqrt(
            validation_metrics["rmse_c"] ** 2
            + (spatial_weight * validation_metrics["spatial_centered_rmse_c"]) ** 2
        )
        candidates.append({
            "cross_tier_weight": weight,
            "validation_score": score,
            "validation_rmse_c": validation_metrics["rmse_c"],
            "validation_spatial_centered_rmse_c": validation_metrics[
                "spatial_centered_rmse_c"
            ],
            "validation_spatial_spearman": validation_metrics["spatial_spearman"],
            "training_fit": fitted,
        })
    selected = min(candidates, key=lambda item: (
        item["validation_score"], item["cross_tier_weight"]
    ))
    return {
        "method": "held-out grid search because max_i makes cross-tier response non-smooth",
        "step": step,
        "score": "sqrt(validation_RMSE^2 + (spatial_weight * validation_spatial_RMSE)^2)",
        "selected": selected,
        "candidates": candidates,
    }


def write_samples_csv(path: Path, samples: list[dict], default_parameters: list[float],
                      fitted_parameters: list[float], config: dict) -> None:
    fields = [
        "model_label", "split", "tier", "row", "column", "fx", "fy",
        "x_mm", "y_mm", "tmax_c", "default_proxy_c", "fitted_proxy_c",
        "default_error_c", "fitted_error_c", "peak_unit", "case_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            layout = read_json(Path(sample["case_dir"]) / "layout.json")
            l2 = next(module for module in layout["modules"] if module["kind"] == "l2")
            default_prediction = proxy_prediction(sample, default_parameters, config)
            fitted_prediction = proxy_prediction(sample, fitted_parameters, config)
            writer.writerow(format_temperature_csv_row({
                "model_label": sample["model_label"], "split": sample["split"],
                "tier": sample["tier"], "row": sample["row"],
                "column": sample["column"], "fx": sample["fx"], "fy": sample["fy"],
                "x_mm": l2["x_mm"], "y_mm": l2["y_mm"], "tmax_c": sample["tmax_c"],
                "default_proxy_c": default_prediction, "fitted_proxy_c": fitted_prediction,
                "default_error_c": default_prediction - sample["tmax_c"],
                "fitted_error_c": fitted_prediction - sample["tmax_c"],
                "peak_unit": sample["peak_unit"], "case_dir": sample["case_dir"],
            }))


def calibrate(models: list[tuple[str, Path]], config_path: Path, output_dir: Path,
              grid_points: int = 4, workers: int = 4,
              hotspot: Path = DEFAULT_HOTSPOT, force: bool = False,
              hotspot_grid_size: int | None = None,
              spatial_weight: float = 10.0,
              cross_weight_step: float = 0.005,
              external_cases: list[tuple[str, str, Path]] | None = None,
              allowed_l2_tiers: tuple[int, ...] = (0, 1),
              fixed_beta: float | None = None,
              target_grid_size: int = 32) -> dict:
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    if workers < 1:
        raise ValueError("workers must be positive")
    if target_grid_size < 2:
        raise ValueError("target_grid_size must be at least 2")
    allowed_l2_tiers = tuple(allowed_l2_tiers)
    target_config = read_json(config_path)
    config_strict_p1 = target_config.get("formal_validation", {}).get("strict_p1") is True
    strict_p1 = config_strict_p1 or allowed_l2_tiers == (1,)
    if strict_p1 and allowed_l2_tiers != (1,):
        raise ValueError("strict P1 calibration requires allowed_l2_tiers == (1,)")
    if strict_p1 and grid_points != 3:
        raise ValueError("strict P1 calibration requires grid_points == 3")
    if strict_p1 and fixed_beta != 0.0:
        raise ValueError("strict P1 calibration requires fixed_beta == 0.0")
    if strict_p1 and target_grid_size != 32:
        raise ValueError("strict P1 calibration requires target_grid_size == 32")
    for _, model_path in models:
        require_raw_power_provenance(model_path)
    config = copy.deepcopy(target_config)
    if hotspot_grid_size is not None:
        if hotspot_grid_size < 2:
            raise ValueError("hotspot_grid_size must be at least 2")
        config["physical"]["grid_size"] = hotspot_grid_size
    output_dir = output_dir.resolve()
    cases_root = output_dir / "cases"
    samples = []
    for model_index, (label, model_path) in enumerate(models):
        candidates = candidate_layouts(
            model_path, grid_points, float(config["physical"]["utilization"]),
            allowed_l2_tiers,
        )
        for candidate in candidates:
            identifier = (
                f"tier{candidate['tier']}_r{candidate['row']:02d}_c{candidate['column']:02d}"
            )
            samples.append({
                "model_index": model_index, "model_label": label,
                "model": str(model_path), "tier": candidate["tier"],
                "row": candidate["row"], "column": candidate["column"],
                "fx": candidate["fx"], "fy": candidate["fy"],
                "split": sample_split(model_index, candidate["row"],
                                      candidate["column"], candidate["tier"], grid_points),
                "case_dir": str((cases_root / label / identifier).resolve()),
                "layout": candidate["layout"],
            })

    completed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, sample, config, hotspot, force): sample
            for sample in samples
        }
        total = len(futures)
        for future in as_completed(futures):
            completed.append(future.result())
            count = len(completed)
            if count == total or count % max(workers, 1) == 0:
                print(f"HotSpot calibration cases: {count}/{total}", flush=True)
    completed.sort(key=lambda item: (
        item["model_index"], item["tier"], item["row"], item["column"]
    ))
    training = [sample for sample in completed if sample["split"] == "train"]
    validation = [sample for sample in completed if sample["split"] == "validation"]
    absolute_fit = fit(training, config, fixed_beta=fixed_beta)
    direct_joint_fit = fit(
        training, config, spatial_weight=spatial_weight, fixed_beta=fixed_beta,
    )
    cross_validation = cross_validate_weight(
        training, validation, config, spatial_weight, fixed_beta=fixed_beta,
        step=cross_weight_step,
    )
    selected_training_fit = cross_validation["selected"]["training_fit"]
    selected_weight = float(cross_validation["selected"]["cross_tier_weight"])
    # Refit alpha on every position after the held-out search selects the
    # non-smooth cross-tier response.  Keeping that selected weight fixed
    # makes this report's promotable parameter map match the held-out choice.
    fitted = fit(
        completed, config, spatial_weight=spatial_weight,
        fixed_beta=fixed_beta, fixed_cross_tier_weight=selected_weight,
    )
    parameter_dict = fitted["parameters"]
    parameters = [
        parameter_dict["alpha"], parameter_dict["beta"],
        parameter_dict["cross_tier_weight"],
    ]
    optimizer = config["layout_optimizer"]
    default_parameters = [
        float(optimizer["alpha"]), float(optimizer["beta"]),
        float(optimizer["cross_tier_weight"]),
    ]
    evaluations = {
        "defaults": {
            "parameters": dict(zip(("alpha", "beta", "cross_tier_weight"),
                                   default_parameters)),
            "train": metrics(training, default_parameters, config),
            "validation": metrics(validation, default_parameters, config),
            "all": metrics(completed, default_parameters, config),
        },
        "fitted": {
            "parameters": parameter_dict,
            "train": metrics(training, parameters, config),
            "validation": metrics(validation, parameters, config),
            "all": metrics(completed, parameters, config),
        },
        "cross_validated_training_fit": {
            "parameters": selected_training_fit["parameters"],
            "train": metrics(training, list(selected_training_fit["parameters"].values()), config),
            "validation": metrics(validation, list(selected_training_fit["parameters"].values()), config),
            "all": metrics(completed, list(selected_training_fit["parameters"].values()), config),
        },
        "direct_joint_fit_rejected": {
            "parameters": direct_joint_fit["parameters"],
            "train": metrics(training, list(direct_joint_fit["parameters"].values()), config),
            "validation": metrics(validation, list(direct_joint_fit["parameters"].values()), config),
            "all": metrics(completed, list(direct_joint_fit["parameters"].values()), config),
        },
        "absolute_only_fitted": {
            "parameters": absolute_fit["parameters"],
            "train": metrics(training, list(absolute_fit["parameters"].values()), config),
            "validation": metrics(validation, list(absolute_fit["parameters"].values()), config),
            "all": metrics(completed, list(absolute_fit["parameters"].values()), config),
        },
    }
    external_samples = []
    external_metadata = []
    for group, label, case_dir in external_cases or []:
        thermal = read_json(case_dir / "thermal_result.json")
        manifest = read_json(case_dir / "hotspot_manifest.json")
        if int(manifest.get("grid_size", -1)) != target_grid_size:
            raise ValueError(
                f"external case must use target grid {target_grid_size}: {case_dir}"
            )
        if not math.isclose(
            float(manifest["r_convec_k_per_w"]),
            float(target_config["physical"]["r_convec_k_per_w"]),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"external case has mismatched R_conv: {case_dir}")
        external_samples.append({
            "model_label": group, "sample_label": label,
            "case_dir": str(case_dir),
            "tmax_c": float(thermal["tmax_c"]), "peak_unit": thermal["peak_unit"],
        })
        external_metadata.append({
            "group": group, "label": label, "case_dir": str(case_dir),
            "grid_size": int(manifest["grid_size"]),
            "tmax_c": float(thermal["tmax_c"]),
        })
    external_evaluations = None
    if external_samples:
        external_evaluations = {
            "cases": external_metadata,
            "defaults": metrics(external_samples, default_parameters, target_config),
            "fitted": metrics(external_samples, parameters, target_config),
        }
    leave_one_model_out = {}
    for label, _ in models:
        loo_train = [sample for sample in completed if sample["model_label"] != label]
        loo_test = [sample for sample in completed if sample["model_label"] == label]
        if not loo_train or not loo_test:
            continue
        loo_fit = fit(
            loo_train, config, spatial_weight=spatial_weight,
            fixed_beta=fixed_beta,
            fixed_cross_tier_weight=selected_weight,
        )
        values = loo_fit["parameters"]
        vector = [values["alpha"], values["beta"], values["cross_tier_weight"]]
        leave_one_model_out[label] = {
            "fit": loo_fit,
            "held_out_metrics": metrics(loo_test, vector, config),
        }

    input_hashes = {
        "configuration": {
            "path": str(config_path.resolve()), "sha256": sha256(config_path),
        },
        "models": [
            {"label": label, "path": str(path.resolve()), "sha256": sha256(path)}
            for label, path in models
        ],
        "external_cases": [
            {
                "group": group,
                "label": label,
                "files": {
                    name: {
                        "path": str((case_dir / name).resolve()),
                        "sha256": sha256(case_dir / name),
                    }
                    for name in ("layout.json", "thermal_result.json", "hotspot_manifest.json")
                },
            }
            for group, label, case_dir in external_cases or []
        ],
    }
    report = {
        "schema_version": 1,
        "method": "equation-(14) constrained nonlinear least squares against HotSpot",
        "config": str(config_path.resolve()),
        "models": [{"label": label, "path": str(path)} for label, path in models],
        "input_hashes": input_hashes,
        "sample_count": len(completed),
        "training_count": len(training),
        "validation_count": len(validation),
        "grid_points_per_axis": grid_points,
        "calibration_hotspot_grid_size": int(config["physical"]["grid_size"]),
        "target_hotspot_grid_size": target_grid_size,
        "hotspot_workers": workers,
        "spatial_residual_weight": spatial_weight,
        "cross_weight_step": cross_weight_step,
        "fit": fitted,
        "absolute_only_fit": absolute_fit,
        "direct_joint_fit_rejected": direct_joint_fit,
        "cross_validation": cross_validation,
        "evaluations": evaluations,
        "leave_one_model_out": leave_one_model_out,
        "external_target_grid_validation": external_evaluations,
        "interpretation": {
            "fit_scope": "active thermal-proxy parameters only; R_conv and Lc remain fixed",
            "fit_objective": (
                "absolute temperature residuals plus spatial_weight times within-model "
                "centered residuals; weight 10 makes 0.1 C spatial error comparable to 1 C absolute error"
            ),
            "lambda_wire": "not identifiable from temperature; requires gem5 R2 latency sweep",
            "paper_status": "calibration procedure is a local reproduction enhancement, not disclosed by the paper",
            "acceptance_guidance": (
                "Use fitted values only if held-out error and spatial rank metrics improve, "
                "the Jacobian is full-rank/well-conditioned, and leave-one-model-out values are stable."
            ),
        },
    }
    tier_sets = {
        label: sorted({int(sample["tier"]) for sample in completed
                       if sample["model_label"] == label})
        for label, _ in models
    }
    bottom_power_ranges = {}
    for label, _ in models:
        values = []
        for sample in completed:
            if sample["model_label"] != label:
                continue
            layout = read_json(Path(sample["case_dir"]) / "layout.json")
            values.append(sum(
                float(module["total_power_w"]) for module in layout["modules"]
                if int(module["tier"]) == 0
            ))
        bottom_power_ranges[label] = max(values) - min(values) if values else 0.0
    beta_identifiable = any(value > 1e-12 for value in bottom_power_ranges.values())
    report["identifiability"] = {
        "legal_l2_tiers_by_model": tier_sets,
        "within_model_bottom_power_range_w": bottom_power_ranges,
        "beta_tier_effect_identifiable": beta_identifiable,
        "explanation": (
            "beta multiplies bottom-tier power. It cannot be measured from position-only "
            "samples when every legal candidate keeps L2 on the same tier. Cross-model "
            "absolute fitting is not evidence of the intended tier penalty."
        ),
    }
    baseline_validation = evaluations["defaults"]["validation"]
    calibrated_validation = evaluations["cross_validated_training_fit"]["validation"]
    acceptance_checks = proxy_acceptance_checks(
        calibrated_validation, baseline_validation, direct_joint_fit["rank"], selected_weight,
        fitted["beta_status"],
    )
    if strict_p1:
        report["strict_p1"] = {
            "enabled": True,
            "allowed_l2_tiers": [1],
            "beta": 0.0,
            "beta_status": "fixed_unidentifiable_under_p1",
            "candidate_layout_policy": "top-tier L2 positions only",
        }
        expected_fitting_workloads = {"fft", "matmul", "stencil"}
        fitting_workloads = {label.casefold() for label, _ in models}
        target_groups = {group.casefold() for group, _, _ in external_cases or []}
        acceptance_checks.update({
            "fitting_workloads_are_fft_matmul_stencil": (
                fitting_workloads == expected_fitting_workloads
            ),
            "target_validation_is_separate_stream_group": target_groups == {"stream"},
            "target_grid_validation_supplied": external_evaluations is not None,
            "leave_one_workload_out_spatial_ranks_at_least_0p8": (
                len(leave_one_model_out) == len(models)
                and all(
                    values["held_out_metrics"]["spatial_spearman"] is not None
                    and values["held_out_metrics"]["spatial_spearman"] >= 0.8
                    for values in leave_one_model_out.values()
                )
            ),
        })
    if external_evaluations is not None:
        external_default = external_evaluations["defaults"]
        external_fitted = external_evaluations["fitted"]
        acceptance_checks.update({
            "lower_external_target_grid_rmse": (
                external_fitted["rmse_c"] < external_default["rmse_c"]
            ),
            "lower_external_target_grid_spatial_rmse": (
                external_fitted["spatial_centered_rmse_c"]
                < external_default["spatial_centered_rmse_c"]
            ),
            "external_target_grid_spatial_rank_at_least_0p8": (
                external_fitted["spatial_spearman"] is not None
                and external_fitted["spatial_spearman"] >= 0.8
            ),
        })
    elif strict_p1:
        acceptance_checks.update({
            "lower_external_target_grid_rmse": False,
            "lower_external_target_grid_spatial_rmse": False,
            "external_target_grid_spatial_rank_at_least_0p8": False,
        })
    accepted = all(acceptance_checks.values())
    report["recommendation"] = {
        "accepted": accepted,
        "checks": acceptance_checks,
        "action": "do not modify a configuration; promote only through formal validation",
    }
    write_json(output_dir / "calibration_report.json", report)
    write_samples_csv(
        output_dir / "samples.csv", completed, default_parameters, parameters, config
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True,
                        help="repeatable LABEL=PATH modules.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--hotspot-grid-size", type=int,
        help="training-grid override; never changes the selected configuration",
    )
    parser.add_argument(
        "--spatial-weight", type=float, default=10.0,
        help="weight for within-model position-delta residuals (default: 10)",
    )
    parser.add_argument(
        "--cross-weight-step", type=float, default=0.005,
        help="held-out grid-search spacing for cross-tier weight (default: 0.005)",
    )
    parser.add_argument(
        "--external-case", action="append", type=parse_external_case, default=[],
        help=("repeatable GROUP:SAMPLE=CASE_DIR target-grid case; cases sharing "
              "GROUP are used to validate spatial ordering"),
    )
    parser.add_argument(
        "--allowed-l2-tiers", type=parse_l2_tiers, default=(0, 1),
        help="comma-separated candidate L2 tiers; strict P1 uses 1",
    )
    parser.add_argument(
        "--fixed-beta", type=float,
        help="fix beta rather than fitting it; strict P1 requires 0.0",
    )
    parser.add_argument(
        "--target-grid-size", type=int, default=32,
        help="required HotSpot grid size for every external target-validation case",
    )
    parser.add_argument("--hotspot", type=Path, default=DEFAULT_HOTSPOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = calibrate(
        args.model, args.config.resolve(), args.output_dir, args.grid_points,
        args.workers, args.hotspot.resolve(), args.force, args.hotspot_grid_size,
        args.spatial_weight, args.cross_weight_step,
        args.external_case, args.allowed_l2_tiers, args.fixed_beta, args.target_grid_size,
    )
    values = report["fit"]["parameters"]
    validation = report["evaluations"]["cross_validated_training_fit"]["validation"]
    print(
        "Fitted alpha={alpha:.6g}, beta={beta:.6g}, cross_tier_weight={cross_tier_weight:.6g}; "
        "validation RMSE={rmse:.4f} C, spatial Spearman={spearman}".format(
            **values, rmse=validation["rmse_c"], spearman=validation["spatial_spearman"]
        )
    )


if __name__ == "__main__":
    main()
