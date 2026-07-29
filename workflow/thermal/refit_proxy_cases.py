#!/usr/bin/env python3
"""Refit equation-(14) variants from existing HotSpot calibration cases."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.thermal.calibrate_proxy import (
    cross_validate_weight,
    fit,
    metrics,
    proxy_prediction,
    sample_split,
)


def load_samples(cases_dir: Path) -> list[dict]:
    samples = []
    signatures = sorted(cases_dir.glob("*/*/calibration_sample.json"))
    if not signatures:
        raise ValueError(f"no calibration cases below {cases_dir}")
    grid_points = 1 + max(int(read_json(path)["row"]) for path in signatures)
    for path in signatures:
        signature = read_json(path)
        thermal = read_json(path.parent / "thermal_result.json")
        samples.append({
            "model_index": 0,
            "model_label": signature["model_label"],
            "model": signature["model"],
            "tier": int(signature["tier"]),
            "row": int(signature["row"]),
            "column": int(signature["column"]),
            "fx": float(signature["fx"]),
            "fy": float(signature["fy"]),
            "split": sample_split(
                0, int(signature["row"]), int(signature["column"]),
                int(signature["tier"]), grid_points,
            ),
            "case_dir": str(path.parent.resolve()),
            "tmax_c": float(thermal["tmax_c"]),
            "peak_unit": thermal["peak_unit"],
        })
    return samples


def refit(cases_dir: Path, config_path: Path, output: Path,
          proxy_model: str, quadrature_order: int = 2,
          spatial_weight: float = 10.0,
          cross_weight_step: float = 0.005,
          lower_c: float = 80.0, upper_c: float = 110.0) -> dict:
    samples = load_samples(cases_dir)
    config = copy.deepcopy(read_json(config_path))
    config["layout_optimizer"]["proxy_spatial_model"] = proxy_model
    config["layout_optimizer"]["proxy_quadrature_order"] = quadrature_order
    training = [sample for sample in samples if sample["split"] == "train"]
    validation = [sample for sample in samples if sample["split"] == "validation"]
    cross = cross_validate_weight(
        training, validation, config, spatial_weight, cross_weight_step
    )
    weight = float(cross["selected"]["cross_tier_weight"])
    unconstrained = fit(
        samples, config, spatial_weight=spatial_weight,
        fixed_cross_tier_weight=weight,
    )

    # The paper specifies unit constants that keep the proxy in 80--110 C.
    # beta cannot be measured when every sample retains the same tier, so use
    # beta=0 and cap alpha without fitting against desired BIPS improvement.
    caps = []
    bases = []
    for sample in samples:
        layout = read_json(Path(sample["case_dir"]) / "layout.json")
        total = sum(float(module["total_power_w"]) for module in layout["modules"])
        base = (
            float(config["frequency"]["ambient_c"])
            + float(config["physical"]["r_convec_k_per_w"]) * total
        )
        unit = proxy_prediction(sample, [1.0, 0.0, weight], config) - base
        bases.append(base)
        if unit > 0:
            caps.append((upper_c - base) / unit)
    alpha = max(0.0, min(float(unconstrained["parameters"]["alpha"]), min(caps)))
    constrained = [alpha, 0.0, weight]
    predictions = [proxy_prediction(sample, constrained, config) for sample in samples]
    tier_values = sorted({int(sample["tier"]) for sample in samples})
    result = {
        "schema_version": 1,
        "method": "refit existing HotSpot cases with paper 80--110 C constraint",
        "cases_dir": str(cases_dir.resolve()),
        "config": str(config_path.resolve()),
        "proxy_spatial_model": proxy_model,
        "proxy_quadrature_order": quadrature_order,
        "sample_count": len(samples),
        "legal_l2_tiers": tier_values,
        "beta_tier_effect_identifiable": len(tier_values) > 1,
        "cross_validation": cross,
        "unconstrained_fit": unconstrained,
        "paper_range_parameters": {
            "alpha": alpha, "beta": 0.0, "cross_tier_weight": weight,
        },
        "paper_range_c": [lower_c, upper_c],
        "prediction_range_c": [min(predictions), max(predictions)],
        "training_metrics": metrics(training, constrained, config),
        "validation_metrics": metrics(validation, constrained, config),
        "all_metrics": metrics(samples, constrained, config),
    }
    spatial_rank = result["validation_metrics"]["spatial_spearman"]
    all_spatial_rank = result["all_metrics"]["spatial_spearman"]
    result["recommendation"] = {
        "accepted_for_directional_search": bool(
            spatial_rank is not None and spatial_rank >= 0.8
            and all_spatial_rank is not None and all_spatial_rank >= 0.8
            and 0.0 < weight < 1.0
            and min(predictions) >= lower_c - 1e-9
            and max(predictions) <= upper_c + 1e-9
        ),
        "not_a_physical_temperature_fit": True,
        "reason": (
            "Acceptance requires held-out and all-sample spatial rank, an interior "
            "cross-tier weight, and the paper proxy range. Reported temperature still "
            "comes from final HotSpot."
        ),
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--proxy-spatial-model", choices=("center", "area-quadrature"), required=True,
    )
    parser.add_argument("--proxy-quadrature-order", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--spatial-weight", type=float, default=10.0)
    parser.add_argument("--cross-weight-step", type=float, default=0.005)
    args = parser.parse_args()
    report = refit(
        args.cases_dir.resolve(), args.config.resolve(), args.output.resolve(),
        args.proxy_spatial_model, args.proxy_quadrature_order,
        args.spatial_weight, args.cross_weight_step,
    )
    print(report["paper_range_parameters"])
    print(report["recommendation"])


if __name__ == "__main__":
    main()
