#!/usr/bin/env python3
"""Evaluate the separate, non-formal raw-power P1 operational policy."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from workflow.common import read_json, write_json


ACTION = "operational use permitted; non-formal and not promotable"
APPROVED_PARAMETERS = {
    "alpha": 1.5643788695171585,
    "beta": 0.0,
    "cross_tier_weight": 0.995,
}
APPROVED_MINIMUM_RANK = 0.5
APPROVED_LOO_POLICY = "diagnostic_only"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one report artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_value(value: object, label: str) -> float:
    """Return one finite numeric value, or reject malformed evidence."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is missing or not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def evaluate(proxy_report: Path, config: Path, output: Path) -> dict:
    """Write an operational decision without changing its config or evidence."""
    proxy_report = proxy_report.resolve()
    config = config.resolve()
    output = output.resolve()
    if output == proxy_report or output == config:
        raise ValueError("output must not alias an input")
    report = read_json(proxy_report)
    configured = read_json(config)
    policy = configured.get("operational_validation", {})
    if policy.get("mode") != "operational":
        raise ValueError("operational_validation.mode must be operational")

    validation = report["evaluations"]["cross_validated_training_fit"]["validation"]
    baseline = report["evaluations"]["defaults"]["validation"]
    target = report["external_target_grid_validation"]["fitted"]
    target_default = report["external_target_grid_validation"]["defaults"]
    parameters = report["fit"]["parameters"]

    alpha = finite_value(parameters.get("alpha"), "report alpha")
    beta = finite_value(parameters.get("beta"), "report beta")
    cross_tier_weight = finite_value(
        parameters.get("cross_tier_weight"), "report cross_tier_weight"
    )
    optimizer = configured.get("layout_optimizer", {})
    configured_parameters = {
        "alpha": finite_value(optimizer.get("alpha"), "config alpha"),
        "beta": finite_value(optimizer.get("beta"), "config beta"),
        "cross_tier_weight": finite_value(
            optimizer.get("cross_tier_weight"), "config cross_tier_weight"
        ),
    }
    report_parameters = {
        "alpha": alpha,
        "beta": beta,
        "cross_tier_weight": cross_tier_weight,
    }
    for name, value in report_parameters.items():
        if abs(configured_parameters[name] - value) > 1e-12:
            raise ValueError(f"config {name} does not match report {name}")
        if abs(value - APPROVED_PARAMETERS[name]) > 1e-12:
            raise ValueError(f"report {name} does not match approved operational {name}")
        if abs(configured_parameters[name] - APPROVED_PARAMETERS[name]) > 1e-12:
            raise ValueError(f"config {name} does not match approved operational {name}")

    minimum_validation_rank = finite_value(
        policy.get("minimum_validation_spatial_spearman"),
        "minimum_validation_spatial_spearman",
    )
    minimum_target_rank = finite_value(
        policy.get("minimum_external_target_spatial_spearman"),
        "minimum_external_target_spatial_spearman",
    )
    if abs(minimum_validation_rank - APPROVED_MINIMUM_RANK) > 1e-12:
        raise ValueError(
            "operational_validation.minimum_validation_spatial_spearman "
            "must equal the approved value"
        )
    if abs(minimum_target_rank - APPROVED_MINIMUM_RANK) > 1e-12:
        raise ValueError(
            "operational_validation.minimum_external_target_spatial_spearman "
            "must equal the approved value"
        )
    if policy.get("leave_one_workload_out") != APPROVED_LOO_POLICY:
        raise ValueError(
            "operational_validation.leave_one_workload_out must equal the approved value"
        )
    fitted_validation_rmse = finite_value(validation.get("rmse_c"), "validation rmse_c")
    default_validation_rmse = finite_value(baseline.get("rmse_c"), "baseline rmse_c")
    fitted_validation_centered_rmse = finite_value(
        validation.get("spatial_centered_rmse_c"), "validation spatial_centered_rmse_c"
    )
    default_validation_centered_rmse = finite_value(
        baseline.get("spatial_centered_rmse_c"), "baseline spatial_centered_rmse_c"
    )
    fitted_validation_rank = finite_value(
        validation.get("spatial_spearman"), "validation spatial_spearman"
    )
    fitted_target_rmse = finite_value(target.get("rmse_c"), "target rmse_c")
    default_target_rmse = finite_value(target_default.get("rmse_c"), "target default rmse_c")
    fitted_target_centered_rmse = finite_value(
        target.get("spatial_centered_rmse_c"), "target spatial_centered_rmse_c"
    )
    default_target_centered_rmse = finite_value(
        target_default.get("spatial_centered_rmse_c"),
        "target default spatial_centered_rmse_c",
    )
    fitted_target_rank = finite_value(
        target.get("spatial_spearman"), "target spatial_spearman"
    )

    checks = {
        "lower_validation_rmse": fitted_validation_rmse < default_validation_rmse,
        "lower_validation_centered_rmse": (
            fitted_validation_centered_rmse < default_validation_centered_rmse
        ),
        "validation_spatial_rank_at_least_0p5": (
            fitted_validation_rank >= APPROVED_MINIMUM_RANK
        ),
        "lower_target_rmse": fitted_target_rmse < default_target_rmse,
        "lower_target_centered_rmse": (
            fitted_target_centered_rmse < default_target_centered_rmse
        ),
        "target_spatial_rank_at_least_0p5": fitted_target_rank >= APPROVED_MINIMUM_RANK,
        "cross_tier_weight_is_interior": 0.0 < cross_tier_weight < 1.0,
        "beta_is_fixed_unidentifiable_under_p1": (
            report.get("strict_p1", {}).get("beta_status")
            == "fixed_unidentifiable_under_p1"
        ),
        "config_parameters_match_report": True,
    }
    leave_one_workload_out = {}
    for workload, values in report.get("leave_one_model_out", {}).items():
        leave_one_workload_out[workload] = finite_value(
            values.get("held_out_metrics", {}).get("spatial_spearman"),
            f"leave-one-workload-out {workload} spatial_spearman",
        )
    result = {
        "mode": "operational",
        "non_formal": True,
        "source": {"path": str(proxy_report), "sha256": sha256(proxy_report)},
        "parameters": report_parameters,
        "policy": {
            "minimum_validation_spatial_spearman": minimum_validation_rank,
            "minimum_external_target_spatial_spearman": minimum_target_rank,
            "leave_one_workload_out": policy.get("leave_one_workload_out"),
        },
        "checks": checks,
        "diagnostics": {"leave_one_workload_out": leave_one_workload_out},
        "recommendation": {
            "accepted": all(checks.values()),
            "action": ACTION,
        },
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.proxy_report, args.config, args.output)
    print(f"operational accepted={result['recommendation']['accepted']} -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
