#!/usr/bin/env python3
"""Fit auditable McPAT/thermal scales from published Table-III anchors.

This utility does not fit layout improvement.  It fits only quantities that
the paper exposes directly: total power, leakage fraction, and fixed-bin
Tmax.  Power scales are least-squares fits to target dynamic/leakage power.
The stack scale is a first-order fit to the local rise left after subtracting
the exact ambient and R_conv*P terms.  Rerun all anchors with the suggested
config and repeat until the stack-scale update is negligible.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

from workflow.common import read_json, write_json


@dataclass(frozen=True)
class Anchor:
    label: str
    point_dir: Path
    target_power_w: float
    target_gamma: float
    target_tmax_c: float


def parse_anchor(text: str) -> Anchor:
    fields = text.split("=")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "anchor must be LABEL=POINT_DIR=TARGET_POWER_W=TARGET_GAMMA=TARGET_TMAX_C"
        )
    label, raw_dir, power, gamma, tmax = fields
    point_dir = Path(raw_dir).expanduser().resolve()
    if not (point_dir / "pipeline_summary.json").is_file():
        raise argparse.ArgumentTypeError(f"pipeline summary missing below {point_dir}")
    result = Anchor(label, point_dir, float(power), float(gamma), float(tmax))
    if result.target_power_w <= 0 or not 0 < result.target_gamma < 1:
        raise argparse.ArgumentTypeError("anchor power/gamma is invalid")
    return result


def power_fit(records: list[dict]) -> dict:
    dynamic_denominator = sum(item["raw_dynamic_w"] ** 2 for item in records)
    leakage_denominator = sum(item["raw_leakage_w"] ** 2 for item in records)
    dynamic_scale = sum(
        item["raw_dynamic_w"] * item["target_dynamic_w"] for item in records
    ) / dynamic_denominator
    leakage_scale = sum(
        item["raw_leakage_w"] * item["target_leakage_w"] for item in records
    ) / leakage_denominator
    predictions = []
    for item in records:
        dynamic = dynamic_scale * item["raw_dynamic_w"]
        leakage = leakage_scale * item["raw_leakage_w"]
        total = dynamic + leakage
        predictions.append({
            "label": item["label"],
            "predicted_power_w": total,
            "target_power_w": item["target_power_w"],
            "power_error_w": total - item["target_power_w"],
            "predicted_gamma": leakage / total,
            "target_gamma": item["target_gamma"],
            "gamma_error": leakage / total - item["target_gamma"],
        })
    return {
        "dynamic_scale": dynamic_scale,
        "leakage_scale": leakage_scale,
        "anchor_count": len(records),
        "predictions": predictions,
    }


def calibrate(anchors: list[Anchor], output: Path,
              input_config: Path | None = None,
              output_config: Path | None = None) -> dict:
    if not anchors:
        raise ValueError("at least one anchor is required")
    records = []
    for anchor in anchors:
        summary = read_json(anchor.point_dir / "pipeline_summary.json")
        mcpat = read_json(anchor.point_dir / "mcpat/mcpat.json")
        run_config = read_json(anchor.point_dir / "run_config.json")["config"]
        raw = mcpat["raw_module_totals"]
        ambient = float(summary["cooling"]["ambient_c"])
        r_convec = float(summary["cooling"]["r_convec_k_per_w"])
        observed_power = float(summary["total_power_w"])
        current_scale = float(
            run_config["physical"]["thermal_stack"]["local_resistance_scale"]
        )
        observed_local = float(summary["tmax_c"]) - ambient - r_convec * observed_power
        target_local = anchor.target_tmax_c - ambient - r_convec * anchor.target_power_w
        records.append({
            "label": anchor.label,
            "workload": summary["workload"],
            "point_dir": str(anchor.point_dir),
            "raw_dynamic_w": float(raw["dynamic_power_w"]),
            "raw_leakage_w": float(raw["leakage_power_w"]),
            "target_dynamic_w": anchor.target_power_w * (1.0 - anchor.target_gamma),
            "target_leakage_w": anchor.target_power_w * anchor.target_gamma,
            "target_power_w": anchor.target_power_w,
            "target_gamma": anchor.target_gamma,
            "observed_power_w": observed_power,
            "observed_gamma": float(summary["gamma"]),
            "observed_tmax_c": float(summary["tmax_c"]),
            "target_tmax_c": anchor.target_tmax_c,
            "ambient_c": ambient,
            "r_convec_k_per_w": r_convec,
            "current_local_resistance_scale": current_scale,
            "observed_local_rise_c": observed_local,
            "target_local_rise_c": target_local,
            "implied_local_resistance_scale": (
                current_scale * target_local / observed_local
                if observed_local > 0 and target_local > 0 else None
            ),
        })

    global_power = power_fit(records)
    by_workload = {}
    for workload in sorted({item["workload"] for item in records}):
        by_workload[workload] = power_fit([
            item for item in records if item["workload"] == workload
        ])

    # First-order local-rise model: local_rise(scale) ~= slope * scale.
    slopes = [
        item["observed_local_rise_c"] / item["current_local_resistance_scale"]
        for item in records
    ]
    numerator = sum(
        slope * item["target_local_rise_c"]
        for slope, item in zip(slopes, records)
    )
    denominator = sum(slope * slope for slope in slopes)
    fitted_stack_scale = numerator / denominator
    stack_predictions = []
    for slope, item in zip(slopes, records):
        predicted_local = slope * fitted_stack_scale
        stack_predictions.append({
            "label": item["label"],
            "predicted_local_rise_c": predicted_local,
            "target_local_rise_c": item["target_local_rise_c"],
            "local_rise_error_c": predicted_local - item["target_local_rise_c"],
            "implied_local_resistance_scale": item["implied_local_resistance_scale"],
        })

    report = {
        "schema_version": 1,
        "method": "Table-III observable-only calibration",
        "anchors": records,
        "power_fit": {
            "global": global_power,
            "by_workload": by_workload,
            "equations": {
                "target_dynamic": "P_target*(1-gamma_target)",
                "target_leakage": "P_target*gamma_target",
                "fit": "independent zero-intercept least squares on raw McPAT components",
            },
        },
        "thermal_stack_fit": {
            "local_resistance_scale": fitted_stack_scale,
            "predictions": stack_predictions,
            "equation": "Tlocal=Tmax-Tamb-Rconv*P; first-order Tlocal proportional to scale",
            "status": "provisional; rerun anchors because power scales and stack conductance are coupled",
        },
        "limits": [
            "The paper does not publish its McPAT XML or complete Cool-3D layer file.",
            "Workload overrides are local reproduction calibrations, not published constants.",
            "A common physical stack is retained; per-anchor thermal scales are diagnostics only.",
            "No parameter is fitted to fixed-vs-optimized BIPS improvement.",
        ],
    }
    max_power_relative_error = max(
        abs(item["observed_power_w"] - item["target_power_w"])
        / item["target_power_w"] for item in records
    )
    max_gamma_error = max(
        abs(item["observed_gamma"] - item["target_gamma"]) for item in records
    )
    max_temperature_error = max(
        abs(item["observed_tmax_c"] - item["target_tmax_c"]) for item in records
    )
    accepted = bool(
        max_power_relative_error < 0.03
        and max_gamma_error < 0.02
        and max_temperature_error < 1.0
    )
    report["recommendation"] = {
        "accepted": accepted,
        "max_observed_power_relative_error": max_power_relative_error,
        "max_observed_gamma_error": max_gamma_error,
        "max_observed_temperature_error_c": max_temperature_error,
        "acceptance": "power <3%, gamma <0.02, and fixed-layout Tmax <1 C on every anchor",
        "action": (
            "freeze the common parameters and validate new held-out anchors"
            if accepted else
            "do not claim paper-parameter recovery; missing module placement or package inputs remain"
        ),
    }
    write_json(output, report)

    if bool(input_config) != bool(output_config):
        raise ValueError("input_config and output_config must be supplied together")
    if input_config and output_config:
        config = copy.deepcopy(read_json(input_config))
        config["name"] = f"{config.get('name', input_config.stem)}_table3_multianchor"
        global_values = report["power_fit"]["global"]
        calibration = {
            "dynamic_scale": global_values["dynamic_scale"],
            "leakage_scale": global_values["leakage_scale"],
            "provenance": {
                "kind": "Table III multi-anchor least-squares calibration",
                "report": str(output.resolve()),
                "scope": "global fallback",
            },
            "by_workload": {},
        }
        for workload, values in by_workload.items():
            calibration["by_workload"][workload] = {
                "dynamic_scale": values["dynamic_scale"],
                "leakage_scale": values["leakage_scale"],
                "provenance": {
                    "kind": "Table III workload-anchor calibration",
                    "report": str(output.resolve()),
                    "workload": workload,
                    "anchor_count": values["anchor_count"],
                },
            }
        config["mcpat"]["power_calibration"] = calibration
        config["physical"]["thermal_stack"][
            "local_resistance_scale"
        ] = fitted_stack_scale
        config["physical"]["thermal_stack"]["calibration"] = {
            "kind": "provisional Table III multi-anchor local-rise fit",
            "report": str(output.resolve()),
            "requires_anchor_rerun": True,
        }
        write_json(output_config, config)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", action="append", type=parse_anchor, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-config", type=Path)
    parser.add_argument("--output-config", type=Path)
    args = parser.parse_args()
    report = calibrate(
        args.anchor, args.output.resolve(),
        args.input_config.resolve() if args.input_config else None,
        args.output_config.resolve() if args.output_config else None,
    )
    fit = report["thermal_stack_fit"]
    print(
        f"Fitted common local_resistance_scale={fit['local_resistance_scale']:.6g}; "
        f"report={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
