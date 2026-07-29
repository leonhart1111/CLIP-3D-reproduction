#!/usr/bin/env python3
"""Estimate equation-(15) lambda_wire from matched gem5 R2 runs.

All runs must use the same workload and architecture and differ only in the
integer layout-wire latency.  A two-run finite difference is retained for
backward compatibility.  The preferred mode fits a line through at least
three latency levels so one noisy or threshold-sensitive pair is not silently
promoted to a universal parameter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow.common import read_json, write_json


def injected_wire_cycles(latency: dict) -> int:
    """Return the integer latency actually injected into gem5 R2.

    ``layout_delays`` intentionally preserves the geometry-derived value even
    when build_latency_vector is called with an explicit ``--wire-cycles``
    sensitivity override.  The applied value therefore lives in
    ``components_cycles.layout_wire``; old vectors fall back to the former
    field for compatibility.
    """
    components = latency.get("components_cycles") or {}
    if "layout_wire" in components:
        return int(components["layout_wire"])
    layout = latency.get("layout_delays") or {}
    if "wire_cycles" in layout:
        return int(layout["wire_cycles"])
    raise KeyError("latency vector does not record an injected layout-wire cycle")


def read_sample(label: str, result_path: Path, latency_path: Path) -> dict:
    result = read_json(result_path)
    latency = read_json(latency_path)
    return {
        "label": label,
        "result": str(result_path.resolve()),
        "latency": str(latency_path.resolve()),
        "ipc2": float(result["ipc2"]),
        "wire_cycles": injected_wire_cycles(latency),
        "gem5_overrides": latency.get("gem5_overrides"),
    }


def calibrate_series(samples: list[tuple[str, Path, Path]], ipc1: float,
                     frequency_ghz: float) -> dict:
    """Fit IPC2 = intercept - loss_per_cycle * wire_cycles by OLS."""
    if ipc1 <= 0 or frequency_ghz <= 0:
        raise ValueError("ipc1 and frequency_ghz must be positive")
    records = [read_sample(label, result, latency)
               for label, result, latency in samples]
    if len(records) < 3:
        raise ValueError("series calibration requires at least three R2 samples")
    if len({record["wire_cycles"] for record in records}) < 3:
        raise ValueError("series calibration requires at least three distinct wire-cycle levels")
    x_mean = sum(record["wire_cycles"] for record in records) / len(records)
    y_mean = sum(record["ipc2"] for record in records) / len(records)
    denominator = sum((record["wire_cycles"] - x_mean) ** 2 for record in records)
    slope = sum(
        (record["wire_cycles"] - x_mean) * (record["ipc2"] - y_mean)
        for record in records
    ) / denominator
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * record["wire_cycles"] for record in records]
    residual_ss = sum((record["ipc2"] - prediction) ** 2
                      for record, prediction in zip(records, predictions))
    total_ss = sum((record["ipc2"] - y_mean) ** 2 for record in records)
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
    ipc_loss_per_added_cycle = -slope
    if ipc_loss_per_added_cycle <= 0:
        raise ValueError("R2 series does not show positive IPC benefit from lower latency")
    ordered = sorted(records, key=lambda record: (record["wire_cycles"], record["label"]))
    monotonic_violations = []
    for left, right in zip(ordered, ordered[1:]):
        if right["wire_cycles"] > left["wire_cycles"] and right["ipc2"] > left["ipc2"]:
            monotonic_violations.append({
                "lower_cycle_sample": left["label"],
                "higher_cycle_sample": right["label"],
            })
    lambda_wire = frequency_ghz * ipc_loss_per_added_cycle / ipc1
    for record, prediction in zip(records, predictions):
        record["predicted_ipc2"] = prediction
        record["residual_ipc2"] = record["ipc2"] - prediction
    accepted = r_squared >= 0.95 and not monotonic_violations
    return {
        "schema_version": 2,
        "method": "multi-level ordinary least-squares regression of gem5 R2 IPC",
        "samples": records,
        "ipc1": ipc1,
        "reference_sustainable_frequency_ghz": frequency_ghz,
        "fit": {
            "equation": "IPC2 = intercept - ipc_loss_per_added_cycle * wire_cycles",
            "intercept": intercept,
            "ipc_loss_per_added_cycle": ipc_loss_per_added_cycle,
            "r_squared": r_squared,
            "monotonic_violations": monotonic_violations,
        },
        "lambda_wire": lambda_wire,
        "lambda_equation": (
            "lambda_wire = f_sus * ipc_loss_per_added_cycle / IPC1"
        ),
        "units": "GHz per wire cycle in the current IPC1-scaled equation-(15) implementation",
        "recommendation": {
            "accepted_for_this_workload": accepted,
            "acceptance": "R^2 >= 0.95 and no monotonicity violation",
            "cross_workload_transfer_validated": False,
            "action": (
                "repeat for every workload and report the distribution before choosing a shared value"
            ),
        },
        "limitations": [
            "The regression is local to one workload and architecture.",
            "All samples must differ only in the injected wire-cycle latency.",
            "The fitted value is cooling-point dependent because BIPS latency loss scales with frequency.",
        ],
    }


def calibrate(baseline_result_path: Path, candidate_result_path: Path,
              baseline_latency_path: Path, candidate_latency_path: Path,
              ipc1: float, frequency_ghz: float) -> dict:
    if ipc1 <= 0 or frequency_ghz <= 0:
        raise ValueError("ipc1 and frequency_ghz must be positive")
    baseline_result = read_json(baseline_result_path)
    candidate_result = read_json(candidate_result_path)
    baseline_latency = read_json(baseline_latency_path)
    candidate_latency = read_json(candidate_latency_path)
    baseline_cycles = injected_wire_cycles(baseline_latency)
    candidate_cycles = injected_wire_cycles(candidate_latency)
    removed_cycles = baseline_cycles - candidate_cycles
    ipc_gain = float(candidate_result["ipc2"]) - float(baseline_result["ipc2"])
    if removed_cycles == 0:
        raise ValueError("matched R2 pair has no discrete wire-cycle difference")
    ipc_loss_per_added_cycle = ipc_gain / removed_cycles
    if ipc_loss_per_added_cycle <= 0:
        raise ValueError("R2 pair does not show positive IPC benefit from lower latency")
    lambda_wire = frequency_ghz * ipc_loss_per_added_cycle / ipc1
    return {
        "schema_version": 1,
        "method": "matched-pair finite difference of gem5 R2 IPC",
        "baseline": {
            "result": str(baseline_result_path.resolve()),
            "latency": str(baseline_latency_path.resolve()),
            "ipc2": float(baseline_result["ipc2"]),
            "wire_cycles": baseline_cycles,
        },
        "candidate": {
            "result": str(candidate_result_path.resolve()),
            "latency": str(candidate_latency_path.resolve()),
            "ipc2": float(candidate_result["ipc2"]),
            "wire_cycles": candidate_cycles,
        },
        "ipc1": ipc1,
        "reference_sustainable_frequency_ghz": frequency_ghz,
        "removed_wire_cycles": removed_cycles,
        "ipc_gain": ipc_gain,
        "ipc_loss_per_added_cycle": ipc_loss_per_added_cycle,
        "lambda_wire": lambda_wire,
        "equation": "lambda_wire = f_sus * (delta_IPC2 / delta_wire_cycles) / IPC1",
        "units": "GHz per wire cycle in the current IPC1-scaled equation-(15) implementation",
        "limitations": [
            "This is a local one-workload, one-cycle finite difference, not a paper-disclosed value.",
            "Repeat across workloads and latency levels when more matched R2 runs are available.",
            "The fitted value is cooling-point dependent because BIPS latency loss scales with frequency.",
        ],
    }


def parse_sample(text: str) -> tuple[str, Path, Path]:
    if "=" not in text or "," not in text:
        raise argparse.ArgumentTypeError(
            "sample must be LABEL=R2_RESULT.json,R2_LATENCY.json"
        )
    label, paths = text.split("=", 1)
    result_text, latency_text = paths.split(",", 1)
    result = Path(result_text).expanduser().resolve()
    latency = Path(latency_text).expanduser().resolve()
    if not label or not result.is_file() or not latency.is_file():
        raise argparse.ArgumentTypeError(f"invalid R2 sample: {text}")
    return label, result, latency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--candidate-result", type=Path)
    parser.add_argument("--baseline-latency", type=Path)
    parser.add_argument("--candidate-latency", type=Path)
    parser.add_argument(
        "--sample", action="append", type=parse_sample,
        help="preferred repeatable LABEL=R2_RESULT.json,R2_LATENCY.json mode",
    )
    parser.add_argument("--ipc1", type=float, required=True)
    parser.add_argument("--frequency-ghz", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-config", type=Path)
    parser.add_argument("--output-config", type=Path)
    args = parser.parse_args()
    if bool(args.input_config) != bool(args.output_config):
        parser.error("--input-config and --output-config must be supplied together")
    legacy = (args.baseline_result, args.candidate_result,
              args.baseline_latency, args.candidate_latency)
    if args.sample and any(legacy):
        parser.error("choose either repeated --sample or the legacy baseline/candidate pair")
    if args.sample:
        report = calibrate_series(args.sample, args.ipc1, args.frequency_ghz)
    elif all(legacy):
        report = calibrate(
            args.baseline_result.resolve(), args.candidate_result.resolve(),
            args.baseline_latency.resolve(), args.candidate_latency.resolve(),
            args.ipc1, args.frequency_ghz,
        )
    else:
        parser.error(
            "provide at least three --sample entries or all four legacy pair paths"
        )
    write_json(args.output, report)
    if args.input_config:
        config = read_json(args.input_config)
        config["name"] = f"{config.get('name', args.input_config.stem)}_wire_fitted"
        config["layout_optimizer"]["lambda_wire"] = report["lambda_wire"]
        config["layout_optimizer"]["lambda_wire_calibration"] = {
            "report": str(args.output.resolve()),
            "method": report["method"],
            "reference_sustainable_frequency_ghz": args.frequency_ghz,
            "status": "local pilot estimate; cross-workload R2 validation pending",
        }
        config.setdefault("provenance", {}).setdefault(
            "reproduction_assumptions", []
        ).append(
            "lambda_wire was estimated from a matched local gem5 R2 latency pair; "
            "it is not a paper-disclosed constant."
        )
        write_json(args.output_config, config)
    ipc_loss = (
        report.get("ipc_loss_per_added_cycle")
        if "ipc_loss_per_added_cycle" in report
        else report["fit"]["ipc_loss_per_added_cycle"]
    )
    print(f"lambda_wire={report['lambda_wire']:.9g}; "
          f"IPC loss={ipc_loss:.9g} per added cycle")


if __name__ == "__main__":
    main()
