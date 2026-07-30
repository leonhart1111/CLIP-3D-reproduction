#!/usr/bin/env python3
"""Promote a strict-P1 candidate only from accepted validation reports."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.run_lifting_pipeline import validate_config


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one report artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted(report: dict, name: str) -> None:
    if report.get("recommendation", {}).get("accepted") is not True:
        raise ValueError(f"{name} report is not accepted")


def finite_value(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is missing or not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def promote(proxy_report: Path, wire_summary: Path, frequency_report: Path,
            candidate_config: Path, output_config: Path) -> dict:
    """Write a formal config with values traceable to accepted report files."""
    proxy_report = proxy_report.resolve()
    wire_summary = wire_summary.resolve()
    frequency_report = frequency_report.resolve()
    candidate_config = candidate_config.resolve()
    proxy = read_json(proxy_report)
    wire = read_json(wire_summary)
    frequency = read_json(frequency_report)
    accepted(proxy, "proxy")
    accepted(wire, "wire")
    accepted(frequency, "frequency")
    if proxy.get("strict_p1", {}).get("beta_status") != "fixed_unidentifiable_under_p1":
        raise ValueError("proxy strict-P1 beta status is not fixed_unidentifiable_under_p1")

    config = copy.deepcopy(read_json(candidate_config))
    validate_config(config, "clip3d")
    optimizer = config["layout_optimizer"]
    parameters = proxy.get("fit", {}).get("parameters", {})
    optimizer["alpha"] = finite_value(parameters.get("alpha"), "proxy alpha")
    optimizer["cross_tier_weight"] = finite_value(
        parameters.get("cross_tier_weight"), "proxy cross_tier_weight"
    )
    optimizer["lambda_wire"] = finite_value(
        wire.get("selected_lambda_wire"), "wire selected_lambda_wire"
    )
    optimizer["beta"] = 0.0

    artifacts = {}
    for key, path in (("proxy_report", proxy_report), ("wire_summary", wire_summary),
                      ("frequency_report", frequency_report)):
        artifacts[key] = {"path": str(path), "sha256": sha256(path)}
    formal = config.setdefault("formal_validation", {})
    formal.update({"strict_p1": True, "accepted": True, "artifacts": artifacts})
    config["name"] = "constrained_5p0_raw_power_p1_formal"
    provenance = optimizer.setdefault("parameter_provenance", {})
    provenance.update({
        "alpha": "accepted strict-P1 proxy report",
        "beta": "fixed_unidentifiable_under_p1",
        "cross_tier_weight": "accepted strict-P1 proxy report",
        "lambda_wire": "accepted cross-workload R2 wire report",
    })
    validate_config(config, "clip3d")
    write_json(output_config, config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-report", type=Path, required=True)
    parser.add_argument("--wire-summary", type=Path, required=True)
    parser.add_argument("--frequency-report", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = promote(
        args.proxy_report, args.wire_summary, args.frequency_report,
        args.candidate_config, args.output,
    )
    print(f"promoted {result['name']} to {args.output.resolve()}")


if __name__ == "__main__":
    main()
