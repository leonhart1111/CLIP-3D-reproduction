#!/usr/bin/env python3
"""Score, Pareto-filter, and prune LogicFolding candidates without ML."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from workflow.common import read_json, sha256_file, write_json
from workflow.thermal.linear_response import LinearThermalResponse
from workflow.thermal.linear_search import (
    evaluate_candidates,
    pareto_front,
    rank_for_pruning,
)


def validate_config(config: dict) -> dict:
    if config.get("schema_version") != 1:
        raise ValueError("linear search config schema_version must be 1")
    contract = config.get("thermal_contract")
    response = config.get("response")
    search = config.get("search")
    if not all(isinstance(item, dict) for item in (contract, response, search)):
        raise ValueError("thermal_contract, response, and search must be objects")
    if contract.get("active_tiers") != 2 or contract.get("grid_size") != 32:
        raise ValueError("phase-one contract requires two 32x32 active tiers")
    thermal_limit = contract.get("thermal_limit_c")
    tau = response.get("smooth_peak_tau_c")
    top_k = search.get("top_k_full_evaluation")
    if (isinstance(thermal_limit, bool)
            or not isinstance(thermal_limit, (int, float))
            or not math.isfinite(float(thermal_limit))):
        raise ValueError("thermal_limit_c must be finite")
    if (isinstance(tau, bool) or not isinstance(tau, (int, float))
            or not math.isfinite(float(tau)) or float(tau) <= 0.0):
        raise ValueError("smooth_peak_tau_c must be finite and positive")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k_full_evaluation must be a positive integer")
    objectives = search.get("pareto_objectives")
    if not isinstance(objectives, dict):
        raise ValueError("pareto_objectives must be an object")
    if objectives.get("maximize") != ["performance"]:
        raise ValueError("phase-one maximize objective must be performance")
    if objectives.get("minimize") != [
        "energy", "area", "communication", "predicted_tsoft_c"
    ]:
        raise ValueError("phase-one minimize objectives do not match the contract")
    return config


def run_search(response_manifest: Path, candidates_path: Path,
               config_path: Path, output: Path) -> dict:
    response = LinearThermalResponse.load(response_manifest)
    config = validate_config(read_json(config_path))
    candidate_document = read_json(candidates_path)
    if candidate_document.get("schema_version") != 1:
        raise ValueError("candidate document schema_version must be 1")
    candidates = candidate_document.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate document needs a non-empty candidates list")
    evaluated = evaluate_candidates(
        response,
        candidates,
        float(config["thermal_contract"]["thermal_limit_c"]),
        float(config["response"]["smooth_peak_tau_c"]),
    )
    front = pareto_front(evaluated)
    ranked = rank_for_pruning(evaluated)
    front_ids = {record["id"] for record in front}
    queue = rank_for_pruning(front) + [
        record for record in ranked if record["id"] not in front_ids
    ]
    top_k = int(config["search"]["top_k_full_evaluation"])
    selected = queue[:top_k]
    selected_ids = {record["id"] for record in selected}
    report = {
        "schema_version": 1,
        "mode": "non-ML-linear-response-ranking",
        "response_manifest": str(response_manifest.resolve()),
        "response_manifest_sha256": sha256_file(response_manifest),
        "candidate_document": str(candidates_path.resolve()),
        "candidate_document_sha256": sha256_file(candidates_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "candidate_count": len(evaluated),
        "pareto_ids": [record["id"] for record in front],
        "selected_for_full_evaluation_ids": [record["id"] for record in selected],
        "pruned_ids": [
            record["id"] for record in evaluated
            if record["id"] not in selected_ids
        ],
        "candidates": evaluated,
        "selection_policy": {
            "pareto_front_first": True,
            "thermal_feasibility_first": True,
            "top_k": top_k,
            "selected_candidates_require_full_floorplan_and_HotSpot": True
        }
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True,
                        help="linear_response.json")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_search(
        args.response.resolve(), args.candidates.resolve(),
        args.config.resolve(), args.output.resolve(),
    )
    print(
        f"linear search: {report['candidate_count']} candidates; "
        f"selected {len(report['selected_for_full_evaluation_ids'])}; "
        f"output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
