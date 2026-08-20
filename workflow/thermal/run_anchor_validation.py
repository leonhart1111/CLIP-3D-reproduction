#!/usr/bin/env python3
"""Run equation-(13) validation for every case listed in a JSON manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.thermal.validate_frequency import validate_case


def run_manifest(manifest_path: Path, output: Path) -> dict:
    manifest = read_json(manifest_path)
    results = []
    for case in manifest["cases"]:
        case_dir = Path(case["case_dir"]).resolve()
        modules = Path(case.get("modules", case_dir.parent / "modules.json")).resolve()
        case_output = output.parent / "anchor_cases" / f"{case['label']}.json"
        result = validate_case(
            case_dir, modules, case_output,
            [float(value) for value in manifest.get("frequencies_ghz", [0.5, 1.0, 2.0])],
            bool(manifest.get("validate_solution", True)),
            manifest.get("frequency_settings"),
        )
        results.append({"label": case["label"], "result": result})
    requested_frequency_hotspot_run_count = sum(
        len(item["result"]["frequencies"]) for item in results
    )
    fsus_safety_solve_count = sum(
        item["result"]["solution_validation"] is not None for item in results
    )
    two_point_safety_solve_count = sum(
        (item["result"].get("two_point_affine_frequency") or {})
        .get("solution_validation", {}).get("not_required") is not True
        and (item["result"].get("two_point_affine_frequency") or {})
        .get("solution_validation") is not None
        for item in results
    )
    two_point_available_count = sum(
        bool((item["result"].get("two_point_affine_frequency") or {}).get("available"))
        for item in results
    )
    two_point_accepted_count = sum(
        bool((item["result"].get("two_point_affine_frequency") or {})
             .get("recommendation", {}).get("accepted"))
        for item in results
    )
    safe_errors = [
        item["result"]["solution_validation"]["safe_error_c"]
        for item in results
        if item["result"]["solution_validation"] is not None
        and item["result"]["solution_validation"]["safe_error_c"] is not None
    ]
    safe_error_limits = {
        float(item["result"]["frequency_settings"]["max_safe_error_c"])
        for item in results
    }
    if len(safe_error_limits) != 1:
        raise ValueError(
            "all frequency anchors must use the same max_safe_error_c "
            "acceptance threshold"
        )
    summary = {
        "schema_version": 1, "manifest": str(manifest_path.resolve()),
        "case_count": len(results),
        "requested_frequency_hotspot_run_count": requested_frequency_hotspot_run_count,
        "fsus_safety_solve_count": fsus_safety_solve_count,
        "two_point_safety_solve_count": two_point_safety_solve_count,
        "hotspot_run_count": (
            requested_frequency_hotspot_run_count + fsus_safety_solve_count
            + two_point_safety_solve_count
        ),
        "max_abs_uniform_gamma_comparison_error_c": max(
            item["result"]["max_abs_uniform_gamma_comparison_error_c"] for item in results
        ),
        "safe_error_limit_c": safe_error_limits.pop(),
        "max_safe_error_c": max(safe_errors, default=0.0),
        "recommendation": {
            "accepted": all(item["result"]["recommendation"]["accepted"]
                            for item in results),
        },
        "two_point_fallback_recommendation": {
            "available_case_count": two_point_available_count,
            "accepted_case_count": two_point_accepted_count,
            # A scalar-gamma rejection is not a global rejection if and only
            # if that case's explicitly reported Eq.(9) fallback passed its
            # own independent HotSpot safety validation.  Keep this separate
            # from ``recommendation`` so reports never relabel the scalar
            # Equation-(13) shortcut as accepted.
            "accepted": all(
                item["result"]["recommendation"]["accepted"]
                or bool((item["result"].get("two_point_affine_frequency") or {})
                        .get("recommendation", {}).get("accepted"))
                for item in results
            ),
        },
        "cases": results,
    }
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_manifest(args.manifest.resolve(), args.output.resolve())
    print(
        f"validated {result['case_count']} anchors / "
        f"{result['requested_frequency_hotspot_run_count']} requested frequency runs / "
        f"{result['hotspot_run_count']} total HotSpot runs"
    )


if __name__ == "__main__":
    main()
