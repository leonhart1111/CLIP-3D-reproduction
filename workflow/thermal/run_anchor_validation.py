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
        )
        results.append({"label": case["label"], "result": result})
    summary = {
        "schema_version": 1, "manifest": str(manifest_path.resolve()),
        "case_count": len(results),
        "frequency_run_count": sum(len(item["result"]["frequencies"]) for item in results),
        "max_abs_linear_error_c": max(
            item["result"]["max_abs_linear_error_c"] for item in results
        ),
        "max_safe_error_c": max(
            (item["result"]["solution_validation"] or {"safe_error_c": 0.0})["safe_error_c"]
            for item in results
        ),
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
    print(f"validated {result['case_count']} anchors / {result['frequency_run_count']} frequency runs")


if __name__ == "__main__":
    main()
