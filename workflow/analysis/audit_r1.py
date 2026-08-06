#!/usr/bin/env python3
"""Audit completeness and instruction-window consistency of a formal R1 grid."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.r1_catalog import build_catalogue


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "configs/experiments/r1_cache_sweep.json"


def audit(root: Path, output: Path, experiment_path: Path | int | None = None,
          profile: str = "paper", expected_points: int = 100) -> dict:
    """Audit the configured canonical R1 grid, excluding unrelated statuses.

    ``experiment_path`` defaults to the project R1 experiment for compatibility
    with legacy callers.  A positional integer third argument retains the former
    ``audit(root, output, expected_points)`` form.
    """
    if isinstance(experiment_path, int):
        expected_points = experiment_path
        experiment_path = None
    experiment_path = Path(experiment_path or DEFAULT_EXPERIMENT).resolve()
    root = Path(root).resolve()
    output = Path(output).resolve()

    catalogue = build_catalogue(root, experiment_path, profile=profile)
    planned_path = root / "planned_jobs.json"
    planned = read_json(planned_path) if planned_path.is_file() else None
    planned_count = int(planned["job_count"]) if planned else expected_points
    records = catalogue["canonical_records"]
    counts = Counter(record["state"] for record in records)
    scopes = sorted({record.get("instruction_window_scope") for record in records
                     if record.get("instruction_window_scope")})
    result = {
        "schema_version": 2,
        "root": str(root),
        "experiment": str(experiment_path),
        "profile": profile,
        "expected_points": expected_points,
        "canonical_status_count": len(records),
        "status_file_count": len(records),
        "state_counts": dict(counts),
        "valid_success_count": catalogue["valid_count"],
        "excluded_noncanonical_count": len(catalogue["excluded_noncanonical"]),
        "excluded_noncanonical": catalogue["excluded_noncanonical"],
        "planned_points": planned_count,
        "instruction_window_scopes": scopes,
        "complete": (planned_count == expected_points and catalogue["complete"]
                     and len(scopes) == 1),
        "records": records,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--profile", default="paper")
    parser.add_argument("--expected-points", type=int, default=100)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.root, args.output, args.experiment, args.profile,
                   args.expected_points)
    print(f"R1 audit: states={result['state_counts']} valid_success="
          f"{result['valid_success_count']}/{result['expected_points']} "
          f"complete={result['complete']}")
    if args.require_complete and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
