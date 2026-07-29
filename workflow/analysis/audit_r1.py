#!/usr/bin/env python3
"""Audit completeness and instruction-window consistency of a formal R1 grid."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from workflow.common import parse_gem5_stats, read_json, write_json


def audit(root: Path, output: Path, expected_points: int = 100) -> dict:
    planned_path = root / "planned_jobs.json"
    planned = read_json(planned_path) if planned_path.is_file() else None
    point_dirs = sorted(path.parent for path in root.rglob("status.json"))
    records = []
    for directory in point_dirs:
        status = read_json(directory / "status.json")
        record = {"directory": str(directory.resolve()), "state": status.get("state"),
                  "job_id": status.get("job_id"), "valid": False, "errors": []}
        if status.get("state") == "success":
            for name in ("stats.txt", "r1_metadata.json"):
                if not (directory / name).is_file():
                    record["errors"].append(f"missing {name}")
            if not record["errors"]:
                metadata = read_json(directory / "r1_metadata.json")
                stats = parse_gem5_stats(directory / "stats.txt")
                scope = metadata.get("instruction_window_scope", "cpu0")
                target = int(metadata.get("measure_insts", metadata["measure_insts_cpu0"]))
                per_core = []
                for core in range(int(metadata.get("num_cores", 4))):
                    instructions = int(stats.get(
                        f"system.cpu{core}.commitStats0.numInsts", 0
                    ))
                    cycles = int(stats.get(f"system.cpu{core}.numCycles", 0))
                    per_core.append({"core": core, "instructions": instructions,
                                     "cycles": cycles})
                    minimum = target if scope == "all-cores" else 1
                    if instructions < minimum or cycles <= 0:
                        record["errors"].append(
                            f"CPU{core} instructions={instructions}, cycles={cycles}, minimum={minimum}"
                        )
                record.update({"instruction_window_scope": scope,
                               "measurement_target": target, "per_core": per_core})
            record["valid"] = not record["errors"]
        records.append(record)

    counts = Counter(record["state"] for record in records)
    valid_success = sum(record["state"] == "success" and record["valid"] for record in records)
    scopes = sorted({record.get("instruction_window_scope") for record in records
                     if record.get("instruction_window_scope")})
    planned_count = int(planned["job_count"]) if planned else expected_points
    complete = (
        planned_count == expected_points and valid_success == expected_points and
        len(records) == expected_points and counts.get("success", 0) == expected_points and
        len(scopes) == 1
    )
    result = {
        "schema_version": 1, "root": str(root.resolve()),
        "expected_points": expected_points, "planned_points": planned_count,
        "status_file_count": len(records), "state_counts": dict(counts),
        "valid_success_count": valid_success, "instruction_window_scopes": scopes,
        "complete": complete, "records": records,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, default=100)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.output.resolve(), args.expected_points)
    print(f"R1 audit: states={result['state_counts']} valid_success="
          f"{result['valid_success_count']}/{result['expected_points']} "
          f"complete={result['complete']}")
    if args.require_complete and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
