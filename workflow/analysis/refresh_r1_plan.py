#!/usr/bin/env python3
"""Regenerate an R1 plan while proving canonical measured artifacts are unchanged."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from workflow.analysis.audit_r1 import audit
from workflow.common import read_json, write_json
from workflow.r1_catalog import build_catalogue, snapshot_canonical_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = PROJECT_ROOT / "scripts/run_r1_sweep.py"


def _require_complete_catalogue(root: Path, experiment_path: Path, profile: str) -> dict:
    catalogue = build_catalogue(root, experiment_path, profile=profile)
    if not catalogue["complete"]:
        raise RuntimeError("canonical R1 catalogue is incomplete or invalid")
    return catalogue


def refresh(root: Path, experiment_path: Path, profile: str, output: Path) -> dict:
    """Refresh only ``planned_jobs.json`` and prove R1 result artifacts persist."""
    root = Path(root).resolve()
    experiment_path = Path(experiment_path).resolve()
    output = Path(output).resolve()
    if root.name != profile:
        raise ValueError(
            f"root directory must match profile: {root.name!r} != {profile!r}"
        )
    catalogue = _require_complete_catalogue(root, experiment_path, profile)
    before_path = output / "canonical_r1_before.sha256.json"
    after_path = output / "canonical_r1_after.sha256.json"
    before = snapshot_canonical_artifacts(catalogue)
    write_json(before_path, before)

    command = [
        sys.executable,
        str(SWEEP_SCRIPT),
        "--experiment", str(experiment_path),
        "--profile", profile,
        "--output-root", str(root.parent),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    after_catalogue = build_catalogue(root, experiment_path, profile=profile)
    after = snapshot_canonical_artifacts(after_catalogue)
    write_json(after_path, after)
    if before != after:
        raise RuntimeError("canonical R1 artifacts changed during plan refresh")
    if completed.returncode:
        raise RuntimeError(f"R1 plan refresh failed with exit status {completed.returncode}")
    if not after_catalogue["complete"]:
        raise RuntimeError("canonical R1 catalogue is incomplete or invalid after plan refresh")

    planned_path = root / "planned_jobs.json"
    if not planned_path.is_file():
        raise RuntimeError("R1 plan refresh did not write planned_jobs.json")
    planned = read_json(planned_path)
    planned_count = int(planned["job_count"])
    expected_count = after_catalogue["expected_count"]
    if planned_count != expected_count:
        raise RuntimeError(
            f"planned_jobs.json contains {planned_count} jobs; expected {expected_count}"
        )

    audit_path = output / "audit.json"
    audit_result = audit(root, audit_path, experiment_path, profile,
                         expected_points=expected_count)
    if not audit_result["complete"]:
        raise RuntimeError("canonical R1 audit is incomplete after plan refresh")

    report = {
        "schema_version": 1,
        "root": str(root),
        "experiment": str(experiment_path),
        "profile": profile,
        "command": command,
        "before_manifest": str(before_path),
        "after_manifest": str(after_path),
        "audit_path": str(audit_path),
        "plan_count": planned_count,
        "canonical_artifacts_unchanged": True,
    }
    write_json(output / "refresh_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--profile", default="paper")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = refresh(args.root, args.experiment, args.profile, args.output)
    print(f"Refreshed {result['plan_count']}-job R1 plan; canonical artifacts unchanged")


if __name__ == "__main__":
    main()
