#!/usr/bin/env python3
"""Run the exploratory MATMUL/STENCIL transient-ROM validation pair."""

from __future__ import annotations

import csv
import math
from numbers import Real
from pathlib import Path
import subprocess
import sys
from typing import Callable

from workflow.common import read_json, sha256_file, write_json
from workflow.run_lifting_pipeline import validate_config
from workflow.transient.run_transient_r1 import completed as transient_r1_completed


EXPECTED_POINTS = (
    {"workload": "matmul", "l1d_size": "64kB", "l2_size": "512kB"},
    {"workload": "stencil", "l1d_size": "64kB", "l2_size": "512kB"},
)
SAMPLE_INTERVAL_MS = 2.0
Invoke = Callable[[list[str], Path, Path], int]


def _point_key(point: dict) -> str:
    return (
        f"{point['workload']}/l1d_{point['l1d_size']}"
        f"/l2_{point['l2_size']}"
    )


def load_selection(path: Path) -> list[dict]:
    """Load and require the immutable ordered Balanced-2 selection."""
    path = Path(path).resolve()
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported Balanced-2 selection schema")
    classification = value.get("experiment_classification")
    if (
        not isinstance(classification, dict)
        or classification.get("non_formal") is not True
        or classification.get("paper_equivalent") is not False
    ):
        raise ValueError("Balanced-2 selection must remain exploratory")
    sample_ms = value.get("sample_interval_ms")
    if (
        isinstance(sample_ms, bool)
        or not isinstance(sample_ms, Real)
        or not math.isfinite(float(sample_ms))
        or float(sample_ms) != SAMPLE_INTERVAL_MS
    ):
        raise ValueError("selection sample_interval_ms must equal 2.0")
    points = value.get("points")
    if not isinstance(points, list) or points != list(EXPECTED_POINTS):
        raise ValueError("selection must contain the exact ordered points")
    return [dict(point) for point in points]


def _require_distinct_roots(canonical_root: Path, periodic_root: Path) -> None:
    if (
        canonical_root == periodic_root
        or canonical_root in periodic_root.parents
        or periodic_root in canonical_root.parents
    ):
        raise ValueError("canonical and periodic R1 roots must not overlap")


def _baseline_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("steady baseline CSV is empty")
    return rows


def _matching_baseline(rows: list[dict[str, str]], point: dict) -> dict[str, str]:
    matches = [
        row for row in rows
        if row.get("workload") == point["workload"]
        and row.get("l1d_size") == point["l1d_size"]
        and row.get("l2_size") == point["l2_size"]
    ]
    if len(matches) != 1 or matches[0].get("state") != "success":
        raise ValueError(
            f"point {_point_key(point)} lacks one successful steady baseline"
        )
    return dict(matches[0])


def _require_canonical_point(path: Path, point: dict) -> dict:
    required = (path / "status.json", path / "r1_metadata.json", path / "stats.txt")
    for artifact in required:
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    status = read_json(path / "status.json")
    metadata = read_json(path / "r1_metadata.json")
    if not isinstance(status, dict) or status.get("state") != "success":
        raise ValueError(f"canonical R1 is not successful: {path}")
    if not isinstance(metadata, dict) or any(
        metadata.get(field) != point[field]
        for field in ("workload", "l1d_size", "l2_size")
    ):
        raise ValueError(f"canonical R1 metadata differs from {_point_key(point)}")
    return metadata


def preflight_inputs(
    selection_path: Path,
    canonical_r1_root: Path,
    periodic_r1_root: Path,
    steady_baseline_csv: Path,
    config_path: Path,
) -> dict:
    """Validate all two-point read-only inputs before any experiment writes."""
    selection_path = Path(selection_path).resolve()
    canonical_r1_root = Path(canonical_r1_root).resolve()
    periodic_r1_root = Path(periodic_r1_root).resolve()
    steady_baseline_csv = Path(steady_baseline_csv).resolve()
    config_path = Path(config_path).resolve()
    _require_distinct_roots(canonical_r1_root, periodic_r1_root)
    for artifact in (selection_path, steady_baseline_csv, config_path):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)

    points = load_selection(selection_path)
    config = read_json(config_path)
    validate_config(config, "clip3d")
    classification = config.get("experiment_classification")
    rom = config.get("transient_rom")
    if (
        not isinstance(classification, dict)
        or classification.get("non_formal") is not True
        or classification.get("paper_equivalent") is not False
        or not isinstance(rom, dict)
        or rom.get("enabled") is not True
        or rom.get("non_formal") is not True
        or rom.get("paper_equivalent") is not False
        or rom.get("sample_interval_ms") != SAMPLE_INTERVAL_MS
    ):
        raise ValueError("scientific config must remain exploratory transient-ROM")

    baseline = _baseline_rows(steady_baseline_csv)
    validated_points = []
    for point in points:
        canonical = (
            canonical_r1_root / point["workload"]
            / f"l1d_{point['l1d_size']}" / f"l2_{point['l2_size']}"
        ).resolve()
        periodic = (
            periodic_r1_root
            / f"{point['workload']}_{point['l1d_size']}_{point['l2_size']}"
        ).resolve()
        _require_canonical_point(canonical, point)
        if not transient_r1_completed(periodic, SAMPLE_INTERVAL_MS, canonical):
            raise ValueError(
                f"point {_point_key(point)} lacks a compatible 2 ms periodic R1"
            )
        periodic_metadata = read_json(periodic / "r1_metadata.json")
        if any(
            periodic_metadata.get(field) != point[field]
            for field in ("workload", "l1d_size", "l2_size")
        ):
            raise ValueError(f"periodic R1 metadata differs from {_point_key(point)}")
        validated_points.append({
            **point,
            "key": _point_key(point),
            "canonical_r1": str(canonical),
            "periodic_r1": str(periodic),
            "steady_baseline": _matching_baseline(baseline, point),
            "input_sha256": {
                "canonical_status": sha256_file(canonical / "status.json"),
                "canonical_metadata": sha256_file(canonical / "r1_metadata.json"),
                "canonical_stats": sha256_file(canonical / "stats.txt"),
                "periodic_status": sha256_file(periodic / "status.json"),
                "periodic_metadata": sha256_file(periodic / "r1_metadata.json"),
                "periodic_stats": sha256_file(periodic / "stats.txt"),
            },
        })

    return {
        "schema_version": 1,
        "name": "transient_rom_balanced2",
        "non_formal": True,
        "paper_equivalent": False,
        "sample_interval_ms": SAMPLE_INTERVAL_MS,
        "selection": str(selection_path),
        "config": str(config_path),
        "steady_baseline_csv": str(steady_baseline_csv),
        "input_sha256": {
            "selection": sha256_file(selection_path),
            "config": sha256_file(config_path),
            "steady_baseline_csv": sha256_file(steady_baseline_csv),
        },
        "points": validated_points,
    }


def _finite_positive(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _point_slug(point: dict) -> str:
    return f"{point['workload']}_{point['l1d_size']}_{point['l2_size']}"


def _summary_path(point_output: Path) -> Path:
    return point_output / "transient_rom/transient_rom_summary.json"


def _require_summary_identity(summary: dict, point: dict, *, r2: bool) -> None:
    expected_state = "success" if r2 else "validated"
    if (
        summary.get("schema_version") != 1
        or summary.get("thermal_mode") != "transient-rom"
        or summary.get("non_formal") is not True
        or summary.get("paper_equivalent") is not False
        or summary.get("state") != expected_state
        or summary.get("sample_interval_ms") != SAMPLE_INTERVAL_MS
        or Path(str(summary.get("source_r1", ""))).resolve()
        != Path(point["canonical_r1"]).resolve()
        or Path(str(summary.get("transient_r1", ""))).resolve()
        != Path(point["periodic_r1"]).resolve()
    ):
        raise ValueError(
            f"{_point_key(point)} {'R2' if r2 else 'thermal'} checkpoint identity differs"
        )
    acceptance = summary.get("rom_acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise ValueError(f"{_point_key(point)} checkpoint lacks accepted ROM evidence")
    if (
        summary.get("training_hotspot_calls") != 8
        or summary.get("holdout_initialization_hotspot_calls") != 2
        or summary.get("holdout_transient_hotspot_calls") != 2
        or summary.get("calibration_hotspot_calls") != 12
        or summary.get("optimizer_hotspot_calls") != 0
    ):
        raise ValueError(f"{_point_key(point)} checkpoint call accounting differs")
    initialization_calls = summary.get("final_initialization_hotspot_calls")
    transient_calls = summary.get("final_transient_hotspot_calls")
    total_calls = summary.get("final_validation_hotspot_calls")
    if (
        isinstance(initialization_calls, bool)
        or not isinstance(initialization_calls, int)
        or initialization_calls < 2
        or isinstance(transient_calls, bool)
        or not isinstance(transient_calls, int)
        or transient_calls != initialization_calls
        or total_calls != initialization_calls + transient_calls
    ):
        raise ValueError(f"{_point_key(point)} final HotSpot accounting differs")


def _validated_branch(summary: dict, key: str, *, r2: bool) -> dict:
    branches = summary.get("branches")
    branch = branches.get(key) if isinstance(branches, dict) else None
    expected_name = "fixed-bin" if key == "fixed_bin" else "clip3d"
    if (
        not isinstance(branch, dict)
        or branch.get("branch") != expected_name
        or branch.get("failure") is not None
        or branch.get("validation_classification") != "validated"
    ):
        raise ValueError(f"{expected_name} thermal branch is not validated")
    _finite_positive(
        branch.get("validated_f_sus_trans_hotspot_ghz"),
        f"{expected_name} validated frequency",
    )
    if r2:
        ipc = _finite_positive(branch.get("measured_ipc2"), f"{expected_name} IPC2")
        bips = _finite_positive(
            branch.get("measured_bips2_trans"), f"{expected_name} BIPS2_trans"
        )
        frequency = float(branch["validated_f_sus_trans_hotspot_ghz"])
        if (
            branch.get("r2_requested") is not True
            or branch.get("r2_executed") is not True
            or not math.isclose(bips, ipc * frequency, rel_tol=1e-12)
        ):
            raise ValueError(f"{expected_name} measured evidence is inconsistent")
    elif (
        branch.get("r2_requested") is not False
        or branch.get("r2_executed") is not False
        or branch.get("measured_ipc2") is not None
        or branch.get("measured_bips2_trans") is not None
    ):
        raise ValueError(f"{expected_name} thermal checkpoint contains R2 evidence")
    return branch


def validate_thermal_checkpoint(point_output: Path, point: dict) -> dict:
    """Require a complete paired real-HotSpot checkpoint without R2."""
    point_output = Path(point_output).resolve()
    summary_path = _summary_path(point_output)
    if not summary_path.is_file():
        raise ValueError(f"thermal checkpoint missing for {_point_key(point)}")
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"thermal checkpoint is malformed for {_point_key(point)}")
    _require_summary_identity(summary, point, r2=False)
    _validated_branch(summary, "fixed_bin", r2=False)
    _validated_branch(summary, "clip3d", r2=False)
    package_value = summary.get("rom_package")
    if not isinstance(package_value, str):
        raise ValueError(f"thermal checkpoint lacks a ROM package for {_point_key(point)}")
    package = Path(package_value).resolve()
    acceptance = package / "rom_acceptance.json"
    if not acceptance.is_file() or read_json(acceptance).get("accepted") is not True:
        raise ValueError(f"thermal checkpoint ROM package is invalid for {_point_key(point)}")
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "rom_package": str(package),
        "rom_acceptance_sha256": sha256_file(acceptance),
    }


def validate_r2_checkpoint(point_output: Path, point: dict) -> dict:
    """Require paired measured R2 evidence for one transient-ROM point."""
    point_output = Path(point_output).resolve()
    summary_path = _summary_path(point_output)
    if not summary_path.is_file():
        raise ValueError(f"R2 checkpoint missing for {_point_key(point)}")
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"R2 checkpoint is malformed for {_point_key(point)}")
    _require_summary_identity(summary, point, r2=True)
    fixed = _validated_branch(summary, "fixed_bin", r2=True)
    clip = _validated_branch(summary, "clip3d", r2=True)
    paired_path = point_output / "transient_rom/final_validation/paired_comparison.json"
    if not paired_path.is_file():
        raise ValueError(f"R2 checkpoint lacks paired comparison for {_point_key(point)}")
    paired = read_json(paired_path)
    fixed_bips = float(fixed["measured_bips2_trans"])
    clip_bips = float(clip["measured_bips2_trans"])
    expected = (clip_bips / fixed_bips - 1.0) * 100.0
    if (
        not isinstance(paired, dict)
        or paired.get("non_formal") is not True
        or paired.get("paper_equivalent") is not False
        or not math.isclose(
            _finite_number(
                paired.get("bips2_trans_improvement_percent"),
                "paired BIPS2_trans improvement",
            ),
            expected,
            rel_tol=1e-12,
        )
    ):
        raise ValueError(f"R2 paired comparison differs for {_point_key(point)}")
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "paired": paired,
        "paired_path": str(paired_path.resolve()),
        "paired_sha256": sha256_file(paired_path),
    }


def _default_invoke(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        return subprocess.run(command, stdout=stdout, stderr=stderr).returncode


def _run_one(
    point: dict,
    output_root: Path,
    config_path: Path,
    *,
    execute_r2: bool,
    invoke: Invoke,
) -> dict:
    phase = "r2" if execute_r2 else "thermal"
    slug = _point_slug(point)
    point_output = output_root / phase / slug
    validator = validate_r2_checkpoint if execute_r2 else validate_thermal_checkpoint
    summary_path = _summary_path(point_output)
    if summary_path.is_file():
        return validator(point_output, point)
    if point_output.exists() and any(point_output.iterdir()):
        raise ValueError(
            f"{phase} checkpoint directory is nonempty but incomplete: {point_output}"
        )

    command = [
        sys.executable,
        "-m", "workflow.run_lifting_pipeline",
        "--r1-dir", point["canonical_r1"],
        "--output-dir", str(point_output.resolve()),
        "--config", str(config_path.resolve()),
        "--thermal-mode", "transient-rom",
        "--transient-rom-r1-dir", point["periodic_r1"],
    ]
    if execute_r2:
        package = (
            output_root / "thermal" / slug
            / "transient_rom/rom_package"
        ).resolve()
        command.extend(("--transient-rom-package-dir", str(package), "--run-r2"))
    else:
        command.append("--transient-rom-calibrate")

    status_path = output_root / "status" / f"{phase}_{slug}.json"
    write_json(status_path, {
        "state": "running",
        "phase": phase,
        "point": point["key"],
        "command": command,
    })
    log_root = output_root / "logs"
    return_code = invoke(
        command, log_root / f"{phase}_{slug}.stdout.log",
        log_root / f"{phase}_{slug}.stderr.log",
    )
    if return_code != 0:
        write_json(status_path, {
            "state": "failed", "phase": phase, "point": point["key"],
            "return_code": return_code, "command": command,
        })
        raise RuntimeError(f"{phase} pipeline failed for {point['key']}: rc={return_code}")
    try:
        checkpoint = validator(point_output, point)
    except (OSError, ValueError) as error:
        write_json(status_path, {
            "state": "failed", "phase": phase, "point": point["key"],
            "return_code": return_code, "command": command,
            "error": f"{type(error).__name__}: {error}",
        })
        raise
    write_json(status_path, {
        "state": f"{phase}_validated",
        "phase": phase,
        "point": point["key"],
        "return_code": return_code,
        "command": command,
        "summary": checkpoint["summary_path"],
        "summary_sha256": checkpoint["summary_sha256"],
    })
    return checkpoint


def run_validation_set(
    selection_path: Path,
    canonical_r1_root: Path,
    periodic_r1_root: Path,
    steady_baseline_csv: Path,
    config_path: Path,
    output_root: Path,
    *,
    execute_r2: bool = False,
    invoke: Invoke | None = None,
) -> dict:
    """Run or strictly resume the selected thermal or R2 validation phase."""
    inputs = preflight_inputs(
        selection_path, canonical_r1_root, periodic_r1_root,
        steady_baseline_csv, config_path,
    )
    output_root = Path(output_root).resolve()
    for read_only in (
        Path(selection_path).resolve(), Path(canonical_r1_root).resolve(),
        Path(periodic_r1_root).resolve(), Path(steady_baseline_csv).resolve(),
        Path(config_path).resolve(),
    ):
        if output_root == read_only or output_root in read_only.parents or read_only in output_root.parents:
            raise ValueError("Balanced-2 output root overlaps a read-only input")

    if execute_r2:
        try:
            thermal = [
                validate_thermal_checkpoint(
                    output_root / "thermal" / _point_slug(point), point,
                )
                for point in inputs["points"]
            ]
        except (OSError, ValueError) as error:
            raise ValueError(
                "both thermal points must validate before R2"
            ) from error
        if len(thermal) != len(inputs["points"]):
            raise ValueError("both thermal points must validate before R2")

    selected_invoke = invoke or _default_invoke
    checkpoints = []
    for point in inputs["points"]:
        checkpoints.append(_run_one(
            point, output_root, Path(config_path), execute_r2=execute_r2,
            invoke=selected_invoke,
        ))
    state = "r2_validated" if execute_r2 else "thermal_validated"
    result = {
        "schema_version": 1,
        "state": state,
        "non_formal": True,
        "paper_equivalent": False,
        "point_count": len(checkpoints),
        "points": [
            {
                "key": point["key"],
                "summary": checkpoint["summary_path"],
                "summary_sha256": checkpoint["summary_sha256"],
            }
            for point, checkpoint in zip(inputs["points"], checkpoints)
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "status.json", result)
    return result
