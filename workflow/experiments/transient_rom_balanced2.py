#!/usr/bin/env python3
"""Run the exploratory MATMUL/STENCIL transient-ROM validation pair."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
import math
from numbers import Real
from pathlib import Path
import subprocess
import sys
from typing import Callable

from workflow.common import (
    PROJECT_ROOT,
    atomic_write_bytes,
    format_temperature_c,
    read_json,
    sha256_file,
    write_json,
)
from workflow.run_lifting_pipeline import validate_config
from workflow.transient.rom.contracts import (
    parse_settings,
    require_package_calibration_evidence,
    require_rom_artifact_manifest,
)
from workflow.transient.rom.evidence import ROM_CLASSIFICATION
from workflow.transient.run_transient_r1 import completed as transient_r1_completed


EXPECTED_POINTS = (
    {"workload": "matmul", "l1d_size": "64kB", "l2_size": "512kB"},
    {"workload": "stencil", "l1d_size": "64kB", "l2_size": "512kB"},
)
SAMPLE_INTERVAL_MS = 2.0
Invoke = Callable[[list[str], Path, Path], int]
DEFAULT_SELECTION = (
    PROJECT_ROOT / "configs/experiments/transient_rom_balanced2_selection.json"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/experiments/"
    "clip3d_transient_rom_lambda0020119_traffic_weighted_"
    "discrete_partition_exploratory.json"
)
DEFAULT_CANONICAL_R1_ROOT = PROJECT_ROOT / "runs/architecture_sweep/r1/paper"
DEFAULT_PERIODIC_R1_ROOT = (
    PROJECT_ROOT / "runs/transient_r1/balanced2_2ms_20260811"
)
DEFAULT_STEADY_BASELINE = (
    PROJECT_ROOT
    / "runs/discrete_partition_validation/balanced5_midcache_20260809/summary.csv"
)
CSV_FIELDS = (
    "workload", "l1d_size", "l2_size", "state",
    "steady_fixed_tmax_c", "steady_clip3d_tmax_c",
    "steady_fixed_frequency_ghz", "steady_clip3d_frequency_ghz",
    "steady_fixed_ipc2", "steady_clip3d_ipc2",
    "steady_fixed_bips2", "steady_clip3d_bips2",
    "steady_bips2_improvement_percent",
    "transient_fixed_frequency_ghz", "transient_clip3d_frequency_ghz",
    "transient_fixed_ipc2", "transient_clip3d_ipc2",
    "transient_fixed_bips2", "transient_clip3d_bips2",
    "transient_bips2_improvement_percent",
    "improvement_shift_percentage_points",
    "calibration_hotspot_calls", "final_validation_hotspot_calls",
    "thermal_summary", "r2_summary",
)


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

    global_hashes = {
        "selection": sha256_file(selection_path),
        "config": sha256_file(config_path),
        "steady_baseline_csv": sha256_file(steady_baseline_csv),
    }
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
        r1_hashes = {
            "canonical_status": sha256_file(canonical / "status.json"),
            "canonical_metadata": sha256_file(canonical / "r1_metadata.json"),
            "canonical_stats": sha256_file(canonical / "stats.txt"),
            "periodic_status": sha256_file(periodic / "status.json"),
            "periodic_metadata": sha256_file(periodic / "r1_metadata.json"),
            "periodic_stats": sha256_file(periodic / "stats.txt"),
        }
        validated_points.append({
            **point,
            "key": _point_key(point),
            "canonical_r1": str(canonical),
            "periodic_r1": str(periodic),
            "steady_baseline": _matching_baseline(baseline, point),
            "input_sha256": r1_hashes,
            "checkpoint_identity": {
                "selection": str(selection_path),
                "config": str(config_path),
                "steady_baseline_csv": str(steady_baseline_csv),
                "canonical_r1": str(canonical),
                "periodic_r1": str(periodic),
                "input_sha256": {**global_hashes, **r1_hashes},
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
        "input_sha256": global_hashes,
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


def _checkpoint_path(point_output: Path) -> Path:
    return point_output / "balanced2_checkpoint.json"


def _package_evidence(package: Path, point: dict) -> dict:
    package = Path(package).resolve()
    acceptance = package / "rom_acceptance.json"
    manifest = package / "rom_artifact_manifest.json"
    acceptance_record = read_json(acceptance) if acceptance.is_file() else None
    if (
        not acceptance.is_file()
        or not manifest.is_file()
        or not isinstance(acceptance_record, dict)
        or acceptance_record.get("schema_version") != 1
        or acceptance_record.get("accepted") is not True
        or any(
            acceptance_record.get(field) != value
            for field, value in ROM_CLASSIFICATION.items()
        )
    ):
        raise ValueError(
            f"thermal checkpoint ROM package is invalid for {_point_key(point)}"
        )
    require_rom_artifact_manifest(package)
    identity = acceptance_record.get("identity")
    checkpoint_identity = point.get("checkpoint_identity")
    input_hashes = (
        checkpoint_identity.get("input_sha256")
        if isinstance(checkpoint_identity, dict) else None
    )
    if not isinstance(identity, dict) or not isinstance(input_hashes, dict):
        raise ValueError(f"ROM package identity is invalid for {_point_key(point)}")
    expected_identities = {
        "configuration_hash": "sha256:" + str(input_hashes.get("config", "")),
        "canonical_r1_metadata_hash": (
            "sha256:" + str(input_hashes.get("canonical_metadata", ""))
        ),
        "r1_input_hashes": {
            field: "sha256:" + str(input_hashes.get(field, ""))
            for field in (
                "canonical_status", "canonical_metadata", "canonical_stats",
                "periodic_status", "periodic_metadata", "periodic_stats",
            )
        },
    }
    for field, expected in expected_identities.items():
        if identity.get(field) != expected:
            label = (
                "R1 input hash identity"
                if field == "r1_input_hashes" else field.replace("_", " ")
            )
            raise ValueError(f"ROM package {label} differs for {_point_key(point)}")
    config_value = checkpoint_identity.get("config")
    if not isinstance(config_value, str):
        raise ValueError(f"ROM package config identity is invalid for {_point_key(point)}")
    package_evidence = require_package_calibration_evidence(
        package, identity, parse_settings(read_json(Path(config_value).resolve()))
    )
    if package_evidence.get("acceptance") != acceptance_record:
        raise ValueError(
            f"ROM package calibration acceptance differs for {_point_key(point)}"
        )
    return {
        "rom_package": str(package),
        "rom_acceptance_sha256": sha256_file(acceptance),
        "rom_manifest_sha256": sha256_file(manifest),
        "rom_acceptance_record": acceptance_record,
    }


def _checkpoint_record(phase: str, point: dict, artifacts: dict) -> dict:
    identity = point.get("checkpoint_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{_point_key(point)} lacks checkpoint identity")
    return {
        "schema_version": 1,
        "phase": phase,
        "point": point["key"],
        "scientific_identity": identity,
        "artifacts": artifacts,
    }


def _bind_checkpoint(
    point_output: Path, phase: str, point: dict, artifacts: dict,
) -> None:
    write_json(
        _checkpoint_path(point_output),
        _checkpoint_record(phase, point, artifacts),
    )


def _require_bound_checkpoint(
    point_output: Path, phase: str, point: dict, artifacts: dict,
) -> dict:
    path = _checkpoint_path(point_output)
    if not path.is_file():
        raise ValueError(f"{phase} checkpoint identity is missing")
    record = read_json(path)
    expected = _checkpoint_record(phase, point, artifacts)
    if record != expected:
        raise ValueError(f"{phase} checkpoint scientific identity differs")
    return {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
    }


def _require_summary_identity(summary: dict, point: dict, *, r2: bool) -> None:
    expected_state = "success" if r2 else "validated"
    checkpoint_identity = point.get("checkpoint_identity")
    expected_config = (
        checkpoint_identity.get("config")
        if isinstance(checkpoint_identity, dict) else None
    )
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
        or not isinstance(expected_config, str)
        or Path(str(summary.get("config", ""))).resolve()
        != Path(expected_config).resolve()
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


def _thermal_artifacts(point_output: Path, point: dict) -> dict:
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
    expected_package = (point_output / "transient_rom/rom_package").resolve()
    if package != expected_package:
        raise ValueError(f"thermal ROM package differs for {_point_key(point)}")
    package_evidence = _package_evidence(package, point)
    if summary.get("rom_acceptance") != package_evidence["rom_acceptance_record"]:
        raise ValueError(f"thermal ROM acceptance differs for {_point_key(point)}")
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        **package_evidence,
    }


def validate_thermal_checkpoint(point_output: Path, point: dict) -> dict:
    """Require a complete scientifically bound HotSpot checkpoint without R2."""
    point_output = Path(point_output).resolve()
    result = _thermal_artifacts(point_output, point)
    bound = _require_bound_checkpoint(
        point_output,
        "thermal",
        point,
        {key: value for key, value in result.items() if key != "summary"},
    )
    return {**result, **bound}


def _r2_artifacts(point_output: Path, point: dict) -> dict:
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
    slug = _point_slug(point)
    expected_package = (
        point_output.parents[1] / "thermal" / slug
        / "transient_rom/rom_package"
    ).resolve()
    if Path(str(summary.get("rom_package", ""))).resolve() != expected_package:
        raise ValueError(f"R2 ROM package differs for {_point_key(point)}")
    package_evidence = _package_evidence(expected_package, point)
    if summary.get("rom_acceptance") != package_evidence["rom_acceptance_record"]:
        raise ValueError(f"R2 ROM acceptance differs for {_point_key(point)}")
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
        or paired.get("fixed_bin") != fixed
        or paired.get("clip3d") != clip
        or not math.isclose(
            _finite_number(
                paired.get("bips2_trans_improvement_percent"),
                "paired BIPS2_trans improvement",
            ),
            expected,
            rel_tol=1e-12,
        )
    ):
        raise ValueError(f"R2 paired branch comparison differs for {_point_key(point)}")
    thermal_checkpoint = _checkpoint_path(
        point_output.parents[1] / "thermal" / slug
    )
    if not thermal_checkpoint.is_file():
        raise ValueError(f"R2 thermal checkpoint differs for {_point_key(point)}")
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "paired": paired,
        "paired_path": str(paired_path.resolve()),
        "paired_sha256": sha256_file(paired_path),
        **package_evidence,
        "thermal_checkpoint": str(thermal_checkpoint.resolve()),
        "thermal_checkpoint_sha256": sha256_file(thermal_checkpoint),
    }


def validate_r2_checkpoint(point_output: Path, point: dict) -> dict:
    """Require scientifically bound paired measured R2 evidence."""
    point_output = Path(point_output).resolve()
    result = _r2_artifacts(point_output, point)
    bound = _require_bound_checkpoint(
        point_output,
        "r2",
        point,
        {
            key: value for key, value in result.items()
            if key not in ("summary", "paired")
        },
    )
    return {**result, **bound}


def _default_invoke(command: list[str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        return subprocess.run(command, stdout=stdout, stderr=stderr).returncode


def _attempt_paths(output_root: Path, phase: str, point: dict) -> tuple[Path, ...]:
    slug = _point_slug(point)
    return (
        output_root / phase / slug,
        output_root / "status" / f"{phase}_{slug}.json",
        output_root / "logs" / f"{phase}_{slug}.stdout.log",
        output_root / "logs" / f"{phase}_{slug}.stderr.log",
    )


def _has_attempt_evidence(output_root: Path, phase: str, point: dict) -> bool:
    for path in _attempt_paths(output_root, phase, point):
        if path.is_symlink() or path.is_file():
            return True
        if path.is_dir() and any(path.iterdir()):
            return True
    return False


def _require_resumable_phase(
    inputs: dict, output_root: Path, *, execute_r2: bool,
) -> None:
    phase = "r2" if execute_r2 else "thermal"
    validator = validate_r2_checkpoint if execute_r2 else validate_thermal_checkpoint
    for point in inputs["points"]:
        point_output = output_root / phase / _point_slug(point)
        if _summary_path(point_output).is_file():
            validator(point_output, point)
        elif _has_attempt_evidence(output_root, phase, point):
            raise ValueError(
                f"{phase} has an existing attempt for {_point_key(point)}; "
                "use a new output root"
            )


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
    artifact_validator = _r2_artifacts if execute_r2 else _thermal_artifacts
    summary_path = _summary_path(point_output)
    if summary_path.is_file():
        return validator(point_output, point)
    if _has_attempt_evidence(output_root, phase, point):
        raise ValueError(
            f"{phase} has an existing attempt for {_point_key(point)}; "
            "use a new output root"
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
        checkpoint = artifact_validator(point_output, point)
        bound_artifacts = {
            key: value for key, value in checkpoint.items()
            if key not in ("summary", "paired")
        }
        _bind_checkpoint(point_output, phase, point, bound_artifacts)
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
        "checkpoint": checkpoint["checkpoint_path"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
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

    _require_resumable_phase(inputs, output_root, execute_r2=execute_r2)

    selected_invoke = invoke or _default_invoke
    checkpoints = []
    phase = "r2" if execute_r2 else "thermal"
    root_status = {
        "schema_version": 1,
        "state": "running",
        "phase": phase,
        "non_formal": True,
        "paper_equivalent": False,
        "point_count": len(inputs["points"]),
        "completed_points": [],
    }
    write_json(output_root / "status.json", root_status)
    try:
        for point in inputs["points"]:
            checkpoints.append(_run_one(
                point, output_root, Path(config_path), execute_r2=execute_r2,
                invoke=selected_invoke,
            ))
            root_status["completed_points"].append(point["key"])
            write_json(output_root / "status.json", root_status)
    except Exception as error:
        root_status.update({
            "state": "failed",
            "error": f"{type(error).__name__}: {error}",
        })
        write_json(output_root / "status.json", root_status)
        raise
    state = "r2_validated" if execute_r2 else "thermal_validated"
    result = {
        "schema_version": 1,
        "state": state,
        "non_formal": True,
        "paper_equivalent": False,
        "point_count": len(checkpoints),
        "completed_points": [point["key"] for point in inputs["points"]],
        "points": [
            {
                "key": point["key"],
                "summary": checkpoint["summary_path"],
                "summary_sha256": checkpoint["summary_sha256"],
                "checkpoint": checkpoint["checkpoint_path"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            }
            for point, checkpoint in zip(inputs["points"], checkpoints)
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "status.json", result)
    return result


def _steady_metrics(row: dict[str, str], point: dict) -> dict:
    def positive(field: str) -> float:
        try:
            value: object = float(row[field])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"steady baseline {field} is invalid for {_point_key(point)}"
            ) from error
        return _finite_positive(value, f"steady baseline {field}")

    fixed_bips = positive("fixed_bips2")
    clip_bips = positive("clip3d_bips2")
    improvement = (clip_bips / fixed_bips - 1.0) * 100.0
    try:
        claimed: object = float(row["bips2_improvement_percent"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"steady baseline improvement is invalid for {_point_key(point)}"
        ) from error
    if not math.isclose(
        _finite_number(claimed, "steady baseline improvement"),
        improvement,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"steady baseline improvement differs for {_point_key(point)}")
    return {
        "steady_fixed_tmax_c": positive("fixed_tmax_c"),
        "steady_clip3d_tmax_c": positive("clip3d_tmax_c"),
        "steady_fixed_frequency_ghz": positive("fixed_frequency_ghz"),
        "steady_clip3d_frequency_ghz": positive("clip3d_frequency_ghz"),
        "steady_fixed_ipc2": positive("fixed_ipc2"),
        "steady_clip3d_ipc2": positive("clip3d_ipc2"),
        "steady_fixed_bips2": fixed_bips,
        "steady_clip3d_bips2": clip_bips,
        "steady_bips2_improvement_percent": improvement,
    }


def _csv_value(field: str, value: object) -> object:
    if value is None:
        return ""
    if field.endswith("_tmax_c"):
        return format_temperature_c(float(value))
    return value


def summarize_validation_set(
    inputs: dict,
    output_root: Path,
    require_r2: bool,
) -> dict:
    """Publish a deterministic steady/transient report from validated facts."""
    if (
        not isinstance(inputs, dict)
        or inputs.get("non_formal") is not True
        or inputs.get("paper_equivalent") is not False
        or not isinstance(inputs.get("points"), list)
        or len(inputs["points"]) != len(EXPECTED_POINTS)
    ):
        raise ValueError("Balanced-2 summarized inputs are invalid")
    output_root = Path(output_root).resolve()
    rows = []
    transient_gains: list[float] = []
    for point in inputs["points"]:
        slug = _point_slug(point)
        thermal = validate_thermal_checkpoint(
            output_root / "thermal" / slug, point,
        )
        checkpoint = (
            validate_r2_checkpoint(output_root / "r2" / slug, point)
            if require_r2 else thermal
        )
        summary = checkpoint["summary"]
        fixed = summary["branches"]["fixed_bin"]
        clip = summary["branches"]["clip3d"]
        steady = _steady_metrics(point["steady_baseline"], point)
        transient_fixed_ipc = fixed.get("measured_ipc2") if require_r2 else None
        transient_clip_ipc = clip.get("measured_ipc2") if require_r2 else None
        transient_fixed_bips = (
            fixed.get("measured_bips2_trans") if require_r2 else None
        )
        transient_clip_bips = (
            clip.get("measured_bips2_trans") if require_r2 else None
        )
        transient_gain = None
        gain_shift = None
        if require_r2:
            transient_fixed_bips = _finite_positive(
                transient_fixed_bips, "transient fixed BIPS2"
            )
            transient_clip_bips = _finite_positive(
                transient_clip_bips, "transient CLIP-3D BIPS2"
            )
            transient_gain = (
                transient_clip_bips / transient_fixed_bips - 1.0
            ) * 100.0
            paired_claim = checkpoint["paired"].get(
                "bips2_trans_improvement_percent"
            )
            if not math.isclose(
                _finite_number(paired_claim, "paired transient improvement"),
                transient_gain,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"paired improvement differs for {_point_key(point)}")
            gain_shift = transient_gain - steady["steady_bips2_improvement_percent"]
            transient_gains.append(transient_gain)
        rows.append({
            "workload": point["workload"],
            "l1d_size": point["l1d_size"],
            "l2_size": point["l2_size"],
            "state": "r2_validated" if require_r2 else "thermal_validated",
            **steady,
            "transient_fixed_frequency_ghz": _finite_positive(
                fixed.get("validated_f_sus_trans_hotspot_ghz"),
                "transient fixed validated frequency",
            ),
            "transient_clip3d_frequency_ghz": _finite_positive(
                clip.get("validated_f_sus_trans_hotspot_ghz"),
                "transient CLIP-3D validated frequency",
            ),
            "transient_fixed_ipc2": transient_fixed_ipc,
            "transient_clip3d_ipc2": transient_clip_ipc,
            "transient_fixed_bips2": transient_fixed_bips,
            "transient_clip3d_bips2": transient_clip_bips,
            "transient_bips2_improvement_percent": transient_gain,
            "improvement_shift_percentage_points": gain_shift,
            "calibration_hotspot_calls": summary["calibration_hotspot_calls"],
            "final_validation_hotspot_calls": summary[
                "final_validation_hotspot_calls"
            ],
            "thermal_summary": thermal["summary_path"],
            "r2_summary": checkpoint["summary_path"] if require_r2 else None,
            "artifact_sha256": {
                "thermal_summary": thermal["summary_sha256"],
                "r2_summary": checkpoint["summary_sha256"] if require_r2 else None,
                "paired_comparison": checkpoint.get("paired_sha256"),
            },
        })

    statistics = None
    if require_r2:
        statistics = {
            "wins": sum(value > 0.0 for value in transient_gains),
            "ties": sum(value == 0.0 for value in transient_gains),
            "losses": sum(value < 0.0 for value in transient_gains),
            "mean_bips2_improvement_percent": (
                sum(transient_gains) / len(transient_gains)
            ),
        }
    report = {
        "schema_version": 1,
        "name": "transient_rom_balanced2",
        "state": "r2_validated" if require_r2 else "thermal_validated",
        "non_formal": True,
        "paper_equivalent": False,
        "point_count": len(rows),
        "sample_interval_ms": inputs["sample_interval_ms"],
        "input_sha256": inputs["input_sha256"],
        "transient_statistics": statistics,
        "points": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "summary.json", report)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(field, row.get(field)) for field in CSV_FIELDS})
    atomic_write_bytes(
        output_root / "summary.csv", stream.getvalue().encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--canonical-r1-root", type=Path, default=DEFAULT_CANONICAL_R1_ROOT,
    )
    parser.add_argument(
        "--periodic-r1-root", type=Path, default=DEFAULT_PERIODIC_R1_ROOT,
    )
    parser.add_argument(
        "--steady-baseline-csv", type=Path, default=DEFAULT_STEADY_BASELINE,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-r2", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    inputs = preflight_inputs(
        args.selection, args.canonical_r1_root, args.periodic_r1_root,
        args.steady_baseline_csv, args.config,
    )
    if not args.summarize_only:
        run_validation_set(
            args.selection, args.canonical_r1_root, args.periodic_r1_root,
            args.steady_baseline_csv, args.config, args.output_root,
            execute_r2=args.run_r2,
        )
    report = summarize_validation_set(
        inputs, args.output_root, require_r2=args.run_r2,
    )
    print(
        f"Balanced-2 {report['state']}: {report['point_count']} points; "
        f"summary={args.output_root.resolve() / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()
