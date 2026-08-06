#!/usr/bin/env python3
"""Run the fixed-bin and CLIP-3D R2 points as resumable ordered pairs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import math
from pathlib import Path
import sys
import time

from workflow.common import read_json, sha256_file, write_json
from workflow.experiments.balanced50 import (
    load_selection,
    selection_keys,
    validate_layout_roots,
)
from workflow.r1_catalog import ArchitectureKey
from workflow.r2 import attach_result, reuse_result, run_r2


SCHEMA_VERSION = 1


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _paths(key: ArchitectureKey, r1_root: Path, fixed_root: Path,
           clip_root: Path, status_root: Path) -> dict[str, Path]:
    relative = key.relative_path()
    return {
        "r1": Path(r1_root).resolve() / relative,
        "fixed": Path(fixed_root).resolve() / relative,
        "clip": Path(clip_root).resolve() / relative,
        "status": Path(status_root).resolve() / relative / "pair_status.json",
    }


def _key_dict(key: ArchitectureKey) -> dict[str, str]:
    return asdict(key)


def _read_object(path: Path) -> dict | None:
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _finite_positive(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) > 0)


def _same_number(left: object, right: object) -> bool:
    return (_finite_positive(left) and _finite_positive(right)
            and math.isclose(float(left), float(right), rel_tol=1e-12,
                             abs_tol=1e-12))


def _validated_vectors(fixed_point: Path, clip_point: Path) -> tuple[dict, dict]:
    fixed = read_json(fixed_point / "r2_latency.json")
    clip = read_json(clip_point / "r2_latency.json")
    if not isinstance(fixed, dict) or not isinstance(clip, dict):
        raise ValueError("R2 latency vectors must contain objects")
    for label, vector in (("fixed-bin", fixed), ("CLIP-3D", clip)):
        arguments = run_r2.canonical_gem5_args(vector.get("gem5_overrides"))
        if vector.get("gem5_args") != arguments:
            raise ValueError(
                f"{label} gem5_args are not the canonical override rendering"
            )
    return fixed, clip


def _summary_metrics(point: Path, result: dict, label: str) -> dict[str, float]:
    summary = read_json(point / "pipeline_summary.json")
    if not isinstance(summary, dict):
        raise ValueError(f"{label} pipeline summary must contain an object")
    ipc2 = summary.get("ipc2")
    bips2 = summary.get("bips2")
    if not _finite_positive(ipc2) or not _finite_positive(bips2):
        raise ValueError(f"{label} summary lacks finite positive IPC2/BIPS2")
    if not _same_number(ipc2, result.get("ipc2")):
        raise ValueError(f"{label} summary IPC2 differs from validated R2 result")
    return {"ipc2": float(ipc2), "bips2": float(bips2)}


def _identity(record: dict, key: ArchitectureKey, paths: dict[str, Path],
              config_path: Path) -> bool:
    try:
        return (
            record.get("schema_version") == SCHEMA_VERSION
            and record.get("key") == _key_dict(key)
            and record.get("config") == str(config_path)
            and record.get("config_sha256") == sha256_file(config_path)
            and record.get("fixed_vector_sha256")
            == sha256_file(paths["fixed"] / "r2_latency.json")
            and record.get("clip3d_vector_sha256")
            == sha256_file(paths["clip"] / "r2_latency.json")
            and record.get("pair_status") == str(paths["status"])
        )
    except OSError:
        return False


def _validate_local_attachment(r1_dir: Path, point: Path,
                               label: str) -> tuple[dict, dict[str, float]] | None:
    """Bind an attached local summary/performance to its validated gem5 result."""
    decision = run_r2.validate_local_result(
        r1_dir, point / "r2_latency.json", point / "gem5_r2",
    )
    if decision.get("accepted") is not True:
        return None
    result = decision.get("result")
    if not isinstance(result, dict):
        return None
    result_path = (point / "gem5_r2/r2_result.json").resolve()
    summary = _read_object(point / "pipeline_summary.json")
    performance = _read_object(point / "performance.json")
    if summary is None or performance is None:
        return None
    try:
        metrics = _summary_metrics(point, result, label)
    except (OSError, ValueError):
        return None
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    if (summary.get("r2_source") != str(result_path)
            or artifacts.get("r2_result") != str(result_path)
            or summary.get("r2_reused") is True
            or summary.get("r2_reuse_artifact") is not None
            or artifacts.get("r2_reuse") is not None
            or (point / "r2_reuse.json").exists()
            or not _same_number(performance.get("ipc2"), metrics["ipc2"])
            or not _same_number(performance.get("bips2"), metrics["bips2"])
            or not _finite_positive(performance.get("sustainable_frequency_ghz"))
            or not math.isclose(
                metrics["bips2"],
                metrics["ipc2"] * float(
                    performance["sustainable_frequency_ghz"]
                ),
                rel_tol=1e-12, abs_tol=1e-12,
            )):
        return None
    return result, metrics


def _validate_fixed(paths: dict[str, Path]) -> tuple[dict, dict[str, float]] | None:
    return _validate_local_attachment(paths["r1"], paths["fixed"], "fixed-bin")


def _reuse_attachment_matches(point: Path, decision: dict,
                              summary: dict) -> bool:
    """Bind the committed marker to Task 5's freshly validated provenance."""
    marker_path = point / "r2_reuse.json"
    marker = _read_object(marker_path)
    expected = decision.get("artifact")
    if not isinstance(marker, dict) or not isinstance(expected, dict):
        return False
    if any(marker.get(field) != value for field, value in expected.items()):
        return False
    outputs = marker.get("target_outputs")
    if not isinstance(outputs, dict):
        return False
    performance_path = point / "performance.json"
    summary_path = point / "pipeline_summary.json"
    performance = _read_object(performance_path)
    source_ipc2 = expected.get("source_ipc2")
    if performance is None or not _finite_positive(source_ipc2):
        return False
    frequency = performance.get("sustainable_frequency_ghz")
    if not _finite_positive(frequency):
        return False
    expected_bips2 = float(source_ipc2) * float(frequency)
    try:
        if outputs.get("performance") != {
                "path": str(performance_path.resolve()),
                "sha256": sha256_file(performance_path),
        }:
            return False
        if outputs.get("summary") != {
                "path": str(summary_path.resolve()),
                "sha256": sha256_file(summary_path),
        }:
            return False
    except OSError:
        return False
    return (
        summary.get("r2_reused") is True
        and summary.get("r2_reuse_artifact") == str(marker_path.resolve())
        and summary.get("r2_source") == expected["source_result"]["path"]
        and summary.get("artifacts", {}).get("r2_result")
        == expected["source_result"]["path"]
        and summary.get("artifacts", {}).get("r2_reuse")
        == str(marker_path.resolve())
        and _same_number(performance.get("ipc2"), source_ipc2)
        and _same_number(summary.get("ipc2"), source_ipc2)
        and _same_number(performance.get("bips2"), expected_bips2)
        and _same_number(summary.get("bips2"), expected_bips2)
        and _same_number(
            summary.get("sustainable_frequency_ghz"), frequency
        )
        and _same_number(outputs.get("bips2"), expected_bips2)
    )


def _can_resume_fixed(record: dict | None, key: ArchitectureKey,
                      paths: dict[str, Path], config_path: Path) -> bool:
    return bool(
        isinstance(record, dict)
        and record.get("fixed_complete") is True
        and _identity(record, key, paths, config_path)
        and _validate_fixed(paths) is not None
    )


def _validate_completed(record: dict | None, key: ArchitectureKey,
                        paths: dict[str, Path], config_path: Path) -> bool:
    if (not isinstance(record, dict) or record.get("state") != "success"
            or not _identity(record, key, paths, config_path)):
        return False
    fixed = _validate_fixed(paths)
    if fixed is None:
        return False
    fixed_result, fixed_metrics = fixed
    try:
        fixed_vector, clip_vector = _validated_vectors(paths["fixed"], paths["clip"])
    except (KeyError, OSError, TypeError, ValueError):
        return False
    reused = fixed_vector["gem5_overrides"] == clip_vector["gem5_overrides"]
    if record.get("clip3d_reused_fixed_r2") is not reused:
        return False

    clip_summary = _read_object(paths["clip"] / "pipeline_summary.json")
    if clip_summary is None:
        return False
    if reused:
        if not (paths["clip"] / "r2_reuse.json").is_file():
            return False
        decision = reuse_result.validate_reuse(
            paths["fixed"], paths["clip"], paths["r1"], config_path
        )
        if (decision.get("accepted") is not True
                or not _reuse_attachment_matches(
                    paths["clip"], decision, clip_summary
                )):
            return False
        clip_result = fixed_result
    else:
        clip_validated = _validate_local_attachment(
            paths["r1"], paths["clip"], "CLIP-3D"
        )
        if clip_validated is None:
            return False
        clip_result, _validated_clip_metrics = clip_validated
    try:
        clip_metrics = _summary_metrics(paths["clip"], clip_result, "CLIP-3D")
    except (OSError, ValueError):
        return False
    return (
        record.get("physical_r2_runs") == (1 if reused else 2)
        and _same_number(record.get("fixed_ipc2"), fixed_metrics["ipc2"])
        and _same_number(record.get("fixed_bips2"), fixed_metrics["bips2"])
        and _same_number(record.get("clip3d_ipc2"), clip_metrics["ipc2"])
        and _same_number(record.get("clip3d_bips2"), clip_metrics["bips2"])
    )


def _base_record(key: ArchitectureKey, paths: dict[str, Path],
                 config_path: Path, started: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "running",
        "phase": "fixed",
        "key": _key_dict(key),
        "started_unix": started,
        "updated_unix": started,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "r1_directory": str(paths["r1"]),
        "fixed_point": str(paths["fixed"]),
        "clip3d_point": str(paths["clip"]),
        "fixed_vector": str((paths["fixed"] / "r2_latency.json").resolve()),
        "fixed_vector_sha256": sha256_file(paths["fixed"] / "r2_latency.json"),
        "clip3d_vector": str((paths["clip"] / "r2_latency.json").resolve()),
        "clip3d_vector_sha256": sha256_file(paths["clip"] / "r2_latency.json"),
        "pair_status": str(paths["status"]),
        "fixed_complete": False,
    }


def _clear_reuse_attachment(point: Path) -> None:
    """Remove stale reuse commit state before publishing a local attachment."""
    (point / "r2_reuse.json").unlink(missing_ok=True)
    summary_path = point / "pipeline_summary.json"
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("CLIP-3D pipeline summary must contain an object")
    summary.pop("r2_reused", None)
    summary.pop("r2_reuse_artifact", None)
    artifacts = summary.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.pop("r2_reuse", None)
    write_json(summary_path, summary)


def run_pair(key: ArchitectureKey, r1_root: Path, fixed_root: Path,
             clip_root: Path, config_path: Path, rerun: bool = False,
             *, status_root: Path | None = None) -> dict:
    """Run one fixed-first pair, returning a persisted success or failure record."""
    key = ArchitectureKey(key.workload, key.l1d_size, key.l2_size)
    r1_root = Path(r1_root).resolve()
    fixed_root = Path(fixed_root).resolve()
    clip_root = Path(clip_root).resolve()
    config_path = Path(config_path).resolve()
    if status_root is None:
        status_root = fixed_root.parent / "paired_r2_status"
    paths = _paths(key, r1_root, fixed_root, clip_root, Path(status_root))
    previous = _read_object(paths["status"])
    if not rerun and _validate_completed(previous, key, paths, config_path):
        return previous

    started = time.time()
    try:
        record = _base_record(key, paths, config_path, started)
    except Exception as error:
        record = {
            "schema_version": SCHEMA_VERSION,
            "state": "failed",
            "key": _key_dict(key),
            "started_unix": started,
            "finished_unix": time.time(),
            "pair_status": str(paths["status"]),
            "fixed_complete": False,
            "error": f"{type(error).__name__}: {error}",
        }
        write_json(paths["status"], record)
        return record

    resume_fixed = not rerun and _can_resume_fixed(
        previous, key, paths, config_path
    )
    write_json(paths["status"], record)
    try:
        if resume_fixed:
            fixed_validated = _validate_fixed(paths)
            if fixed_validated is None:  # Defensive against replacement after check.
                raise ValueError("fixed R2 changed after resume validation")
            fixed_result, fixed_metrics = fixed_validated
        else:
            fixed_result = run_r2.run(
                paths["r1"], paths["fixed"] / "r2_latency.json",
                paths["fixed"] / "gem5_r2", rerun=rerun,
            )
            attach_result.attach(paths["fixed"])
            fixed_validated = _validate_fixed(paths)
            if fixed_validated is None:
                raise ValueError("fixed R2 failed post-attachment validation")
            fixed_result, fixed_metrics = fixed_validated

        record.update({
            "phase": "clip-decision",
            "fixed_complete": True,
            "fixed_ipc2": fixed_metrics["ipc2"],
            "fixed_bips2": fixed_metrics["bips2"],
            "fixed_r2_result": str(
                (paths["fixed"] / "gem5_r2/r2_result.json").resolve()
            ),
            "fixed_r2_status": str(
                (paths["fixed"] / "gem5_r2/status.json").resolve()
            ),
            "updated_unix": time.time(),
        })
        write_json(paths["status"], record)

        fixed_vector, clip_vector = _validated_vectors(
            paths["fixed"], paths["clip"]
        )
        reused = fixed_vector["gem5_overrides"] == clip_vector["gem5_overrides"]
        if reused:
            decision = reuse_result.validate_reuse(
                paths["fixed"], paths["clip"], paths["r1"], config_path
            )
            if decision.get("accepted") is not True:
                raise ValueError(
                    "exact-vector R2 reuse rejected: "
                    + "; ".join(decision.get("reasons", []))
                )
            reuse_result.attach_reused_result(
                paths["fixed"], paths["clip"], paths["r1"], config_path
            )
            if not (paths["clip"] / "r2_reuse.json").is_file():
                raise ValueError("reuse attachment did not publish r2_reuse.json")
            decision = reuse_result.validate_reuse(
                paths["fixed"], paths["clip"], paths["r1"], config_path
            )
            if decision.get("accepted") is not True:
                raise ValueError(
                    "attached R2 reuse failed validation: "
                    + "; ".join(decision.get("reasons", []))
                )
            clip_summary = _read_object(paths["clip"] / "pipeline_summary.json")
            if (clip_summary is None
                    or not _reuse_attachment_matches(
                        paths["clip"], decision, clip_summary
                    )):
                raise ValueError(
                    "reuse attachment marker/output binding is invalid"
                )
            clip_result = fixed_result
            clip_artifact = paths["clip"] / "r2_reuse.json"
        else:
            _clear_reuse_attachment(paths["clip"])
            run_r2.run(
                paths["r1"], paths["clip"] / "r2_latency.json",
                paths["clip"] / "gem5_r2", rerun=rerun,
            )
            attach_result.attach(paths["clip"])
            clip_validated = _validate_local_attachment(
                paths["r1"], paths["clip"], "CLIP-3D"
            )
            if clip_validated is None:
                raise ValueError(
                    "CLIP-3D local attachment failed result/performance/summary "
                    "validation"
                )
            clip_result, _clip_metrics = clip_validated
            clip_artifact = paths["clip"] / "gem5_r2/r2_result.json"

        clip_metrics = _summary_metrics(paths["clip"], clip_result, "CLIP-3D")
        record.update({
            "state": "success",
            "phase": "complete",
            "finished_unix": time.time(),
            "updated_unix": time.time(),
            "clip3d_reused_fixed_r2": reused,
            "physical_r2_runs": 1 if reused else 2,
            "clip3d_ipc2": clip_metrics["ipc2"],
            "clip3d_bips2": clip_metrics["bips2"],
            "clip3d_r2_artifact": str(clip_artifact.resolve()),
            "artifacts": {
                "fixed_summary": str(
                    (paths["fixed"] / "pipeline_summary.json").resolve()
                ),
                "fixed_r2_result": str(
                    (paths["fixed"] / "gem5_r2/r2_result.json").resolve()
                ),
                "clip3d_summary": str(
                    (paths["clip"] / "pipeline_summary.json").resolve()
                ),
                "clip3d_r2": str(clip_artifact.resolve()),
            },
        })
        write_json(paths["status"], record)
        return record
    except Exception as error:
        record.update({
            "state": "failed",
            "phase": record.get("phase", "fixed"),
            "finished_unix": time.time(),
            "updated_unix": time.time(),
            "error": f"{type(error).__name__}: {error}",
        })
        write_json(paths["status"], record)
        return record


def _existing_r2_validator(r1_root: Path, fixed_root: Path, clip_root: Path,
                           config_path: Path,
                           rerun_requested: bool = False):
    """Build the Task 4 callback solely from the Task 5 provenance validators."""
    r1_root = Path(r1_root).resolve()
    fixed_root = Path(fixed_root).resolve()
    clip_root = Path(clip_root).resolve()
    config_path = Path(config_path).resolve()

    def validate(method: str, key: ArchitectureKey, point_dir: Path) -> dict:
        r1_dir = r1_root / key.relative_path()
        point_dir = Path(point_dir).resolve()
        fixed_point = fixed_root / key.relative_path()
        clip_point = clip_root / key.relative_path()
        if method not in ("fixed-bin", "clip3d"):
            return {"accepted": False, "reasons": [f"unknown method {method}"]}
        try:
            fixed_vector, clip_vector = _validated_vectors(
                fixed_point, clip_point
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            return {
                "accepted": False,
                "reasons": [f"cannot validate paired latency vectors: {error}"],
            }
        equal_overrides = (
            fixed_vector["gem5_overrides"] == clip_vector["gem5_overrides"]
        )
        branch = "reuse" if equal_overrides else "local"
        if rerun_requested:
            # Accepted here means safe to schedule replacement; the existing
            # scientific evidence remains explicitly uncertified.
            return {
                "accepted": True,
                "reasons": [],
                "branch": "fixed" if method == "fixed-bin" else branch,
                "repair_required": True,
                "rerun_requested": True,
                "scientific_evidence_accepted": False,
            }
        if method == "fixed-bin":
            decision = run_r2.validate_local_result(
                r1_dir, point_dir / "r2_latency.json", point_dir / "gem5_r2"
            )
            return {**decision, "branch": "fixed",
                    "repair_required": False,
                    "rerun_requested": False,
                    "scientific_evidence_accepted": decision.get("accepted") is True}
        if not equal_overrides:
            decision = run_r2.validate_local_result(
                r1_dir, point_dir / "r2_latency.json", point_dir / "gem5_r2"
            )
            if decision.get("accepted") is not True:
                return {**decision, "branch": "local",
                        "repair_required": False,
                        "rerun_requested": False,
                        "scientific_evidence_accepted": False}
            attachment_valid = _validate_local_attachment(
                r1_dir, point_dir, "CLIP-3D"
            ) is not None
            if attachment_valid:
                return {**decision, "branch": "local",
                        "repair_required": False,
                        "rerun_requested": False,
                        "scientific_evidence_accepted": True}
            return {
                "accepted": True,
                "branch": "local",
                "repair_required": True,
                "rerun_requested": False,
                "scientific_evidence_accepted": False,
                "reasons": ["local CLIP attachment requires repair"],
                "local_result_validation": {
                    "accepted": True,
                    "result_path": decision.get("result_path"),
                    "status_path": decision.get("status_path"),
                },
            }

        source = run_r2.validate_local_result(
            r1_dir, fixed_point / "r2_latency.json", fixed_point / "gem5_r2"
        )
        if source.get("accepted") is not True:
            return {
                "accepted": False,
                "branch": "reuse",
                "repair_required": False,
                "rerun_requested": False,
                "scientific_evidence_accepted": False,
                "reasons": [f"source {reason}" for reason in source["reasons"]],
            }
        reuse = reuse_result.validate_reuse(
            fixed_point, point_dir, r1_dir, config_path
        )
        if reuse.get("accepted") is not True:
            return {**reuse, "branch": "reuse",
                    "repair_required": False,
                    "rerun_requested": False,
                    "scientific_evidence_accepted": False}
        summary = _read_object(point_dir / "pipeline_summary.json")
        attachment_valid = (
            summary is not None
            and _reuse_attachment_matches(point_dir, reuse, summary)
        )
        if attachment_valid:
            return {**reuse, "branch": "reuse",
                    "repair_required": False,
                    "rerun_requested": False,
                    "scientific_evidence_accepted": True}
        return {
            "accepted": True,
            "branch": "reuse",
            "repair_required": True,
            "rerun_requested": False,
            "scientific_evidence_accepted": False,
            "reasons": ["reuse attachment marker/output binding requires repair"],
            "source_validation": {
                "accepted": True,
                "result_path": source.get("result_path"),
                "status_path": source.get("status_path"),
            },
            "reuse_eligibility": reuse,
        }

    return validate


def _pending_pair(key: ArchitectureKey, status_root: Path) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "pending",
        "key": _key_dict(key),
        "pair_status": str(
            (status_root / key.relative_path() / "pair_status.json").resolve()
        ),
    }


def _experiment_status(keys: list[ArchitectureKey], results: dict[ArchitectureKey, dict],
                       status_root: Path, preflight: dict, limited_run: bool,
                       started: float) -> dict:
    pairs = [results.get(key, _pending_pair(key, status_root)) for key in keys]
    completed = sum(pair.get("state") == "success" for pair in pairs)
    failed = sum(pair.get("state") == "failed" for pair in pairs)
    reuse = sum(
        pair.get("state") == "success"
        and pair.get("clip3d_reused_fixed_r2") is True
        for pair in pairs
    )
    separate = sum(
        pair.get("state") == "success"
        and pair.get("clip3d_reused_fixed_r2") is False
        for pair in pairs
    )
    successful_pairs = [pair for pair in pairs if pair.get("state") == "success"]
    physical_values = [pair.get("physical_r2_runs") for pair in successful_pairs]
    physical_values_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in physical_values
    )
    physical_r2_runs = sum(physical_values) if physical_values_valid else 0
    per_pair_run_counts_valid = physical_values_valid and all(
        pair.get("physical_r2_runs")
        == (1 if pair.get("clip3d_reused_fixed_r2") is True else 2)
        for pair in successful_pairs
        if pair.get("clip3d_reused_fixed_r2") in (True, False)
    ) and reuse + separate == completed
    reported_keys = [
        pair.get("key") for pair in pairs if pair.get("state") == "success"
    ]
    keys_match_manifest = (
        len(reported_keys) == len(keys)
        and all(reported == _key_dict(key)
                for key, reported in zip(keys, reported_keys))
        and len({
            (reported.get("workload"), reported.get("l1d_size"),
             reported.get("l2_size"))
            for reported in reported_keys if isinstance(reported, dict)
        }) == len(keys)
    )
    selected_results_valid = (
        completed == len(keys)
        and failed == 0
        and keys_match_manifest
        and reuse + separate == len(keys)
        and per_pair_run_counts_valid
        and physical_r2_runs == len(keys) + separate
    )
    complete = (
        not limited_run
        and len(keys) == 50
        and len(set(keys)) == 50
        and selected_results_valid
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "success" if complete else ("failed" if failed else "running"),
        "complete": complete,
        "limited_run": limited_run,
        "started_unix": started,
        "updated_unix": time.time(),
        "status_root": str(status_root),
        "full_selected_pair_count": 50,
        "selected_pair_count": len(keys),
        "completed_pair_count": completed,
        "failed_pair_count": failed,
        "pending_pair_count": len(keys) - completed - failed,
        "reuse_pair_count": reuse,
        "separate_clip_r2_count": separate,
        "physical_r2_runs": physical_r2_runs,
        "per_pair_run_counts_valid": per_pair_run_counts_valid,
        "result_keys_match_manifest": keys_match_manifest,
        "selected_results_valid": selected_results_valid,
        "preflight": preflight,
        "pairs": pairs,
    }


def run_sweep(r1_root: Path, fixed_root: Path, clip_root: Path,
              selection_path: Path, config_path: Path, status_root: Path,
              jobs: int = 1, rerun: bool = False,
              limit: int | None = None) -> dict:
    """Validate both full roots, then concurrently run selected ordered pairs."""
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    r1_root = Path(r1_root).resolve()
    fixed_root = Path(fixed_root).resolve()
    clip_root = Path(clip_root).resolve()
    selection_path = Path(selection_path).resolve()
    config_path = Path(config_path).resolve()
    status_root = Path(status_root).resolve()
    selection = load_selection(selection_path)
    all_keys = selection_keys(selection)
    preflight = validate_layout_roots(
        fixed_root, clip_root, all_keys, config_path,
        selection=selection, require_layout_only=False,
        existing_r2_validator=_existing_r2_validator(
            r1_root, fixed_root, clip_root, config_path,
            rerun_requested=rerun,
        ),
    )
    keys = all_keys if limit is None else all_keys[:limit]
    limited_run = limit is not None
    started = time.time()
    results: dict[ArchitectureKey, dict] = {}
    experiment_path = status_root / "status.json"
    write_json(
        experiment_path,
        _experiment_status(keys, results, status_root, preflight, limited_run, started),
    )

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                run_pair, key, r1_root, fixed_root, clip_root, config_path, rerun,
                status_root=status_root,
            ): key
            for key in keys
        }
        for completed_count, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                result = future.result()
            except Exception as error:
                pair_path = status_root / key.relative_path() / "pair_status.json"
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "state": "failed",
                    "key": _key_dict(key),
                    "pair_status": str(pair_path.resolve()),
                    "fixed_complete": False,
                    "physical_r2_runs": 0,
                    "finished_unix": time.time(),
                    "error": f"{type(error).__name__}: {error}",
                }
                write_json(pair_path, result)
            results[key] = result
            status = _experiment_status(
                keys, results, status_root, preflight, limited_run, started
            )
            write_json(experiment_path, status)
            print(
                f"[{completed_count}/{len(keys)}] "
                f"{key.workload} {key.l1d_size} {key.l2_size}: "
                f"{result.get('state')}"
            )

    final = _experiment_status(
        keys, results, status_root, preflight, limited_run, started
    )
    final["finished_unix"] = time.time()
    if limited_run and final["selected_results_valid"]:
        final["state"] = "success"
    elif not final["complete"]:
        final["state"] = "failed"
    write_json(experiment_path, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status-root", type=Path, required=True)
    parser.add_argument("--jobs", type=_positive_int, default=1)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    result = run_sweep(
        args.r1_root, args.fixed_root, args.clip_root, args.selection,
        args.config, args.status_root, jobs=args.jobs, rerun=args.rerun,
        limit=args.limit,
    )
    if result.get("limited_run"):
        return 0 if result.get("selected_results_valid") is True else 1
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    sys.exit(main())
