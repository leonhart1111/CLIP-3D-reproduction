#!/usr/bin/env python3
"""Run the fixed-bin and CLIP-3D R2 points as resumable ordered pairs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import asdict
import fcntl
import math
import os
from pathlib import Path
import sys
import time

from workflow.common import read_json, sha256_file, write_json
from workflow.experiments.balanced50 import (
    load_selection,
    selection_keys,
    validate_layout_point,
    validate_layout_roots,
)
from workflow.r1_catalog import ArchitectureKey
from workflow.r2 import attach_result, reuse_result, run_r2
from workflow.r2.attachment_validation import exclusive_point_lock


SCHEMA_VERSION = 1
SWEEP_LOCK_NAME = ".paired-r2-sweep.lock"
PAIR_LOCK_NAME = ".paired-r2-pair.lock"
_NATIVE_CACHE_AUTHORITY = "McPAT 1.3 embedded CACTI-P"

# Global acquisition order: physical-root sweep -> physical-point pair ->
# physical point attachment. Locks within one level use resolved path order.
# A lower-level operation must never attempt to reacquire an earlier lock.


@contextmanager
def _execution_lock(path: Path, *, blocking: bool, conflict_message: str,
                    shared: bool = False):
    """Hold one filesystem execution lock for the entire protected operation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    acquired = False
    try:
        flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, flags)
        except BlockingIOError as error:
            raise RuntimeError(conflict_message) from error
        acquired = True
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _ordered_lock_paths(paths: list[Path]) -> list[Path]:
    """Return deterministic unique physical lock identities."""
    unique = {str(Path(path).resolve()): Path(path).resolve() for path in paths}
    return [unique[name] for name in sorted(unique)]


@contextmanager
def _execution_locks(paths: list[Path], *, blocking: bool,
                     conflict_message: str, shared: bool = False):
    """Acquire one lock level in deterministic order and unwind atomically."""
    with ExitStack() as stack:
        for path in _ordered_lock_paths(paths):
            stack.enter_context(_execution_lock(
                path, blocking=blocking, conflict_message=conflict_message,
                shared=shared,
            ))
        yield


def _sweep_lock_paths(fixed_root: Path, clip_root: Path) -> list[Path]:
    return [
        Path(fixed_root).resolve() / SWEEP_LOCK_NAME,
        Path(clip_root).resolve() / SWEEP_LOCK_NAME,
    ]


def _pair_lock_paths(key: ArchitectureKey, fixed_root: Path,
                     clip_root: Path) -> list[Path]:
    relative = key.relative_path()
    return [
        Path(fixed_root).resolve() / relative / PAIR_LOCK_NAME,
        Path(clip_root).resolve() / relative / PAIR_LOCK_NAME,
    ]


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _require_native_preflight(preflight: dict) -> dict:
    """Reject paired execution unless both roots passed native validation."""
    if not isinstance(preflight, dict) or any(
            not isinstance(preflight.get(branch), dict)
            or preflight[branch].get("physical_model_authority") != (
                _NATIVE_CACHE_AUTHORITY)
            for branch in ("fixed", "clip3d")):
        raise ValueError(
            "paired R2 requires McPAT-native physical preflight authority"
        )
    return preflight


def _validate_direct_native_pair(
        key: ArchitectureKey, r1_root: Path, fixed_root: Path, clip_root: Path,
        config_path: Path, rerun: bool,
) -> None:
    """Apply the shared root contract before direct-pair mutation."""
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("direct paired R2 config must contain an object")
    existing_validator = _existing_r2_validator(
        Path(r1_root).resolve(), Path(fixed_root).resolve(),
        Path(clip_root).resolve(), config_path, rerun_requested=rerun,
    )
    for label, root in (("fixed", fixed_root), ("clip3d", clip_root)):
        point = Path(root).resolve() / key.relative_path()
        try:
            decision = validate_layout_point(
                point, "fixed-bin" if label == "fixed" else "clip3d",
                key, config, config_path, require_layout_only=False,
                existing_r2_validator=existing_validator,
                expected_r1=Path(r1_root).resolve() / key.relative_path(),
            )
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                f"direct paired R2 requires McPAT-native {label} artifacts: {error}"
            ) from error
        if decision.get("cache_authority") != _NATIVE_CACHE_AUTHORITY:
            raise ValueError(
                f"direct paired R2 requires McPAT-native {label} authority"
            )


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
        try:
            run_r2.validate_latency_vector(vector)
        except ValueError as error:
            raise ValueError(f"{label} {error}") from error
    return fixed, clip


def _summary_metrics(point: Path, result: dict, label: str,
                     summary: dict | None = None) -> dict[str, float]:
    summary = (read_json(point / "pipeline_summary.json")
               if summary is None else summary)
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
                               label: str,
                               config_path: Path | None = None,
                               ) -> tuple[dict, dict[str, float]] | None:
    """Bind an attached local summary/performance to its validated gem5 result."""
    decision = attach_result.validate_local_attachment(
        point, r1_dir, config_path
    )
    if decision.get("accepted") is not True:
        return None
    result = decision.get("result")
    if not isinstance(result, dict):
        return None
    summary = decision.get("summary")
    if not isinstance(summary, dict):
        return None
    try:
        metrics = _summary_metrics(point, result, label, summary)
    except (OSError, ValueError):
        return None
    return result, metrics


def _validate_fixed(paths: dict[str, Path],
                    config_path: Path | None = None,
                    ) -> tuple[dict, dict[str, float]] | None:
    return _validate_local_attachment(
        paths["r1"], paths["fixed"], "fixed-bin", config_path
    )


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
        and _validate_fixed(paths, config_path) is not None
    )


def _validate_completed(record: dict | None, key: ArchitectureKey,
                        paths: dict[str, Path], config_path: Path) -> bool:
    if (not isinstance(record, dict) or record.get("state") != "success"
            or not _identity(record, key, paths, config_path)):
        return False
    fixed = _validate_fixed(paths, config_path)
    if fixed is None:
        return False
    fixed_result, fixed_metrics = fixed
    try:
        fixed_vector, clip_vector = _validated_vectors(paths["fixed"], paths["clip"])
    except (KeyError, OSError, TypeError, ValueError):
        return False
    reused = run_r2.strict_latency_vectors_equal(fixed_vector, clip_vector)
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
        try:
            clip_metrics = _summary_metrics(
                paths["clip"], clip_result, "CLIP-3D", clip_summary
            )
        except (OSError, ValueError):
            return False
    else:
        clip_validated = _validate_local_attachment(
            paths["r1"], paths["clip"], "CLIP-3D", config_path
        )
        if clip_validated is None:
            return False
        clip_result, clip_metrics = clip_validated
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
    point = Path(point).resolve()
    with exclusive_point_lock(point):
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
    """Join the physical sweep domain, then serialize one physical pair."""
    normalized_key = ArchitectureKey(key.workload, key.l1d_size, key.l2_size)
    resolved_fixed_root = Path(fixed_root).resolve()
    resolved_clip_root = Path(clip_root).resolve()
    resolved_status_root = (
        resolved_fixed_root.parent / "paired_r2_status"
        if status_root is None else Path(status_root).resolve()
    )
    _validate_direct_native_pair(
        normalized_key, r1_root, resolved_fixed_root, resolved_clip_root,
        config_path, rerun,
    )
    sweep_locks = _sweep_lock_paths(resolved_fixed_root, resolved_clip_root)
    with _execution_locks(
            sweep_locks, blocking=True, shared=True,
            conflict_message="physical R2 sweep lock is unavailable"):
        _validate_direct_native_pair(
            normalized_key, r1_root, resolved_fixed_root, resolved_clip_root,
            config_path, rerun,
        )
        return _run_pair_under_sweep(
            normalized_key, r1_root, resolved_fixed_root, resolved_clip_root,
            config_path,
            rerun, status_root=resolved_status_root,
        )


def _run_pair_under_sweep(key: ArchitectureKey, r1_root: Path,
                          fixed_root: Path, clip_root: Path, config_path: Path,
                          rerun: bool = False, *,
                          status_root: Path | None = None) -> dict:
    """Serialize a pair while the caller participates in the sweep lock level."""
    normalized_key = ArchitectureKey(key.workload, key.l1d_size, key.l2_size)
    resolved_fixed_root = Path(fixed_root).resolve()
    resolved_clip_root = Path(clip_root).resolve()
    resolved_status_root = (
        resolved_fixed_root.parent / "paired_r2_status"
        if status_root is None else Path(status_root).resolve()
    )
    pair_locks = _pair_lock_paths(
        normalized_key, resolved_fixed_root, resolved_clip_root
    )
    with _execution_locks(
            pair_locks, blocking=True,
            conflict_message="physical R2 pair lock is unavailable"):
        return _run_pair_locked(
            normalized_key, r1_root, resolved_fixed_root, resolved_clip_root,
            config_path, rerun, status_root=resolved_status_root,
        )


def _run_pair_locked(key: ArchitectureKey, r1_root: Path, fixed_root: Path,
                     clip_root: Path, config_path: Path, rerun: bool = False,
                     *, status_root: Path | None = None) -> dict:
    """Run one pair while its execution lock is already held."""
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
            fixed_validated = _validate_fixed(paths, config_path)
            if fixed_validated is None:  # Defensive against replacement after check.
                raise ValueError("fixed R2 changed after resume validation")
            fixed_result, fixed_metrics = fixed_validated
        else:
            fixed_result = run_r2.run(
                paths["r1"], paths["fixed"] / "r2_latency.json",
                paths["fixed"] / "gem5_r2", rerun=rerun,
            )
            attach_result.attach(paths["fixed"])
            fixed_validated = _validate_fixed(paths, config_path)
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
        reused = run_r2.strict_latency_vectors_equal(fixed_vector, clip_vector)
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
            clip_metrics = _summary_metrics(
                paths["clip"], clip_result, "CLIP-3D", clip_summary
            )
            clip_artifact = paths["clip"] / "r2_reuse.json"
        else:
            _clear_reuse_attachment(paths["clip"])
            run_r2.run(
                paths["r1"], paths["clip"] / "r2_latency.json",
                paths["clip"] / "gem5_r2", rerun=rerun,
            )
            attach_result.attach(paths["clip"])
            clip_validated = _validate_local_attachment(
                paths["r1"], paths["clip"], "CLIP-3D", config_path
            )
            if clip_validated is None:
                raise ValueError(
                    "CLIP-3D local attachment failed result/performance/summary "
                    "validation"
                )
            clip_result, clip_metrics = clip_validated
            clip_artifact = paths["clip"] / "gem5_r2/r2_result.json"

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
        equal_overrides = run_r2.strict_latency_vectors_equal(
            fixed_vector, clip_vector
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
            attachment_valid = _validate_local_attachment(
                r1_dir, point_dir, "fixed-bin", config_path
            ) is not None
            return {**decision, "branch": "fixed",
                    "repair_required": (decision.get("accepted") is True
                                        and not attachment_valid),
                    "rerun_requested": False,
                    "scientific_evidence_accepted": attachment_valid}
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
                r1_dir, point_dir, "CLIP-3D", config_path
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


def _revalidate_persisted_pairs(
        keys: list[ArchitectureKey], r1_root: Path, fixed_root: Path,
        clip_root: Path, status_root: Path, config_path: Path,
        ) -> tuple[dict[ArchitectureKey, dict], dict]:
    """Re-read and live-validate every selected persisted pair claim."""
    certified: dict[ArchitectureKey, dict] = {}
    rejections = []
    for key in keys:
        paths = _paths(key, r1_root, fixed_root, clip_root, status_root)
        pair_locks = _pair_lock_paths(key, fixed_root, clip_root)
        with _execution_locks(
                pair_locks, blocking=True,
                conflict_message="physical R2 pair lock is unavailable"):
            persisted = _read_object(paths["status"])
            accepted = _validate_completed(persisted, key, paths, config_path)
            if accepted:
                record = deepcopy(persisted)
                record["completion_revalidated"] = True
                certified[key] = record
                continue

            if persisted is None:
                reason = "persisted pair status is missing or malformed"
                record = _pending_pair(key, status_root)
            elif persisted.get("state") == "success":
                reason = "persisted success record failed live evidence validation"
                record = deepcopy(persisted)
            else:
                reason = f"persisted pair state is {persisted.get('state')!r}"
                record = deepcopy(persisted)
            record.update({
                "state": "failed",
                "completion_revalidated": False,
                "revalidation_error": reason,
            })
            record.setdefault("key", _key_dict(key))
            record.setdefault("pair_status", str(paths["status"]))
            certified[key] = record
            rejections.append({
                "key": _key_dict(key),
                "pair_status": str(paths["status"]),
                "reason": reason,
            })
    return certified, {
        "performed": True,
        "accepted_pair_count": len(keys) - len(rejections),
        "rejected_pair_count": len(rejections),
        "rejections": rejections,
    }


def _experiment_status(keys: list[ArchitectureKey], results: dict[ArchitectureKey, dict],
                       status_root: Path, preflight: dict, limited_run: bool,
                       started: float, *, completion_certified: bool = False,
                       completion_revalidation: dict | None = None) -> dict:
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
        completion_certified
        and not limited_run
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
        "completion_revalidation": completion_revalidation or {
            "performed": False,
            "accepted_pair_count": 0,
            "rejected_pair_count": 0,
            "rejections": [],
        },
        "preflight": preflight,
        "pairs": pairs,
    }


def run_sweep(r1_root: Path, fixed_root: Path, clip_root: Path,
              selection_path: Path, config_path: Path, status_root: Path,
              jobs: int = 1, rerun: bool = False,
              limit: int | None = None) -> dict:
    """Fail fast on another sweep, then hold the sweep lock through final status."""
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    fixed_root = Path(fixed_root).resolve()
    clip_root = Path(clip_root).resolve()
    status_root = Path(status_root).resolve()
    sweep_locks = _sweep_lock_paths(fixed_root, clip_root)
    with _execution_locks(
            sweep_locks, blocking=False,
            conflict_message=(
                "paired R2 sweep is already active on a physical root"
            )):
        return _run_sweep_locked(
            r1_root, fixed_root, clip_root, selection_path, config_path,
            status_root, jobs=jobs, rerun=rerun, limit=limit,
        )


def _run_sweep_locked(r1_root: Path, fixed_root: Path, clip_root: Path,
                      selection_path: Path, config_path: Path, status_root: Path,
                      jobs: int = 1, rerun: bool = False,
                      limit: int | None = None) -> dict:
    """Validate and execute a sweep while its outer execution lock is held."""
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
    preflight = _require_native_preflight(validate_layout_roots(
        fixed_root, clip_root, all_keys, config_path,
        selection=selection, require_layout_only=False,
        existing_r2_validator=_existing_r2_validator(
            r1_root, fixed_root, clip_root, config_path,
            rerun_requested=rerun,
        ),
    ))
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
                _run_pair_under_sweep, key, r1_root, fixed_root, clip_root,
                config_path, rerun,
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

    persisted_results, revalidation = _revalidate_persisted_pairs(
        keys, r1_root, fixed_root, clip_root, status_root, config_path
    )
    final = _experiment_status(
        keys, persisted_results, status_root, preflight, limited_run, started,
        completion_certified=True, completion_revalidation=revalidation,
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
