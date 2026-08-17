#!/usr/bin/env python3
"""Validate and attach an exact fixed-bin R2 result to a CLIP-3D point."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile

from workflow.common import atomic_write_bytes, write_json
from workflow.r2.attachment_validation import (
    exclusive_point_lock,
    native_cache_identity,
    require_corrected_native_cache,
    validate_physical_coherence,
    validate_vector_native_cache,
)
from workflow.r2.run_r2 import (
    strict_latency_vectors_equal,
    validate_latency_vector,
    validate_local_result,
)
from workflow.thermal.sustainable_frequency import evaluate


def _capture_file(path: Path, reasons: list[str], label: str,
                  snapshots: dict[Path, dict]) -> dict | None:
    """Capture one immutable byte snapshot used for both parsing and hashing."""
    path = Path(path).resolve()
    if path in snapshots:
        return snapshots[path]
    try:
        data = path.read_bytes()
    except OSError as error:
        reasons.append(f"cannot read {label} {path}: {error}")
        return None
    snapshot = {
        "bytes": data,
        "sha256": hashlib.sha256(data).hexdigest(),
        "label": label,
    }
    snapshots[path] = snapshot
    return snapshot


def _read_object(path: Path, reasons: list[str], label: str,
                 snapshots: dict[Path, dict] | None = None) -> dict | None:
    snapshots = {} if snapshots is None else snapshots
    snapshot = _capture_file(path, reasons, label, snapshots)
    if snapshot is None:
        return None
    try:
        value = json.loads(snapshot["bytes"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        reasons.append(f"cannot read {label} {Path(path).resolve()}: {error}")
        return None
    if not isinstance(value, dict):
        reasons.append(f"{label} must contain an object: {path}")
        return None
    return value


def _snapshot_sha256(snapshots: dict[Path, dict], path: Path) -> str:
    return snapshots[Path(path).resolve()]["sha256"]


def _record_changed_snapshots(snapshots: dict[Path, dict],
                              reasons: list[str],
                              ignored: set[Path] | None = None) -> None:
    """Reject any authoritative input replaced after its captured read."""
    ignored = {Path(path).resolve() for path in (ignored or set())}
    for path, snapshot in snapshots.items():
        if path in ignored:
            continue
        try:
            current = path.read_bytes()
        except OSError as error:
            reasons.append(
                f"{snapshot['label']} changed during validation: {path}: {error}"
            )
            continue
        if hashlib.sha256(current).hexdigest() != snapshot["sha256"]:
            reasons.append(
                f"{snapshot['label']} changed during validation: {path}"
            )


def _replace_bytes(path: Path, data: bytes) -> None:
    """Atomically restore exact authoritative bytes without JSON reformatting."""
    atomic_write_bytes(path, data)


def _current_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _restore_if_still_published(path: Path, published_sha256: str,
                                previous_bytes: bytes) -> None:
    """Restore bytes still owned by this attachment under the point lock."""
    if _current_sha256(path) == published_sha256:
        _replace_bytes(path, previous_bytes)


def _accepted_marker_bytes(path: Path, performance_path: Path,
                           performance_sha256: str, summary_path: Path,
                           summary_sha256: str,
                           summary: dict, expected_artifact: dict) -> bytes | None:
    """Return a prior marker only when it binds the current complete outputs."""
    try:
        data = path.read_bytes()
        marker = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(marker, dict):
        return None
    outputs = marker.get("target_outputs", {})
    performance_output = outputs.get("performance", {})
    summary_output = outputs.get("summary", {})
    if (marker.get("decision") != "accepted"
            or any(marker.get(key) != value
                   for key, value in expected_artifact.items())
            or performance_output.get("path") != str(performance_path)
            or performance_output.get("sha256") != performance_sha256
            or summary_output.get("path") != str(summary_path)
            or summary_output.get("sha256") != summary_sha256
            or summary.get("r2_reused") is not True
            or summary.get("r2_reuse_artifact") != str(path)):
        return None
    return data


def _architecture_key(value: dict) -> dict:
    return {
        "workload": value.get("workload"),
        "l1d_size": value.get("l1d_size"),
        "l2_size": value.get("l2_size"),
    }


def _path_matches(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).resolve() == expected.resolve()


def _validate_reuse(fixed_point: Path, clip_point: Path, r1_dir: Path,
                    config_path: Path) -> tuple[dict, dict[Path, dict], dict]:
    """Validate reuse and retain the exact input snapshots used by attachment."""
    fixed_point = Path(fixed_point).resolve()
    clip_point = Path(clip_point).resolve()
    r1_dir = Path(r1_dir).resolve()
    config_path = Path(config_path).resolve()
    source_latency_path = fixed_point / "r2_latency.json"
    target_latency_path = clip_point / "r2_latency.json"
    source_result_path = fixed_point / "gem5_r2/r2_result.json"
    source_status_path = fixed_point / "gem5_r2/status.json"
    reasons: list[str] = []
    snapshots: dict[Path, dict] = {}

    r1_metadata_path = r1_dir / "r1_metadata.json"
    r1_stats_path = r1_dir / "stats.txt"
    source_stats_path = fixed_point / "gem5_r2/stats.txt"
    metadata = _read_object(
        r1_metadata_path, reasons, "R1 metadata", snapshots
    )
    _capture_file(r1_stats_path, reasons, "R1 stats", snapshots)
    config = _read_object(config_path, reasons, "experiment config", snapshots)
    source_vector = _read_object(
        source_latency_path, reasons, "source latency vector", snapshots
    )
    target_vector = _read_object(
        target_latency_path, reasons, "target latency vector", snapshots
    )
    fixed_run_config = _read_object(
        fixed_point / "run_config.json", reasons, "fixed run config", snapshots
    )
    clip_run_config = _read_object(
        clip_point / "run_config.json", reasons, "CLIP run config", snapshots
    )
    fixed_summary = _read_object(
        fixed_point / "pipeline_summary.json", reasons, "fixed pipeline summary",
        snapshots,
    )
    clip_summary_path = clip_point / "pipeline_summary.json"
    clip_summary = _read_object(
        clip_summary_path, reasons, "CLIP pipeline summary", snapshots
    )
    target_modules_path = clip_point / "modules.json"
    source_modules_path = fixed_point / "modules.json"
    target_thermal_path = clip_point / "hotspot/thermal_result.json"
    target_manifest_path = clip_point / "hotspot/hotspot_manifest.json"
    target_steady_path = clip_point / "hotspot/steady.txt"
    target_performance_path = clip_point / "performance.json"
    target_modules = _read_object(
        target_modules_path, reasons, "target modules", snapshots
    )
    source_modules = _read_object(
        source_modules_path, reasons, "source modules", snapshots
    )
    target_thermal = _read_object(
        target_thermal_path, reasons, "target thermal result", snapshots
    )
    target_manifest = _read_object(
        target_manifest_path, reasons, "target HotSpot manifest", snapshots
    )
    target_steady = _capture_file(
        target_steady_path, reasons, "target HotSpot steady output", snapshots
    )
    target_performance = _read_object(
        target_performance_path, reasons, "target performance", snapshots
    )
    captured_source_result = _read_object(
        source_result_path, reasons, "source R2 result", snapshots
    )
    captured_source_status = _read_object(
        source_status_path, reasons, "source R2 status", snapshots
    )
    _capture_file(source_stats_path, reasons, "source R2 stats", snapshots)

    local = validate_local_result(r1_dir, source_latency_path, fixed_point / "gem5_r2")
    if local["accepted"] is not True:
        reasons.extend(f"source {reason}" for reason in local["reasons"])
    if (captured_source_result is not None
            and local.get("result") != captured_source_result):
        reasons.append("source R2 result changed during validation")
    if (captured_source_status is not None
            and local.get("status") != captured_source_status):
        reasons.append("source R2 status changed during validation")
    source_result = captured_source_result
    source_status = captured_source_status

    source_native = None
    target_native = None
    if source_modules is not None:
        try:
            source_native = require_corrected_native_cache(
                fixed_point, source_modules,
            )
        except (OSError, TypeError, ValueError) as error:
            reasons.append(f"source McPAT-native cache authority is invalid: {error}")
    if target_modules is not None:
        try:
            target_native = native_cache_identity(target_modules)
        except (TypeError, ValueError) as error:
            reasons.append(f"target McPAT-native cache identity is invalid: {error}")
    if source_native is not None and target_native is not None \
            and source_native != target_native:
        reasons.append("source and target McPAT-native cache identities differ")

    if source_vector is not None and target_vector is not None:
        for label, vector in (("source", source_vector), ("target", target_vector)):
            try:
                validate_latency_vector(vector)
            except ValueError as error:
                reasons.append(f"{label} {error}")
        for label, vector, identity in (
                ("source", source_vector, source_native),
                ("target", target_vector, target_native)):
            if identity is None:
                continue
            try:
                validate_vector_native_cache(vector, identity)
            except ValueError as error:
                reasons.append(f"{label} {error}")

    canonical_key = _architecture_key(metadata or {})
    if any(not isinstance(value, str) or not value for value in canonical_key.values()):
        reasons.append("R1 metadata does not define a canonical architecture key")
    for label, summary in (("fixed", fixed_summary), ("CLIP", clip_summary)):
        if summary is None:
            continue
        if _architecture_key(summary) != canonical_key:
            reasons.append(f"{label} canonical architecture key differs from R1")
        if not _path_matches(summary.get("r1"), r1_dir):
            reasons.append(f"{label} R1 directory differs from the canonical R1 directory")
    if fixed_summary is not None and fixed_summary.get("layout_method") != "fixed-bin":
        reasons.append("fixed pipeline summary layout method is not fixed-bin")
    if clip_summary is not None and clip_summary.get("layout_method") != "clip3d":
        reasons.append("CLIP pipeline summary layout method is not clip3d")

    if metadata is not None and source_result is not None:
        for field in (
                "instruction_window_scope", "warmup_insts_cpu0", "measure_insts_cpu0"):
            expected = metadata.get(
                field, "cpu0" if field == "instruction_window_scope" else None
            )
            if source_result.get(field) != expected:
                reasons.append(f"source result {field} differs from canonical R1")

    for label, run_config, method in (
            ("fixed", fixed_run_config, "fixed-bin"),
            ("CLIP", clip_run_config, "clip3d")):
        if run_config is None or config is None:
            continue
        if run_config.get("config") != config:
            reasons.append(f"{label} embedded config differs from experiment config")
        if not _path_matches(run_config.get("source"), config_path):
            reasons.append(f"{label} config source path differs from experiment config")
        if run_config.get("layout_method") != method:
            reasons.append(f"{label} run config layout method is not {method}")

    if all(value is not None for value in (
            metadata, config, target_modules, target_thermal, target_manifest,
            target_steady, target_performance, clip_summary)):
        validate_physical_coherence(
            metadata, config, target_modules, target_thermal,
            target_performance, clip_summary, r1_dir, clip_point, reasons,
            r1_stats_bytes=snapshots[r1_stats_path]["bytes"],
            thermal_steady_bytes=target_steady["bytes"],
            thermal_manifest=target_manifest,
        )

    if source_status is not None and source_status.get("state") != "success":
        reasons.append("source status is not success")
    if source_result is not None:
        ipc2 = source_result.get("ipc2")
        if (not isinstance(ipc2, (int, float)) or isinstance(ipc2, bool)
                or not math.isfinite(float(ipc2)) or float(ipc2) <= 0):
            reasons.append("source result IPC2 must be finite and positive")
        if not _path_matches(source_result.get("latency_vector"), source_latency_path):
            reasons.append("source result latency_vector does not identify source vector")
    if fixed_summary is not None and source_result is not None:
        if fixed_summary.get("r2_source") != str(source_result_path):
            reasons.append("fixed source result provenance does not identify r2_result.json")
        if fixed_summary.get("ipc2") != source_result.get("ipc2"):
            reasons.append("fixed summary IPC2 differs from source result IPC2")

    # Exact Task-5 vector equality remains the final scientific reuse gate,
    # after both branches have established the same native physical authority.
    if source_vector is not None and target_vector is not None:
        try:
            vectors_equal = strict_latency_vectors_equal(source_vector, target_vector)
        except ValueError:
            vectors_equal = False
        if not vectors_equal:
            reasons.append("source and target gem5_overrides differ or are invalid")

    _record_changed_snapshots(snapshots, reasons)
    context = {
        "clip_run_config": clip_run_config,
        "clip_summary": clip_summary,
        "clip_summary_path": clip_summary_path,
        "target_modules_path": target_modules_path,
        "target_thermal_path": target_thermal_path,
        "target_performance_path": target_performance_path,
    }
    if reasons:
        return ({
            "accepted": False,
            "decision": "rejected",
            "reasons": reasons,
            "fixed_point": str(fixed_point),
            "clip_point": str(clip_point),
        }, snapshots, context)

    artifact = {
        "schema_version": 1,
        "decision": "accepted",
        "source_point": str(fixed_point),
        "target_point": str(clip_point),
        "source_latency": {
            "path": str(source_latency_path),
            "sha256": _snapshot_sha256(snapshots, source_latency_path),
        },
        "target_latency": {
            "path": str(target_latency_path),
            "sha256": _snapshot_sha256(snapshots, target_latency_path),
        },
        "gem5_overrides": source_vector["gem5_overrides"],
        "gem5_args": source_vector["gem5_args"],
        "source_result": {
            "path": str(source_result_path),
            "sha256": _snapshot_sha256(snapshots, source_result_path),
        },
        "source_status": {
            "path": str(source_status_path),
            "sha256": _snapshot_sha256(snapshots, source_status_path),
        },
        "source_stats": {
            "path": source_result["stats"],
            "sha256": _snapshot_sha256(snapshots, source_stats_path),
        },
        "source_ipc2": source_result["ipc2"],
        "canonical_architecture_key": canonical_key,
        "r1": {
            "directory": str(r1_dir),
            "metadata_path": str(r1_metadata_path.resolve()),
            "metadata_sha256": _snapshot_sha256(snapshots, r1_metadata_path),
            "stats_path": str(r1_stats_path.resolve()),
            "stats_sha256": _snapshot_sha256(snapshots, r1_stats_path),
            "instruction_window_scope": metadata.get(
                "instruction_window_scope", "cpu0"
            ),
            "warmup_insts_cpu0": metadata["warmup_insts_cpu0"],
            "measure_insts_cpu0": metadata["measure_insts_cpu0"],
        },
        "config": {
            "path": str(config_path),
            "sha256": _snapshot_sha256(snapshots, config_path),
            "embedded": config,
        },
        "target_inputs": {
            "modules": {
                "path": str(target_modules_path),
                "sha256": _snapshot_sha256(snapshots, target_modules_path),
            },
            "thermal": {
                "path": str(target_thermal_path),
                "sha256": _snapshot_sha256(snapshots, target_thermal_path),
            },
            "hotspot_manifest": {
                "path": str(target_manifest_path),
                "sha256": _snapshot_sha256(snapshots, target_manifest_path),
            },
            "steady": {
                "path": str(target_steady_path),
                "sha256": _snapshot_sha256(snapshots, target_steady_path),
            },
        },
        "native_cache": {
            "source": source_native,
            "target": target_native,
            "source_modules": {
                "path": str(source_modules_path),
                "sha256": _snapshot_sha256(snapshots, source_modules_path),
            },
        },
    }
    return ({
        "accepted": True,
        "decision": "accepted",
        "reasons": [],
        "artifact": artifact,
    }, snapshots, context)


def validate_reuse(fixed_point: Path, clip_point: Path, r1_dir: Path,
                   config_path: Path) -> dict:
    """Return a complete decision for fixed-bin to CLIP-3D R2 reuse."""
    decision, _snapshots, _context = _validate_reuse(
        fixed_point, clip_point, r1_dir, config_path
    )
    return decision


def attach_reused_result(fixed_point: Path, clip_point: Path, r1_dir: Path,
                         config_path: Path) -> dict:
    """Attach one result while excluding concurrent transactions on the point."""
    clip_point = Path(clip_point).resolve()
    with exclusive_point_lock(clip_point):
        return _attach_reused_result_locked(
            fixed_point, clip_point, r1_dir, config_path
        )


def _attach_reused_result_locked(fixed_point: Path, clip_point: Path, r1_dir: Path,
                                 config_path: Path) -> dict:
    """Attach accepted fixed-bin IPC2 and recompute target CLIP performance."""
    fixed_point = Path(fixed_point).resolve()
    clip_point = Path(clip_point).resolve()
    artifact_path = clip_point / "r2_reuse.json"
    decision, snapshots, context = _validate_reuse(
        fixed_point, clip_point, r1_dir, config_path
    )
    if decision["accepted"] is not True:
        artifact_path.unlink(missing_ok=True)
        raise ValueError("R2 reuse rejected: " + "; ".join(decision["reasons"]))

    artifact = decision["artifact"]
    source_result_path = Path(artifact["source_result"]["path"])
    ipc2 = artifact["source_ipc2"]
    run_config = context["clip_run_config"]
    frequency = run_config["config"]["frequency"]
    modules_path = context["target_modules_path"]
    thermal_path = context["target_thermal_path"]
    performance_path = context["target_performance_path"]
    summary_path = context["clip_summary_path"]
    performance_before = snapshots[performance_path]["bytes"]
    summary_before = snapshots[summary_path]["bytes"]
    previous_marker = _accepted_marker_bytes(
        artifact_path,
        performance_path,
        snapshots[performance_path]["sha256"],
        summary_path,
        snapshots[summary_path]["sha256"],
        context["clip_summary"],
        artifact,
    )
    attempted_outputs: dict[Path, tuple[str, bytes]] = {}

    try:
        with tempfile.TemporaryDirectory(prefix=".r2-reuse-", dir=clip_point) as staging:
            staging = Path(staging)
            staged_modules = staging / "modules.json"
            staged_thermal = staging / "thermal_result.json"
            staged_performance = staging / "performance.json"
            staged_summary = staging / "pipeline_summary.json"
            staged_modules.write_bytes(snapshots[modules_path]["bytes"])
            staged_thermal.write_bytes(snapshots[thermal_path]["bytes"])
            performance = evaluate(
                staged_modules,
                staged_thermal,
                staged_performance,
                frequency["f0_ghz"],
                frequency["fmin_ghz"],
                frequency["tsafe_c"],
                frequency["ambient_c"],
                ipc2,
            )
            performance_sha256 = hashlib.sha256(
                staged_performance.read_bytes()
            ).hexdigest()

            summary = deepcopy(context["clip_summary"])
            summary.update({
                "gamma": performance["gamma"],
                "tmax_c": performance["tmax_f0_c"],
                "sustainable_frequency_ghz": performance[
                    "sustainable_frequency_ghz"
                ],
                "ipc1": performance["ipc1"],
                "bips1_thermal": performance["bips1_thermal"],
                "ipc2": ipc2,
                "bips2": performance["bips2"],
                "r2_source": str(source_result_path),
                "r2_reused": True,
                "r2_reuse_artifact": str(artifact_path),
            })
            stage_seconds = summary.setdefault("stage_seconds", {})
            stage_seconds.pop("gem5_r2", None)
            summary["total_pipeline_seconds"] = sum(
                float(value) for value in stage_seconds.values() if value is not None
            )
            artifacts = summary.setdefault("artifacts", {})
            artifacts["r2_result"] = str(source_result_path)
            artifacts["r2_reuse"] = str(artifact_path)
            write_json(staged_summary, summary)
            summary_sha256 = hashlib.sha256(staged_summary.read_bytes()).hexdigest()

            changed: list[str] = []
            _record_changed_snapshots(snapshots, changed)
            if changed:
                raise ValueError("R2 reuse inputs " + "; ".join(changed))

            artifact["target_outputs"] = {
                "performance": {
                    "path": str(performance_path),
                    "sha256": performance_sha256,
                },
                "summary": {
                    "path": str(summary_path),
                    "sha256": summary_sha256,
                },
                "gamma": performance["gamma"],
                "sustainable_frequency_ghz": performance[
                    "sustainable_frequency_ghz"
                ],
                "ipc1": performance["ipc1"],
                "bips1_thermal": performance["bips1_thermal"],
                "bips2": performance["bips2"],
            }
            artifact_path.unlink(missing_ok=True)
            attempted_outputs[performance_path] = (
                performance_sha256, performance_before
            )
            write_json(performance_path, performance)
            if _current_sha256(performance_path) != performance_sha256:
                raise OSError("published performance differs from staged output")
            attempted_outputs[summary_path] = (summary_sha256, summary_before)
            write_json(summary_path, summary)
            changed = []
            _record_changed_snapshots(
                snapshots, changed, {performance_path, summary_path}
            )
            if changed:
                raise ValueError("R2 reuse inputs " + "; ".join(changed))
            if (_current_sha256(performance_path) != performance_sha256
                    or _current_sha256(summary_path) != summary_sha256):
                raise OSError("published outputs changed before acceptance marker")
            write_json(artifact_path, artifact)
            return summary
    except Exception:
        for path, (published_sha256, previous_bytes) in attempted_outputs.items():
            _restore_if_still_published(path, published_sha256, previous_bytes)
        changed = []
        _record_changed_snapshots(
            snapshots, changed, {performance_path, summary_path}
        )
        outputs_restored = (
            _current_sha256(performance_path) == snapshots[performance_path]["sha256"]
            and _current_sha256(summary_path) == snapshots[summary_path]["sha256"]
        )
        if previous_marker is not None and not changed and outputs_restored:
            _replace_bytes(artifact_path, previous_marker)
        else:
            artifact_path.unlink(missing_ok=True)
        raise
