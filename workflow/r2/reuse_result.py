#!/usr/bin/env python3
"""Validate and attach an exact fixed-bin R2 result to a CLIP-3D point."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile

from workflow.common import write_json
from workflow.r2.run_r2 import canonical_gem5_args, validate_local_result
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
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".rollback.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


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


def _finite_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _validate_target_inputs(metadata: dict, config: dict, modules: dict,
                            thermal: dict, summary: dict, r1_dir: Path,
                            clip_point: Path,
                            reasons: list[str]) -> None:
    architecture = modules.get("architecture")
    if not isinstance(architecture, dict):
        reasons.append("target modules architecture is missing")
    else:
        for field in (
                "workload", "l1i_size", "l1d_size", "l2_size", "num_cores",
                "instruction_window_scope", "warmup_insts_cpu0",
                "measure_insts_cpu0"):
            expected = metadata.get(
                field, "cpu0" if field == "instruction_window_scope" else None
            )
            if architecture.get(field) != expected:
                reasons.append(
                    f"target modules architecture {field} differs from canonical R1"
                )
    if not _path_matches(modules.get("source_r1"), r1_dir):
        reasons.append("target modules source_r1 differs from canonical R1")
    totals = modules.get("totals")
    if not isinstance(totals, dict):
        reasons.append("target modules totals are missing")
    else:
        dynamic = totals.get("dynamic_power_w")
        leakage = totals.get("leakage_power_w")
        total = totals.get("total_power_w")
        if (not all(_finite_number(value) for value in (dynamic, leakage, total))
                or float(dynamic) < 0 or float(leakage) < 0 or float(total) <= 0
                or not math.isclose(
                    float(dynamic) + float(leakage), float(total),
                    rel_tol=1e-12, abs_tol=1e-12,
                )):
            reasons.append("target modules power totals are inconsistent")
        gamma = modules.get("gamma")
        if (not _finite_number(gamma) or not _finite_number(leakage)
                or not _finite_number(total) or float(total) <= 0
                or not math.isclose(
                    float(gamma), float(leakage) / float(total),
                    rel_tol=1e-12, abs_tol=1e-12,
                )):
            reasons.append("target modules gamma differs from leakage/total power")
    if not _finite_number(modules.get("ipc1")) or float(modules["ipc1"]) <= 0:
        reasons.append("target modules IPC1 must be finite and positive")

    frequency = config.get("frequency")
    physical = config.get("physical")
    if not isinstance(frequency, dict) or not isinstance(physical, dict):
        reasons.append("experiment config lacks frequency/physical inputs")
        return
    tmax_c = thermal.get("tmax_c")
    if not _finite_number(tmax_c):
        reasons.append("target thermal tmax_c must be finite")
    for field, expected in (
            ("power_trace", clip_point / "hotspot/power.ptrace"),
            ("steady_file", clip_point / "hotspot/steady.txt"),
            ("grid_steady_file", clip_point / "hotspot/grid.steady.txt")):
        if not _path_matches(thermal.get(field), expected):
            reasons.append(
                f"target thermal {field} does not identify the target HotSpot output"
            )
    if thermal.get("return_code") != 0:
        reasons.append("target thermal return_code is not zero")
    if (not _finite_number(thermal.get("ambient_c"))
            or thermal.get("ambient_c") != frequency.get("ambient_c")):
        reasons.append("target thermal ambient differs from experiment config")
    if (not _finite_number(thermal.get("r_convec_k_per_w"))
            or thermal.get("r_convec_k_per_w") != physical.get("r_convec_k_per_w")):
        reasons.append("target thermal cooling differs from experiment config")
    if not _finite_number(thermal.get("tmax_k")):
        reasons.append("target thermal tmax_k must be finite")
    elif _finite_number(tmax_c) and not math.isclose(
            float(thermal["tmax_k"]) - 273.15, float(tmax_c),
            rel_tol=0.0, abs_tol=1e-9):
        reasons.append("target thermal Kelvin/Celsius values disagree")
    summary_tmax = summary.get("tmax_c")
    if (not _finite_number(summary_tmax) or not _finite_number(tmax_c)
            or not math.isclose(
                float(summary_tmax), float(tmax_c), rel_tol=0.0, abs_tol=1e-9
            )):
        reasons.append("target summary thermal tmax_c differs from thermal result")


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
    target_thermal_path = clip_point / "hotspot/thermal_result.json"
    target_performance_path = clip_point / "performance.json"
    target_modules = _read_object(
        target_modules_path, reasons, "target modules", snapshots
    )
    target_thermal = _read_object(
        target_thermal_path, reasons, "target thermal result", snapshots
    )
    _read_object(
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

    if source_vector is not None and target_vector is not None:
        source_overrides = source_vector.get("gem5_overrides")
        target_overrides = target_vector.get("gem5_overrides")
        if source_overrides != target_overrides:
            reasons.append("source and target gem5_overrides differ")
        for label, vector in (("source", source_vector), ("target", target_vector)):
            try:
                expected_args = canonical_gem5_args(vector.get("gem5_overrides"))
            except ValueError as error:
                reasons.append(f"{label} {error}")
            else:
                if vector.get("gem5_args") != expected_args:
                    reasons.append(
                        f"{label} gem5_args are not the complete canonical rendering "
                        "of gem5_overrides"
                    )

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
            metadata, config, target_modules, target_thermal, clip_summary)):
        _validate_target_inputs(
            metadata, config, target_modules, target_thermal, clip_summary,
            r1_dir, clip_point, reasons
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
    publication_started = False

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
                "sustainable_frequency_ghz": performance[
                    "sustainable_frequency_ghz"
                ],
                "bips2": performance["bips2"],
            }
            publication_started = True
            artifact_path.unlink(missing_ok=True)
            write_json(performance_path, performance)
            if hashlib.sha256(performance_path.read_bytes()).hexdigest() \
                    != performance_sha256:
                raise OSError("published performance differs from staged output")
            write_json(summary_path, summary)
            changed = []
            _record_changed_snapshots(
                snapshots, changed, {performance_path, summary_path}
            )
            if changed:
                raise ValueError("R2 reuse inputs " + "; ".join(changed))
            if (hashlib.sha256(performance_path.read_bytes()).hexdigest()
                    != performance_sha256
                    or hashlib.sha256(summary_path.read_bytes()).hexdigest()
                    != summary_sha256):
                raise OSError("published outputs changed before acceptance marker")
            write_json(artifact_path, artifact)
            return summary
    except Exception:
        if publication_started:
            _replace_bytes(performance_path, performance_before)
            _replace_bytes(summary_path, summary_before)
        changed = []
        _record_changed_snapshots(
            snapshots, changed, {performance_path, summary_path}
        )
        outputs_restored = (
            hashlib.sha256(performance_path.read_bytes()).hexdigest()
            == snapshots[performance_path]["sha256"]
            and hashlib.sha256(summary_path.read_bytes()).hexdigest()
            == snapshots[summary_path]["sha256"]
        )
        if previous_marker is not None and not changed and outputs_restored:
            _replace_bytes(artifact_path, previous_marker)
        else:
            artifact_path.unlink(missing_ok=True)
        raise
