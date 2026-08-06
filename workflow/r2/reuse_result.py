#!/usr/bin/env python3
"""Validate and attach an exact fixed-bin R2 result to a CLIP-3D point."""

from __future__ import annotations

import math
from pathlib import Path

from workflow.common import read_json, sha256_file, write_json
from workflow.r2.run_r2 import validate_local_result
from workflow.thermal.sustainable_frequency import evaluate


GEM5_OVERRIDE_KEYS = (
    "l1i_tag_latency",
    "l1i_data_latency",
    "l1i_response_latency",
    "l1d_tag_latency",
    "l1d_data_latency",
    "l1d_response_latency",
    "l2_tag_latency",
    "l2_data_latency",
    "l2_response_latency",
    "xbar_frontend_latency",
    "xbar_forward_latency",
    "xbar_response_latency",
    "xbar_snoop_response_latency",
)


def canonical_gem5_args(overrides: dict) -> list[str]:
    """Render one complete R2 override mapping in the generator's fixed order."""
    if not isinstance(overrides, dict) or set(overrides) != set(GEM5_OVERRIDE_KEYS):
        raise ValueError(
            "gem5_overrides must contain the complete canonical override key set"
        )
    arguments: list[str] = []
    for key in GEM5_OVERRIDE_KEYS:
        arguments.extend((f"--{key.replace('_', '-')}", str(overrides[key])))
    return arguments


def _read_object(path: Path, reasons: list[str], label: str) -> dict | None:
    if not path.is_file():
        reasons.append(f"missing {label}: {path}")
        return None
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        reasons.append(f"cannot read {label} {path}: {error}")
        return None
    if not isinstance(value, dict):
        reasons.append(f"{label} must contain an object: {path}")
        return None
    return value


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


def validate_reuse(fixed_point: Path, clip_point: Path, r1_dir: Path,
                   config_path: Path) -> dict:
    """Return a complete decision for fixed-bin to CLIP-3D R2 reuse."""
    fixed_point = Path(fixed_point).resolve()
    clip_point = Path(clip_point).resolve()
    r1_dir = Path(r1_dir).resolve()
    config_path = Path(config_path).resolve()
    source_latency_path = fixed_point / "r2_latency.json"
    target_latency_path = clip_point / "r2_latency.json"
    source_result_path = fixed_point / "gem5_r2/r2_result.json"
    source_status_path = fixed_point / "gem5_r2/status.json"
    reasons: list[str] = []

    metadata = _read_object(r1_dir / "r1_metadata.json", reasons, "R1 metadata")
    if not (r1_dir / "stats.txt").is_file():
        reasons.append(f"missing R1 stats: {r1_dir / 'stats.txt'}")
    config = _read_object(config_path, reasons, "experiment config")
    source_vector = _read_object(source_latency_path, reasons, "source latency vector")
    target_vector = _read_object(target_latency_path, reasons, "target latency vector")
    fixed_run_config = _read_object(
        fixed_point / "run_config.json", reasons, "fixed run config"
    )
    clip_run_config = _read_object(
        clip_point / "run_config.json", reasons, "CLIP run config"
    )
    fixed_summary = _read_object(
        fixed_point / "pipeline_summary.json", reasons, "fixed pipeline summary"
    )
    clip_summary = _read_object(
        clip_point / "pipeline_summary.json", reasons, "CLIP pipeline summary"
    )

    local = validate_local_result(r1_dir, source_latency_path, fixed_point / "gem5_r2")
    if local["accepted"] is not True:
        reasons.extend(f"source {reason}" for reason in local["reasons"])
    source_result = local.get("result")
    source_status = local.get("status")

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

    if reasons:
        return {
            "accepted": False,
            "decision": "rejected",
            "reasons": reasons,
            "fixed_point": str(fixed_point),
            "clip_point": str(clip_point),
        }

    artifact = {
        "schema_version": 1,
        "decision": "accepted",
        "source_point": str(fixed_point),
        "target_point": str(clip_point),
        "source_latency": {
            "path": str(source_latency_path),
            "sha256": sha256_file(source_latency_path),
        },
        "target_latency": {
            "path": str(target_latency_path),
            "sha256": sha256_file(target_latency_path),
        },
        "gem5_overrides": source_vector["gem5_overrides"],
        "gem5_args": source_vector["gem5_args"],
        "source_result": {
            "path": str(source_result_path),
            "sha256": sha256_file(source_result_path),
        },
        "source_status": {
            "path": str(source_status_path),
            "sha256": sha256_file(source_status_path),
        },
        "source_stats": {
            "path": source_result["stats"],
            "sha256": source_result["stats_sha256"],
        },
        "source_ipc2": source_result["ipc2"],
        "canonical_architecture_key": canonical_key,
        "r1": {
            "directory": str(r1_dir),
            "metadata_path": str((r1_dir / "r1_metadata.json").resolve()),
            "metadata_sha256": sha256_file(r1_dir / "r1_metadata.json"),
            "stats_path": str((r1_dir / "stats.txt").resolve()),
            "stats_sha256": sha256_file(r1_dir / "stats.txt"),
            "instruction_window_scope": metadata.get(
                "instruction_window_scope", "cpu0"
            ),
            "warmup_insts_cpu0": metadata["warmup_insts_cpu0"],
            "measure_insts_cpu0": metadata["measure_insts_cpu0"],
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "embedded": config,
        },
    }
    return {
        "accepted": True,
        "decision": "accepted",
        "reasons": [],
        "artifact": artifact,
    }


def attach_reused_result(fixed_point: Path, clip_point: Path, r1_dir: Path,
                         config_path: Path) -> dict:
    """Attach accepted fixed-bin IPC2 and recompute target CLIP performance."""
    fixed_point = Path(fixed_point).resolve()
    clip_point = Path(clip_point).resolve()
    decision = validate_reuse(fixed_point, clip_point, r1_dir, config_path)
    if decision["accepted"] is not True:
        raise ValueError("R2 reuse rejected: " + "; ".join(decision["reasons"]))

    artifact_path = clip_point / "r2_reuse.json"
    artifact = decision["artifact"]
    write_json(artifact_path, artifact)
    source_result_path = Path(artifact["source_result"]["path"])
    ipc2 = artifact["source_ipc2"]
    run_config = read_json(clip_point / "run_config.json")
    frequency = run_config["config"]["frequency"]
    performance = evaluate(
        clip_point / "modules.json",
        clip_point / "hotspot/thermal_result.json",
        clip_point / "performance.json",
        frequency["f0_ghz"],
        frequency["fmin_ghz"],
        frequency["tsafe_c"],
        frequency["ambient_c"],
        ipc2,
    )
    summary_path = clip_point / "pipeline_summary.json"
    summary = read_json(summary_path)
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
    write_json(summary_path, summary)
    return summary
