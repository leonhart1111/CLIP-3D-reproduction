#!/usr/bin/env python3
"""Attach and certify a separately completed gem5 R2 result."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile

from workflow.common import (
    atomic_write_bytes,
    instruction_window_scope,
    read_json,
    write_json,
)
from workflow.r2.attachment_validation import (
    exclusive_point_lock,
    is_supported_instruction_window_scope,
    validate_physical_coherence,
)
from workflow.r2.run_r2 import validate_local_result
from workflow.thermal.sustainable_frequency import evaluate


def _capture(path: Path, snapshots: dict[Path, bytes]) -> dict:
    path = Path(path).resolve()
    data = path.read_bytes()
    snapshots[path] = data
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return value


def _snapshots_unchanged(snapshots: dict[Path, bytes]) -> bool:
    return all(path.read_bytes() == data for path, data in snapshots.items())


def _finite_nonnegative(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0)


def _local_record_errors(point_dir: Path, r1_dir: Path, result: dict,
                         summary: dict) -> list[str]:
    reasons: list[str] = []
    result_path = (point_dir / "gem5_r2/r2_result.json").resolve()
    if summary.get("r1") != str(r1_dir.resolve()):
        reasons.append("local summary R1 directory is invalid")
    if summary.get("output") != str(point_dir.resolve()):
        reasons.append("local summary output directory is invalid")
    if summary.get("r2_source") != str(result_path):
        reasons.append("local summary R2 source is invalid")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        reasons.append("local summary artifacts are missing")
    else:
        for field, expected in (
                ("config", point_dir / "run_config.json"),
                ("modules", point_dir / "modules.json"),
                ("thermal", point_dir / "hotspot/thermal_result.json"),
                ("performance", point_dir / "performance.json"),
                ("r2_latency", point_dir / "r2_latency.json"),
                ("r2_result", result_path)):
            value = artifacts.get(field)
            if (not isinstance(value, str)
                    or Path(value).resolve() != expected.resolve()):
                reasons.append(f"local summary {field} artifact is invalid")
    if (summary.get("r2_reused") is True
            or summary.get("r2_reuse_artifact") is not None
            or (isinstance(artifacts, dict)
                and artifacts.get("r2_reuse") is not None)
            or (point_dir / "r2_reuse.json").exists()):
        reasons.append("local attachment is contaminated by reuse state")
    elapsed = result.get("elapsed_seconds")
    stage_seconds = summary.get("stage_seconds")
    if (not _finite_nonnegative(elapsed) or not isinstance(stage_seconds, dict)
            or stage_seconds.get("gem5_r2") != elapsed):
        reasons.append("local summary gem5 R2 elapsed time is invalid")
    if isinstance(stage_seconds, dict):
        values = list(stage_seconds.values())
        if not all(value is None or _finite_nonnegative(value) for value in values):
            reasons.append("local summary stage times are invalid")
        else:
            expected_total = sum(float(value) for value in values if value is not None)
            total = summary.get("total_pipeline_seconds")
            if (not _finite_nonnegative(total) or not math.isclose(
                    float(total), expected_total, rel_tol=1e-12, abs_tol=1e-12)):
                reasons.append("local summary total pipeline time is invalid")
    return reasons


def validate_local_attachment(point_dir: Path, r1_dir: Path | None = None,
                              config_path: Path | None = None) -> dict:
    """Certify one local attachment against current physical and R2 evidence."""
    point_dir = Path(point_dir).resolve()
    snapshots: dict[Path, bytes] = {}
    reasons: list[str] = []
    try:
        run_config = _capture(point_dir / "run_config.json", snapshots)
        modules = _capture(point_dir / "modules.json", snapshots)
        thermal = _capture(point_dir / "hotspot/thermal_result.json", snapshots)
        performance = _capture(point_dir / "performance.json", snapshots)
        summary = _capture(point_dir / "pipeline_summary.json", snapshots)
        vector_path = point_dir / "r2_latency.json"
        _capture(vector_path, snapshots)
        result_path = point_dir / "gem5_r2/r2_result.json"
        status_path = point_dir / "gem5_r2/status.json"
        captured_result = _capture(result_path, snapshots)
        captured_status = _capture(status_path, snapshots)
        if r1_dir is None:
            r1_value = summary.get("r1", modules.get("source_r1"))
            if not isinstance(r1_value, str) or not r1_value:
                raise ValueError("local attachment lacks an R1 directory")
            r1_dir = Path(r1_value)
        r1_dir = Path(r1_dir).resolve()
        metadata = _capture(r1_dir / "r1_metadata.json", snapshots)
        r1_stats_path = (r1_dir / "stats.txt").resolve()
        r1_stats_bytes = r1_stats_path.read_bytes()
        snapshots[r1_stats_path] = r1_stats_bytes
        thermal_manifest = _capture(
            point_dir / "hotspot/hotspot_manifest.json", snapshots
        )
        steady_path = (point_dir / "hotspot/steady.txt").resolve()
        thermal_steady_bytes = steady_path.read_bytes()
        snapshots[steady_path] = thermal_steady_bytes
        config = run_config.get("config")
        if not isinstance(config, dict):
            raise ValueError("local run_config lacks an embedded config")
        if config_path is not None:
            config_path = Path(config_path).resolve()
            if run_config.get("source") != str(config_path):
                reasons.append("local run_config source differs from selected config")
            selected_config = _capture(config_path, snapshots)
            if config != selected_config:
                reasons.append("local embedded config differs from selected config")
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return {"accepted": False, "reasons": [f"cannot capture local attachment: {error}"]}

    local = validate_local_result(r1_dir, vector_path, point_dir / "gem5_r2")
    if local.get("accepted") is not True:
        reasons.extend(local.get("reasons", []))
    if local.get("result") != captured_result:
        reasons.append("local R2 result changed during attachment validation")
    if local.get("status") != captured_status:
        reasons.append("local R2 status changed during attachment validation")
    ipc2 = captured_result.get("ipc2")
    validate_physical_coherence(
        metadata, config, modules, thermal, performance, summary, r1_dir,
        point_dir, reasons, expected_ipc2=ipc2,
        expected_work_units_per_cycle=captured_result.get(
            "work_units_per_cycle"
        ),
        r1_stats_bytes=r1_stats_bytes,
        thermal_steady_bytes=thermal_steady_bytes,
        thermal_manifest=thermal_manifest,
    )
    reasons.extend(_local_record_errors(point_dir, r1_dir, captured_result, summary))
    try:
        if not _snapshots_unchanged(snapshots):
            reasons.append("local attachment inputs changed during validation")
    except OSError as error:
        reasons.append(f"local attachment inputs changed during validation: {error}")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "result": captured_result,
        "status": captured_status,
        "summary": summary,
        "performance": performance,
        "result_path": str(result_path.resolve()),
        "status_path": str(status_path.resolve()),
    }


def _publish_bytes(path: Path, data: bytes) -> str:
    atomic_write_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


def attach(point_dir: Path) -> dict:
    """Recompute and publish a coherent local result attachment transaction."""
    point_dir = Path(point_dir).resolve()
    with exclusive_point_lock(point_dir):
        summary_path = point_dir / "pipeline_summary.json"
        performance_path = point_dir / "performance.json"
        snapshots: dict[Path, bytes] = {}
        try:
            run_config = _capture(point_dir / "run_config.json", snapshots)
            modules = _capture(point_dir / "modules.json", snapshots)
            thermal = _capture(point_dir / "hotspot/thermal_result.json", snapshots)
            summary = _capture(summary_path, {})
            performance_before = performance_path.read_bytes()
            summary_before = summary_path.read_bytes()
            r1_value = summary.get("r1", modules.get("source_r1"))
            if not isinstance(r1_value, str) or not r1_value:
                raise ValueError("local attachment lacks an R1 directory")
            r1_dir = Path(r1_value).resolve()
            metadata = _capture(r1_dir / "r1_metadata.json", snapshots)
            r1_stats_path = (r1_dir / "stats.txt").resolve()
            r1_stats_bytes = r1_stats_path.read_bytes()
            snapshots[r1_stats_path] = r1_stats_bytes
            thermal_manifest = _capture(
                point_dir / "hotspot/hotspot_manifest.json", snapshots
            )
            steady_path = (point_dir / "hotspot/steady.txt").resolve()
            thermal_steady_bytes = steady_path.read_bytes()
            snapshots[steady_path] = thermal_steady_bytes
            vector_path = point_dir / "r2_latency.json"
            snapshots[vector_path.resolve()] = vector_path.read_bytes()
            result_path = point_dir / "gem5_r2/r2_result.json"
            status_path = point_dir / "gem5_r2/status.json"
            result = _capture(result_path, snapshots)
            _capture(status_path, snapshots)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"cannot capture local attachment inputs: {error}") from error

        architecture = modules.get("architecture")
        if (not isinstance(architecture, dict)
                or not is_supported_instruction_window_scope(
                    instruction_window_scope(metadata)
                )
                or not is_supported_instruction_window_scope(
                    instruction_window_scope(architecture)
                )):
            raise ValueError(
                "local attachment rejected: target modules architecture "
                "instruction_window_scope is unsupported"
            )

        local = validate_local_result(r1_dir, vector_path, point_dir / "gem5_r2")
        if local.get("accepted") is not True or local.get("result") != result:
            raise ValueError(
                "local R2 validation rejected: "
                + "; ".join(local.get("reasons", []))
            )
        config = run_config.get("config")
        if not isinstance(config, dict):
            raise ValueError("local run_config lacks an embedded config")
        frequency = config.get("frequency")
        if not isinstance(frequency, dict):
            raise ValueError("local config lacks frequency inputs")

        with tempfile.TemporaryDirectory(prefix=".r2-local-", dir=point_dir) as name:
            staging = Path(name)
            staged_modules = staging / "modules.json"
            staged_thermal = staging / "thermal_result.json"
            staged_performance = staging / "performance.json"
            staged_summary = staging / "pipeline_summary.json"
            staged_modules.write_bytes(snapshots[(point_dir / "modules.json").resolve()])
            staged_thermal.write_bytes(
                snapshots[(point_dir / "hotspot/thermal_result.json").resolve()]
            )
            performance = evaluate(
                staged_modules, staged_thermal, staged_performance,
                frequency["f0_ghz"], frequency["fmin_ghz"],
                frequency["tsafe_c"], frequency["ambient_c"], result["ipc2"],
                result.get("work_units_per_cycle"),
            )
            published_summary = deepcopy(summary)
            published_summary.update({
                "gamma": performance["gamma"],
                "tmax_c": performance["tmax_f0_c"],
                "sustainable_frequency_ghz": performance[
                    "sustainable_frequency_ghz"
                ],
                "ipc1": performance["ipc1"],
                "bips1_thermal": performance["bips1_thermal"],
                "ipc2": performance["ipc2"],
                "bips2": performance["bips2"],
                "primary_performance_metric": performance.get(
                    "primary_performance_metric", "bips2"
                ),
                "work_units_per_cycle": performance.get("work_units_per_cycle"),
                "work_units_per_ns": performance.get("work_units_per_ns"),
                "r2_source": str(result_path.resolve()),
            })
            published_summary.pop("r2_reused", None)
            published_summary.pop("r2_reuse_artifact", None)
            artifacts = published_summary.get("artifacts")
            artifacts = deepcopy(artifacts) if isinstance(artifacts, dict) else {}
            artifacts.update({
                "config": str((point_dir / "run_config.json").resolve()),
                "modules": str((point_dir / "modules.json").resolve()),
                "thermal": str(
                    (point_dir / "hotspot/thermal_result.json").resolve()
                ),
                "performance": str(performance_path.resolve()),
                "r2_latency": str(vector_path.resolve()),
                "r2_result": str(result_path.resolve()),
            })
            published_summary["artifacts"] = artifacts
            artifacts.pop("r2_reuse", None)
            stage_seconds = published_summary.setdefault("stage_seconds", {})
            stage_seconds["gem5_r2"] = result.get("elapsed_seconds")
            published_summary["total_pipeline_seconds"] = sum(
                float(value) for value in stage_seconds.values()
                if value is not None
            )
            write_json(staged_summary, published_summary)

            reasons: list[str] = []
            validate_physical_coherence(
                metadata, config, modules, thermal, performance,
                published_summary, r1_dir, point_dir, reasons,
                expected_ipc2=result["ipc2"],
                expected_work_units_per_cycle=result.get(
                    "work_units_per_cycle"
                ),
                r1_stats_bytes=r1_stats_bytes,
                thermal_steady_bytes=thermal_steady_bytes,
                thermal_manifest=thermal_manifest,
            )
            reasons.extend(_local_record_errors(
                point_dir, r1_dir, result, published_summary
            ))
            # The marker is removed only after the staged local summary has been
            # checked; ignore its old existence while validating the replacement.
            reasons = [reason for reason in reasons
                       if reason != "local attachment is contaminated by reuse state"]
            if reasons:
                raise ValueError("local attachment rejected: " + "; ".join(reasons))
            if not _snapshots_unchanged(snapshots):
                raise ValueError("local attachment inputs changed before publication")

            performance_bytes = staged_performance.read_bytes()
            summary_bytes = staged_summary.read_bytes()
            performance_digest = hashlib.sha256(performance_bytes).hexdigest()
            summary_digest = hashlib.sha256(summary_bytes).hexdigest()
            published: list[tuple[Path, str, bytes]] = []
            try:
                _publish_bytes(performance_path, performance_bytes)
                published.append((performance_path, performance_digest,
                                  performance_before))
                _publish_bytes(summary_path, summary_bytes)
                published.append((summary_path, summary_digest, summary_before))
                if (not _snapshots_unchanged(snapshots)
                        or hashlib.sha256(performance_path.read_bytes()).hexdigest()
                        != performance_digest
                        or hashlib.sha256(summary_path.read_bytes()).hexdigest()
                        != summary_digest):
                    raise ValueError("local attachment changed before final commit")
                (point_dir / "r2_reuse.json").unlink(missing_ok=True)
                return published_summary
            except Exception:
                for path, digest, previous in reversed(published):
                    try:
                        current = hashlib.sha256(path.read_bytes()).hexdigest()
                    except OSError:
                        current = None
                    if current == digest:
                        atomic_write_bytes(path, previous)
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = attach(args.point_dir.resolve())
    print(f"attached IPC2={summary['ipc2']:.6f}, BIPS2={summary['bips2']:.6f}")


if __name__ == "__main__":
    main()
