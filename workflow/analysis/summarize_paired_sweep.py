#!/usr/bin/env python3
"""Produce a strict, deterministic report for the tracked Balanced-50 R2 pairs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile

from workflow.common import read_json, sha256_file, write_json
from workflow.experiments.balanced50 import (
    EXPECTED_CLASSIFICATION,
    load_selection,
    selection_keys,
    validate_selection,
)
from workflow.r2 import attach_result, reuse_result, run_r2
from workflow.r2.semantic_comparison import compare_semantic_results


LEGACY_SCORE_DEFINITION = (
    "paired measured BIPS2=IPC2*f_sus; exact validated reuse allowed"
)
SEMANTIC_SCORE_DEFINITION = (
    "paired fixed-work throughput=work_units_per_cycle*f_sus; "
    "IPC/BIPS comparison allowed only for an identical instruction vector"
)
PAIR_STATUS_DIRECTORY = "paired_r2_status"

CSV_FIELDS = (
    "workload", "l1d_size", "l2_size",
    "fixed_tmax_c", "clip3d_tmax_c",
    "fixed_frequency_ghz", "clip3d_frequency_ghz",
    "fixed_wire_cycles", "clip3d_wire_cycles",
    "fixed_vector_sha256", "clip3d_vector_sha256",
    "fixed_ipc2", "clip3d_ipc2", "fixed_bips2", "clip3d_bips2",
    "primary_performance_metric", "fixed_primary_score", "clip3d_primary_score",
    "same_trace", "ipc_comparison_allowed", "clip3d_reused_fixed_r2",
    "absolute_bips2_difference", "percent_change",
)


def _read_object(path: Path, label: str) -> dict:
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object: {path}")
    return value


def _positive(value: object, label: str) -> float:
    try:
        number = _finite(value, label)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite positive measured value") from error
    if number <= 0:
        raise ValueError(f"{label} must be a finite positive measured value")
    return number


def _finite(value: object, label: str) -> float:
    """Convert a JSON number without leaking OverflowError or non-finite values."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _same(left: object, right: object) -> bool:
    try:
        return math.isclose(_finite(left, "left value"), _finite(right, "right value"),
                            rel_tol=1e-12, abs_tol=1e-12)
    except ValueError:
        return False


def _path(value: object, expected: Path, label: str) -> None:
    if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
        raise ValueError(f"{label} does not match {expected}")


def _key_dict(key: object) -> dict[str, str]:
    return {
        "workload": key.workload,
        "l1d_size": key.l1d_size,
        "l2_size": key.l2_size,
    }


def _reject_proxy(summary: dict, label: str) -> None:
    """Reject explicit R2-score proxy annotations without banning thermal diagnostics."""
    for field in ("bips2_proxy", "ipc2_proxy", "r2_proxy", "score_proxy"):
        if summary.get(field) not in (None, False):
            raise ValueError(f"{label} has forbidden proxy marker {field}")
    source = summary.get("bips2_source", summary.get("score_source"))
    if isinstance(source, str) and "proxy" in source.lower():
        raise ValueError(f"{label} has forbidden proxy score source")


def _validate_point(point: Path, method: str, key: object, config: dict,
                    config_path: Path, r1_dir: Path,
                    status_metrics: tuple[float, float], reused: bool,
                    fixed_point: Path | None = None) -> dict:
    """Validate one summary against its physical/R2 identity before reporting it."""
    run_config = _read_object(point / "run_config.json", f"{method} run_config")
    summary = _read_object(point / "pipeline_summary.json", f"{method} summary")
    performance = _read_object(point / "performance.json", f"{method} performance")
    vector_path = point / "r2_latency.json"
    vector = _read_object(vector_path, f"{method} latency vector")
    if run_config.get("layout_method") != method:
        raise ValueError(f"{method} run_config layout method mismatch")
    if summary.get("layout_method", summary.get("layout_mode")) != method:
        raise ValueError(f"{method} summary layout method mismatch")
    embedded_config = run_config.get("config")
    if (not isinstance(embedded_config, dict)
            or embedded_config.get("experiment_classification")
            != EXPECTED_CLASSIFICATION):
        raise ValueError(f"{method} run_config classification is not the declared non-formal classification")
    if embedded_config != config:
        raise ValueError(f"{method} run_config has mixed config")
    _path(run_config.get("source"), config_path, f"{method} run_config source")
    expected_key = _key_dict(key)
    r1_metadata = _read_object(r1_dir / "r1_metadata.json", "R1 metadata")
    if {field: r1_metadata.get(field) for field in expected_key} != expected_key:
        raise ValueError(f"{method} R1 metadata architecture differs from selected key")
    if {field: summary.get(field) for field in expected_key} != expected_key:
        raise ValueError(f"{method} summary architecture differs from selected key")
    _path(summary.get("r1"), r1_dir, f"{method} summary R1 directory")
    if ("experiment_classification" in summary
            and summary.get("experiment_classification") != EXPECTED_CLASSIFICATION):
        raise ValueError(f"{method} summary classification is not the declared non-formal classification")
    _reject_proxy(summary, method)

    ipc2 = _positive(summary.get("ipc2"), f"{method} IPC2")
    bips2 = _positive(summary.get("bips2"), f"{method} BIPS2")
    frequency = _positive(summary.get("sustainable_frequency_ghz"), f"{method} frequency")
    tmax_c = _finite(summary.get("tmax_c"), f"{method} Tmax")
    if not _same(bips2, ipc2 * frequency):
        raise ValueError(f"{method} measured BIPS2 is not IPC2*f_sus")
    if not _same(ipc2, status_metrics[0]) or not _same(bips2, status_metrics[1]):
        raise ValueError(f"{method} summary metrics differ from pair status")
    for field, expected in (("ipc2", ipc2), ("bips2", bips2),
                            ("sustainable_frequency_ghz", frequency),
                            ("tmax_f0_c", tmax_c)):
        if not _same(performance.get(field), expected):
            raise ValueError(f"{method} performance {field} differs from summary")
    wire_cycles = vector.get("critical_l1d_to_l2_cycles")
    if (not isinstance(wire_cycles, int) or isinstance(wire_cycles, bool)
            or wire_cycles <= 0 or summary.get("r2_critical_path_cycles") != wire_cycles):
        raise ValueError(f"{method} wire-cycle identity is invalid")

    if reused:
        if fixed_point is None:
            raise ValueError("reuse validation requires the paired fixed point")
        _validate_reuse(fixed_point, point, r1_dir, config_path, summary,
                        performance, ipc2, bips2)
    else:
        _validate_local(
            r1_dir, point, summary, performance, ipc2, bips2
        )
    r2_source = summary.get("r2_source")
    if not isinstance(r2_source, str):
        raise ValueError(f"{method} summary lacks an R2 source")
    r2_result = _read_object(Path(r2_source), f"{method} R2 result")
    primary_metric = summary.get("primary_performance_metric", "bips2")
    work_rate = None
    work_score = None
    if primary_metric == "work_units_per_ns":
        work_rate = _positive(
            summary.get("work_units_per_cycle"), f"{method} work-unit rate"
        )
        work_score = _positive(
            summary.get("work_units_per_ns"), f"{method} frequency-scaled work rate"
        )
        if (r2_result.get("instruction_window_scope") != "semantic-work"
                or not _same(r2_result.get("work_units_per_cycle"), work_rate)
                or not _same(work_score, work_rate * frequency)
                or not _same(performance.get("work_units_per_cycle"), work_rate)
                or not _same(performance.get("work_units_per_ns"), work_score)):
            raise ValueError(f"{method} semantic fixed-work metrics are inconsistent")
    elif primary_metric != "bips2":
        raise ValueError(f"{method} summary primary performance metric is unsupported")
    return {
        "tmax_c": tmax_c,
        "frequency_ghz": frequency,
        "wire_cycles": wire_cycles,
        "vector_sha256": sha256_file(vector_path),
        "ipc2": ipc2,
        "bips2": bips2,
        "primary_performance_metric": primary_metric,
        "work_units_per_cycle": work_rate,
        "work_units_per_ns": work_score,
        "r2_result": r2_result,
    }


def _validate_local(r1_dir: Path, point: Path, summary: dict,
                    performance: dict, ipc2: float, bips2: float) -> None:
    result_path = (point / "gem5_r2/r2_result.json").resolve()
    status_path = result_path.parent / "status.json"
    run_config = _read_object(point / "run_config.json", "local run config")
    config_source = run_config.get("source")
    decision = attach_result.validate_local_attachment(
        point, r1_dir,
        Path(config_source) if isinstance(config_source, str) else None,
    )
    if decision.get("accepted") is not True:
        raise ValueError("local R2 provenance rejected: " + "; ".join(
            decision.get("reasons", [])
        ))
    if (decision.get("summary") != summary
            or decision.get("performance") != performance):
        raise ValueError(
            "local physical snapshot changed between report validation layers"
        )
    if summary.get("r2_reused") is True or (point / "r2_reuse.json").exists():
        raise ValueError("separate CLIP/local row is marked as reuse")
    _path(summary.get("r2_source"), result_path, "local R2 source")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("local summary artifacts are missing")
    _path(artifacts.get("r2_result"), result_path, "local R2 result artifact")
    result = _read_object(result_path, "local R2 result")
    status = _read_object(status_path, "local R2 status")
    if status.get("state") != "success":
        raise ValueError("local R2 status is not successful")
    if not _same(result.get("ipc2"), ipc2) or not _same(status.get("ipc2"), ipc2):
        raise ValueError("local R2 IPC2 provenance differs from summary")
    # bips2 is intentionally recomputed from IPC2 and this point's sustainable
    # frequency, never copied from a local gem5 result.
    _positive(bips2, "local BIPS2")


def _validate_reuse(fixed_point: Path, point: Path, r1_dir: Path,
                    config_path: Path, summary: dict, performance: dict,
                    ipc2: float, bips2: float) -> None:
    marker_path = (point / "r2_reuse.json").resolve()
    marker = _read_object(marker_path, "R2 reuse marker")
    decision = reuse_result.validate_reuse(fixed_point, point, r1_dir, config_path)
    if decision.get("accepted") is not True:
        raise ValueError("reuse provenance rejected: " + "; ".join(
            decision.get("reasons", [])
        ))
    expected = decision.get("artifact")
    if not isinstance(expected, dict) or any(
            marker.get(field) != value for field, value in expected.items()):
        raise ValueError("reuse marker differs from Task-5 validated provenance")
    if marker.get("decision") != "accepted":
        raise ValueError("reuse marker is not accepted")
    source = marker.get("source_result")
    outputs = marker.get("target_outputs")
    if not isinstance(source, dict) or not isinstance(outputs, dict):
        raise ValueError("reuse marker provenance is malformed")
    if not _same(marker.get("source_ipc2"), ipc2):
        raise ValueError("reuse source IPC2 differs from target summary")
    _path(summary.get("r2_reuse_artifact"), marker_path, "reuse summary marker")
    _path(summary.get("r2_source"), Path(source["path"]), "reuse summary source")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("reuse summary artifacts are missing")
    _path(artifacts.get("r2_result"), Path(source["path"]), "reuse result artifact")
    _path(artifacts.get("r2_reuse"), marker_path, "reuse marker artifact")
    if summary.get("r2_reused") is not True:
        raise ValueError("reuse summary lacks r2_reused marker")
    expected_bips2 = ipc2 * _positive(performance.get("sustainable_frequency_ghz"),
                                      "reuse performance frequency")
    if not _same(expected_bips2, bips2):
        raise ValueError("reuse BIPS2 does not use target sustainable frequency")
    for name, path in (("performance", point / "performance.json"),
                       ("summary", point / "pipeline_summary.json")):
        output = outputs.get(name)
        if (not isinstance(output, dict) or output.get("path") != str(path.resolve())
                or output.get("sha256") != sha256_file(path)):
            raise ValueError(f"reuse marker {name} output binding mismatch")
    if not _same(outputs.get("bips2"), bips2):
        raise ValueError("reuse marker BIPS2 differs from target summary")


def _load_statuses(status_root: Path, keys: list[object], config_path: Path,
                   fixed_root: Path, clip_root: Path) -> dict[object, dict]:
    if not status_root.is_dir():
        raise ValueError(f"paired status directory is missing: {status_root}")
    paths = sorted(status_root.rglob("pair_status.json"))
    if len(paths) != len(keys):
        raise ValueError(f"paired status count is {len(paths)}, expected {len(keys)}")
    expected = {_key_dict(key)["workload"] + "\0" + _key_dict(key)["l1d_size"]
                + "\0" + _key_dict(key)["l2_size"]: key for key in keys}
    rows: dict[object, dict] = {}
    expected_hash = sha256_file(config_path)
    for path in paths:
        status = _read_object(path, "pair status")
        key = status.get("key")
        if not isinstance(key, dict):
            raise ValueError(f"pair status key is malformed: {path}")
        token = "\0".join(str(key.get(field, "")) for field in
                          ("workload", "l1d_size", "l2_size"))
        selected = expected.get(token)
        if selected is None:
            raise ValueError(f"pair status key is not selected: {path}")
        if selected in rows:
            raise ValueError(f"duplicate pair status key: {key}")
        expected_path = (status_root / selected.relative_path() / "pair_status.json").resolve()
        if path.resolve() != expected_path:
            raise ValueError(f"pair status path does not match its architecture key: {path}")
        _path(status.get("pair_status"), expected_path, "pair status self-binding")
        if status.get("schema_version") != 1 or status.get("state") != "success":
            raise ValueError(f"pair status is not a successful Task-6 record: {path}")
        if status.get("config") != str(config_path.resolve()) or status.get("config_sha256") != expected_hash:
            raise ValueError(f"pair status has mixed config provenance: {path}")
        relative = selected.relative_path()
        _path(status.get("fixed_point"), fixed_root / relative, "pair fixed point")
        _path(status.get("clip3d_point"), clip_root / relative, "pair CLIP point")
        r1_directory = status.get("r1_directory")
        if not isinstance(r1_directory, str) or not Path(r1_directory).is_dir():
            raise ValueError(f"pair R1 directory is invalid: {path}")
        reused = status.get("clip3d_reused_fixed_r2")
        if reused not in (True, False):
            raise ValueError(f"pair reuse classification is malformed: {path}")
        if status.get("physical_r2_runs") != (1 if reused else 2):
            raise ValueError(f"pair physical R2 count is inconsistent: {path}")
        rows[selected] = status
    if set(rows) != set(keys):
        raise ValueError("paired statuses have missing selected architecture keys")
    return rows


def _statistics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("statistics require at least one paired row")
    ratios = [_positive(row["ratio"], "BIPS2 ratio") for row in rows]
    changes = [_finite(row["percent_change"], "percentage change") for row in rows]
    result = {
        "n": len(rows),
        "arithmetic_mean_ratio": sum(ratios) / len(ratios),
        "geometric_mean_ratio": math.exp(sum(math.log(value) for value in ratios) / len(ratios)),
        "median_percent_change": median(changes),
        "wins": sum(value > 1.0 for value in ratios),
        "ties": sum(value == 1.0 for value in ratios),
        "losses": sum(value < 1.0 for value in ratios),
        "fixed_to_clip_reuse_count": sum(row["clip3d_reused_fixed_r2"] for row in rows),
        "separate_clip_r2_count": sum(not row["clip3d_reused_fixed_r2"] for row in rows),
    }
    for field in ("arithmetic_mean_ratio", "geometric_mean_ratio",
                  "median_percent_change"):
        _finite(result[field], f"derived {field}")
    return result


def _csv_row(row: dict) -> dict:
    result = dict(row)
    for field in ("fixed_tmax_c", "clip3d_tmax_c"):
        result[field] = f"{result[field]:.6f}"
    return {field: result[field] for field in CSV_FIELDS}


def _replace(source: Path, destination: Path) -> None:
    """Single publication seam so a failed second replace can be regression-tested."""
    source.replace(destination)


def _stage_bytes(directory: Path, stem: str, data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{stem}-", dir=directory)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _restore_bytes(destination: Path, previous: bytes | None) -> None:
    if previous is None:
        destination.unlink(missing_ok=True)
        return
    staged = _stage_bytes(destination.parent, destination.name + ".rollback", previous)
    os.replace(staged, destination)


def _publish_reports(csv_path: Path, json_path: Path, csv_bytes: bytes,
                     json_bytes: bytes) -> None:
    """Publish both reports as one rollback-capable two-file transaction."""
    csv_path = csv_path.resolve()
    json_path = json_path.resolve()
    if csv_path == json_path:
        raise ValueError("CSV and JSON output paths must be distinct")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    previous = {
        csv_path: csv_path.read_bytes() if csv_path.exists() else None,
        json_path: json_path.read_bytes() if json_path.exists() else None,
    }
    staged = {
        csv_path: _stage_bytes(csv_path.parent, csv_path.name, csv_bytes),
        json_path: _stage_bytes(json_path.parent, json_path.name, json_bytes),
    }
    published: list[Path] = []
    try:
        for destination in (csv_path, json_path):
            _replace(staged[destination], destination)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            _restore_bytes(destination, previous[destination])
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def summarize(fixed_root: Path, clip_root: Path, selection_path: Path,
              config_path: Path, csv_path: Path, json_path: Path, *,
              status_root: Path | None = None) -> dict:
    """Validate exactly the tracked pairs, then write deterministic CSV and JSON reports."""
    fixed_root = Path(fixed_root).resolve()
    clip_root = Path(clip_root).resolve()
    config_path = Path(config_path).resolve()
    status_root = (
        fixed_root.parent / PAIR_STATUS_DIRECTORY
        if status_root is None else Path(status_root).resolve()
    )
    selection = load_selection(Path(selection_path))
    grid_reference = selection["canonical_grid_config"]["path"]
    grid_path = Path(__file__).resolve().parents[2] / grid_reference
    grid = _read_object(grid_path, "canonical grid")
    validate_selection(selection, grid)
    pinned = selection["experiment_config"]
    if (config_path != (Path(__file__).resolve().parents[2] / pinned["path"]).resolve()
            or sha256_file(config_path) != pinned["sha256"]):
        raise ValueError("supplied config does not match the pinned selection config")
    config = _read_object(config_path, "experiment config")
    if config.get("experiment_classification") != EXPECTED_CLASSIFICATION:
        raise ValueError("experiment config classification is not the declared non-formal classification")
    keys = selection_keys(selection)
    if len(keys) != 50 or len(set(keys)) != 50:
        raise ValueError("selection must contain exactly 50 unique keys")
    statuses = _load_statuses(status_root, keys,
                              config_path, fixed_root, clip_root)
    rows = []
    for key in keys:
        relative = key.relative_path()
        status = statuses[key]
        r1_dir = Path(status["r1_directory"]).resolve()
        fixed = _validate_point(
            fixed_root / relative, "fixed-bin", key, config, config_path,
            r1_dir,
            (_positive(status.get("fixed_ipc2"), "pair fixed IPC2"),
             _positive(status.get("fixed_bips2"), "pair fixed BIPS2")), False,
        )
        reused = status["clip3d_reused_fixed_r2"]
        clip = _validate_point(
            clip_root / relative, "clip3d", key, config, config_path,
            r1_dir,
            (_positive(status.get("clip3d_ipc2"), "pair CLIP IPC2"),
             _positive(status.get("clip3d_bips2"), "pair CLIP BIPS2")), reused,
            fixed_root / relative,
        )
        if status.get("fixed_vector_sha256") != fixed["vector_sha256"]:
            raise ValueError("pair fixed vector hash differs from fixed point")
        if status.get("clip3d_vector_sha256") != clip["vector_sha256"]:
            raise ValueError("pair CLIP vector hash differs from CLIP point")
        pair_is_semantic = (
            fixed["primary_performance_metric"] == "work_units_per_ns"
            and clip["primary_performance_metric"] == "work_units_per_ns"
        )
        if ((fixed["primary_performance_metric"] == "work_units_per_ns")
                != (clip["primary_performance_metric"] == "work_units_per_ns")):
            raise ValueError("pair mixes legacy and semantic performance metrics")
        semantic = status.get("semantic_comparison")
        if pair_is_semantic:
            recomputed = compare_semantic_results(
                fixed["r2_result"], clip["r2_result"],
                fixed["frequency_ghz"], clip["frequency_ghz"],
            )
            if not isinstance(semantic, dict) or semantic != recomputed:
                raise ValueError("pair semantic comparison differs from live R2 evidence")
            fixed_score = _positive(
                semantic.get("fixed_score"), "fixed semantic score"
            )
            clip_score = _positive(
                semantic.get("clip3d_score"), "CLIP semantic score"
            )
            primary_metric = semantic.get("primary_metric")
            if primary_metric != "work_units_per_ns":
                raise ValueError("pair semantic primary metric is unsupported")
            same_trace = semantic.get("same_trace")
            ipc_allowed = semantic.get("ipc_comparison_allowed")
            if not isinstance(same_trace, bool) or ipc_allowed is not same_trace:
                raise ValueError("pair same-trace IPC gate is malformed")
        else:
            if semantic is not None:
                raise ValueError("legacy pair unexpectedly contains semantic comparison")
            fixed_score = fixed["bips2"]
            clip_score = clip["bips2"]
            primary_metric = "bips2"
            same_trace = None
            ipc_allowed = None
        ratio = _positive(clip_score / fixed_score, "primary score ratio")
        difference = _finite(abs(clip["bips2"] - fixed["bips2"]),
                             "absolute diagnostic BIPS2 difference")
        percent_change = _finite(100.0 * (ratio - 1.0), "percentage change")
        rows.append({
            **_key_dict(key),
            "fixed_tmax_c": fixed["tmax_c"], "clip3d_tmax_c": clip["tmax_c"],
            "fixed_frequency_ghz": fixed["frequency_ghz"],
            "clip3d_frequency_ghz": clip["frequency_ghz"],
            "fixed_wire_cycles": fixed["wire_cycles"],
            "clip3d_wire_cycles": clip["wire_cycles"],
            "fixed_vector_sha256": fixed["vector_sha256"],
            "clip3d_vector_sha256": clip["vector_sha256"],
            "fixed_ipc2": fixed["ipc2"], "clip3d_ipc2": clip["ipc2"],
            "fixed_bips2": fixed["bips2"], "clip3d_bips2": clip["bips2"],
            "primary_performance_metric": primary_metric,
            "fixed_primary_score": fixed_score,
            "clip3d_primary_score": clip_score,
            "same_trace": same_trace,
            "ipc_comparison_allowed": ipc_allowed,
            "clip3d_reused_fixed_r2": reused,
            "absolute_bips2_difference": difference,
            "percent_change": percent_change,
            "ratio": ratio,
        })
    metric_names = {row["primary_performance_metric"] for row in rows}
    if len(metric_names) != 1:
        raise ValueError(
            "paired report cannot mix legacy instruction windows and semantic-work ROIs"
        )
    score_definition = (
        SEMANTIC_SCORE_DEFINITION
        if metric_names == {"work_units_per_ns"}
        else LEGACY_SCORE_DEFINITION
    )
    workloads = {}
    for workload in dict.fromkeys(row["workload"] for row in rows):
        workloads[workload] = _statistics([row for row in rows if row["workload"] == workload])
    result = {
        "schema_version": 1,
        "complete": len(rows) == 50 and all(item["n"] == 10 for item in workloads.values()),
        "score_definition": score_definition,
        "point_count": len(rows),
        "selection": str(Path(selection_path).resolve()),
        "status_root": str(status_root),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "classification": config["experiment_classification"],
        "experiment_classification": config["experiment_classification"],
        "parameters": {
            "technology_nm": config.get("technology_nm"),
            "frequency": config.get("frequency"),
            "physical": config.get("physical"),
            "layout_optimizer": config.get("layout_optimizer"),
            "delay": config.get("delay"),
        },
        "workloads": workloads,
        "aggregate": _statistics(rows),
        "rows": [{field: row[field] for field in CSV_FIELDS} for row in rows],
    }
    csv_buffer = io.StringIO(newline="")
    with csv_buffer:
        writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
    try:
        json_bytes = (json.dumps(result, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"paired report is not valid finite JSON: {error}") from error
    _publish_reports(Path(csv_path), Path(json_path), csv_bytes, json_bytes)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status-root", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = summarize(args.fixed_root, args.clip_root, args.selection,
                           args.config, args.csv, args.output,
                           status_root=args.status_root)
    except (OSError, ValueError) as error:
        print(f"paired summary failed: {error}")
        return 1
    print(json.dumps({"complete": result["complete"],
                      "point_count": result["point_count"]}, sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
