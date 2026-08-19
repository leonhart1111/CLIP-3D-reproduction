"""Fail-closed evidence helpers for paired transient-ROM validation."""

from __future__ import annotations

import csv
from io import StringIO
import math
from numbers import Integral, Real
from pathlib import Path

from workflow.common import atomic_write_bytes, read_json, sha256_file, write_json
from workflow.r2.attachment_validation import validate_vector_native_cache
from workflow.r2.semantic_comparison import compare_semantic_results


def _finite_optional(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite positive number or None")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} must be a finite positive number or None")
    return normalized


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{label} must be a non-negative integer wire cycle")
    return int(value)


def require_selected_cycle_identity(
    selected: dict, vector: dict, wire_aggregation: str,
) -> int:
    """Require optimizer, vector component, and layout-delay cycles to agree."""
    if wire_aggregation not in ("mean", "maximum", "traffic-weighted"):
        raise ValueError("wire aggregation is invalid")
    if vector.get("wire_cycle_aggregation_for_r2") != wire_aggregation:
        raise ValueError("R2 wire aggregation differs from the optimizer")
    components = vector.get("components_cycles")
    layout_delays = vector.get("layout_delays")
    if not isinstance(components, dict) or not isinstance(layout_delays, dict):
        raise ValueError("R2 vector lacks integer wire cycle evidence")
    delay_field = {
        "mean": "wire_cycles",
        "maximum": "maximum_wire_cycles",
        "traffic-weighted": "traffic_weighted_wire_cycles",
    }[wire_aggregation]
    values = (
        _integer(selected.get("r2_wire_cycles"), "optimizer integer wire cycle"),
        _integer(components.get("layout_wire"), "R2 component integer wire cycle"),
        _integer(layout_delays.get(delay_field), "layout integer wire cycle"),
    )
    if len(set(values)) != 1:
        raise ValueError(
            "optimizer and R2 integer wire cycle representations differ: "
            f"optimizer={values[0]}, component={values[1]}, layout={values[2]}"
        )
    return values[0]


def branch_metrics(ipc2: float | None, f_hotspot_ghz: float | None,
                   work_units_per_cycle: float | None = None) -> dict:
    """Name measured IPC and validated HotSpot frequency without proxy aliases."""
    ipc = _finite_optional(ipc2, "measured IPC2")
    frequency = _finite_optional(
        f_hotspot_ghz, "validated HotSpot sustainable frequency"
    )
    result = {
        "validated_f_sus_trans_hotspot_ghz": frequency,
        "measured_ipc2": ipc,
        "measured_bips2_trans": (
            ipc * frequency if ipc is not None and frequency is not None else None
        ),
    }
    work_rate = _finite_optional(
        work_units_per_cycle, "measured work units per cycle"
    )
    if work_rate is not None:
        result.update({
            "primary_performance_metric": "work_units_per_ns",
            "measured_work_units_per_cycle": work_rate,
            "measured_work_units_per_ns": (
                work_rate * frequency if frequency is not None else None
            ),
        })
    return result


def _validated_artifact(artifacts: dict, field: str, label: str) -> Path:
    value = artifacts.get(field)
    digest = artifacts.get(f"{field}_sha256")
    if not isinstance(value, str) or not value or not isinstance(digest, str):
        raise ValueError(f"{label} lacks measured artifact identity for {field}")
    path = Path(value).resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label} measured artifact identity differs for {field}")
    return path


def _validated_branch(branch: dict, expected_name: str) -> tuple[float, dict, dict]:
    if not isinstance(branch, dict) or branch.get("branch") != expected_name:
        raise ValueError(f"paired comparison requires the {expected_name} branch")
    if branch.get("r2_executed") is not True:
        raise ValueError(f"{expected_name} lacks a completed real R2 measurement")
    if (branch.get("failure") is not None
            or branch.get("validation_classification") != "validated"):
        raise ValueError(f"{expected_name} is not a validated measured branch")
    ipc = _finite_optional(branch.get("measured_ipc2"), f"{expected_name} IPC2")
    frequency = _finite_optional(
        branch.get("validated_f_sus_trans_hotspot_ghz"),
        f"{expected_name} HotSpot frequency",
    )
    claimed_bips = _finite_optional(
        branch.get("measured_bips2_trans"),
        f"{expected_name} measured BIPS2_trans",
    )
    if ipc is None or frequency is None or claimed_bips is None:
        raise ValueError(f"{expected_name} lacks measured BIPS2_trans inputs")
    bips = ipc * frequency
    if not math.isclose(claimed_bips, bips, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"{expected_name} measured BIPS2_trans is inconsistent")
    controls = branch.get("controls")
    artifacts = branch.get("artifacts")
    if not isinstance(controls, dict) or not isinstance(artifacts, dict):
        raise ValueError(f"{expected_name} lacks controls or artifact identity")
    _validated_artifact(artifacts, "layout", expected_name)
    hotspot_path = _validated_artifact(artifacts, "hotspot", expected_name)
    latency_path = _validated_artifact(artifacts, "r2_latency", expected_name)
    r2_path = _validated_artifact(artifacts, "r2_result", expected_name)
    hotspot_result = read_json(hotspot_path)
    latency_vector = read_json(latency_path)
    r2_result = read_json(r2_path)
    if (not isinstance(hotspot_result, dict)
            or hotspot_result.get("f_sus_trans_ghz") != frequency):
        raise ValueError(f"{expected_name} HotSpot artifact frequency differs")
    if not isinstance(r2_result, dict) or r2_result.get("ipc2") != ipc:
        raise ValueError(f"{expected_name} R2 artifact IPC2 differs")
    native_cache = branch.get("native_cache_identity")
    if (not isinstance(native_cache, dict)
            or branch.get("cache_authority") != native_cache.get(
                "cache_authority"
            )):
        raise ValueError(f"{expected_name} lacks McPAT-native cache identity")
    validate_vector_native_cache(latency_vector, native_cache)
    return bips, controls, native_cache


def publish_paired_comparison(
    fixed: dict,
    clip3d: dict,
    output_dir: Path,
    *,
    r2_requested: bool,
) -> dict | None:
    """Publish paired metrics only when both branches have real R2 and HotSpot."""
    output_dir = Path(output_dir)
    json_path = output_dir / "paired_comparison.json"
    csv_path = output_dir / "paired_comparison.csv"
    if not r2_requested or any(
        not isinstance(branch, dict)
        or branch.get("measured_bips2_trans") is None
        for branch in (fixed, clip3d)
    ):
        json_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        return None

    try:
        fixed_bips, fixed_controls, fixed_native = _validated_branch(
            fixed, "fixed-bin"
        )
        clip_bips, clip_controls, clip_native = _validated_branch(
            clip3d, "clip3d"
        )
    except (OSError, ValueError):
        json_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise
    if fixed_controls != clip_controls:
        raise ValueError("paired branches use different optimization controls")
    if fixed_native != clip_native:
        raise ValueError("paired branches use different McPAT-native cache identities")
    lambda_wire = _finite_optional(
        fixed_controls.get("lambda_wire"), "paired lambda_wire"
    )
    if lambda_wire is None:
        raise ValueError("paired comparison lacks lambda_wire")
    if fixed_controls.get("wire_aggregation") not in (
        "mean", "maximum", "traffic-weighted"
    ):
        raise ValueError("paired comparison has invalid wire aggregation")
    if fixed_controls.get("wire_rounding") not in ("nearest", "ceil", "floor"):
        raise ValueError("paired comparison has invalid wire rounding")

    absolute = clip_bips - fixed_bips
    percent = absolute / fixed_bips * 100.0
    semantic_comparison = None
    primary_metric = "bips2_trans"
    primary_fixed = fixed_bips
    primary_clip = clip_bips
    if (fixed.get("measured_work_units_per_cycle") is not None
            or clip3d.get("measured_work_units_per_cycle") is not None):
        fixed_result = read_json(Path(fixed["artifacts"]["r2_result"]))
        clip_result = read_json(Path(clip3d["artifacts"]["r2_result"]))
        semantic_comparison = compare_semantic_results(
            fixed_result, clip_result,
            fixed["validated_f_sus_trans_hotspot_ghz"],
            clip3d["validated_f_sus_trans_hotspot_ghz"],
        )
        primary_metric = semantic_comparison["primary_metric"]
        primary_fixed = semantic_comparison["fixed_score"]
        primary_clip = semantic_comparison["clip3d_score"]
        for branch, result in ((fixed, fixed_result), (clip3d, clip_result)):
            if branch.get("measured_work_units_per_cycle") != result.get(
                    "work_units_per_cycle"):
                raise ValueError("semantic branch work rate differs from R2 evidence")
    primary_percent = (primary_clip / primary_fixed - 1.0) * 100.0
    report = {
        "schema_version": 1,
        "mode": "paired transient ROM real-HotSpot and gem5 R2 comparison",
        "non_formal": True,
        "paper_equivalent": False,
        "controls": fixed_controls,
        "native_cache_identity": fixed_native,
        "fixed_bin": fixed,
        "clip3d": clip3d,
        "bips2_trans_absolute_difference": absolute,
        "bips2_trans_improvement_percent": percent,
        "primary_performance_metric": primary_metric,
        "fixed_primary_score": primary_fixed,
        "clip3d_primary_score": primary_clip,
        "primary_improvement_percent": primary_percent,
        "semantic_comparison": semantic_comparison,
        "score_definition": (
            "fixed semantic work throughput * validated HotSpot frequency"
            if semantic_comparison is not None else
            "measured IPC2 * validated real-HotSpot transient sustainable frequency"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(json_path, report)

    stream = StringIO(newline="")
    fieldnames = (
        "fixed_bips2_trans",
        "clip3d_bips2_trans",
        "absolute_difference",
        "improvement_percent",
        "lambda_wire",
        "wire_aggregation",
        "wire_rounding",
    )
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        "fixed_bips2_trans": fixed_bips,
        "clip3d_bips2_trans": clip_bips,
        "absolute_difference": absolute,
        "improvement_percent": percent,
        "lambda_wire": lambda_wire,
        "wire_aggregation": fixed_controls["wire_aggregation"],
        "wire_rounding": fixed_controls["wire_rounding"],
    })
    atomic_write_bytes(csv_path, stream.getvalue().encode("utf-8"))
    return report
