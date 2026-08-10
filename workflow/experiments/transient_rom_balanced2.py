#!/usr/bin/env python3
"""Run the exploratory MATMUL/STENCIL transient-ROM validation pair."""

from __future__ import annotations

import csv
import math
from numbers import Real
from pathlib import Path

from workflow.common import read_json, sha256_file
from workflow.run_lifting_pipeline import validate_config
from workflow.transient.run_transient_r1 import completed as transient_r1_completed


EXPECTED_POINTS = (
    {"workload": "matmul", "l1d_size": "64kB", "l2_size": "512kB"},
    {"workload": "stencil", "l1d_size": "64kB", "l2_size": "512kB"},
)
SAMPLE_INTERVAL_MS = 2.0


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
