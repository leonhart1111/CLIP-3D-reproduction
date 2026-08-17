"""Derive transient sampling from the measured R1 ROI duration.

The corrected transient protocol samples the measured region with
``min(max_interval_ms, roi_duration_ms / target_window_count)`` so that the
interval is derived from the actual gem5 run instead of a hard-coded 10 ms.
An explicit override keeps the same math but receives a distinct policy ID.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from workflow.common import parse_gem5_stats_text, read_json, sha256_file


DEFAULT_MAX_INTERVAL_MS = 0.5
DEFAULT_TARGET_WINDOW_COUNT = 50


def measured_roi_duration(r1_dir: Path) -> dict:
    """Return the measured post-reset ROI duration with cross-checks."""
    r1_dir = Path(r1_dir).resolve()
    metadata_path = r1_dir / "r1_metadata.json"
    stats_path = r1_dir / "stats.txt"
    if not metadata_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(
            f"R1 directory lacks r1_metadata.json/stats.txt: {r1_dir}"
        )
    metadata = read_json(metadata_path)
    stats = parse_gem5_stats_text(stats_path.read_text(encoding="utf-8"))
    missing = [name for name in ("simSeconds", "simTicks", "simFreq")
               if name not in stats]
    if missing:
        raise ValueError(
            f"R1 stats lack measured-region fields: {', '.join(missing)}"
        )
    sim_seconds = float(stats["simSeconds"])
    sim_ticks = float(stats["simTicks"])
    sim_freq = float(stats["simFreq"])
    for name, value in (
        ("simSeconds", sim_seconds), ("simTicks", sim_ticks),
        ("simFreq", sim_freq),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"R1 measured-region {name} must be finite positive")
    cross_checked = sim_ticks / sim_freq
    # gem5 prints simSeconds with limited significant digits, so the printed
    # value can differ from simTicks/simFreq by ~1e-5 relative; 0.1% tolerance
    # still catches a genuinely mismatched region while absorbing print noise.
    if not math.isclose(
        cross_checked, sim_seconds, rel_tol=1e-3, abs_tol=1e-12
    ):
        raise ValueError(
            "R1 measured-region simSeconds disagrees with simTicks/simFreq: "
            f"{sim_seconds!r} vs {cross_checked!r}"
        )
    protocol_id = metadata.get("r1_protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("R1 metadata lacks an explicit r1_protocol_id")
    return {
        "schema_version": 1,
        "roi_duration_s": sim_seconds,
        "roi_duration_ms": sim_seconds * 1000.0,
        "sim_ticks": sim_ticks,
        "sim_freq_hz": sim_freq,
        "sim_seconds_recorded": sim_seconds,
        "r1_protocol_id": protocol_id,
        "instruction_window_scope": metadata.get(
            "instruction_window_scope", "cpu0"
        ),
        "warmup_insts": metadata.get("warmup_insts"),
        "measure_insts": metadata.get("measure_insts"),
        "stats_sha256": sha256_file(stats_path),
    }


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return converted


def resolve_sampling_policy(
    r1_dir: Path,
    override_ms: float | None = None,
    *,
    max_interval_ms: float = DEFAULT_MAX_INTERVAL_MS,
    target_window_count: int = DEFAULT_TARGET_WINDOW_COUNT,
) -> dict:
    """Resolve and identity the sampling policy for one canonical R1 point."""
    max_interval_ms = _positive(max_interval_ms, "max_interval_ms")
    if (
        isinstance(target_window_count, bool)
        or not isinstance(target_window_count, int)
        or target_window_count <= 0
    ):
        raise ValueError("target_window_count must be a positive integer")
    if override_ms is None:
        mode = "roi-derived"
        roi = measured_roi_duration(r1_dir)
        requested_ms = min(
            max_interval_ms, roi["roi_duration_ms"] / target_window_count
        )
    else:
        mode = "explicit-override"
        requested_ms = _positive(override_ms, "override_ms")
        try:
            roi = measured_roi_duration(r1_dir)
            source_roi_unavailable = False
        except (FileNotFoundError, ValueError) as error:
            roi = {
                "r1_protocol_id": "legacy-unidentified",
                "stats_sha256": None,
                "roi_duration_s": None,
                "roi_duration_ms": None,
            }
            source_roi_unavailable = str(error)
    policy = {
        "schema_version": 1,
        "mode": mode,
        "source_r1": str(Path(r1_dir).resolve()),
        "r1_protocol_id": roi["r1_protocol_id"],
        "stats_sha256": roi["stats_sha256"],
        "roi_duration_s": roi["roi_duration_s"],
        "roi_duration_ms": roi["roi_duration_ms"],
        "formula": (
            "min(max_interval_ms, roi_duration_ms / target_window_count)"
        ),
        "max_interval_ms": max_interval_ms,
        "target_window_count": target_window_count,
        "requested_interval_ms": requested_ms,
    }
    if mode == "explicit-override" and source_roi_unavailable:
        policy["source_roi_unavailable"] = source_roi_unavailable
    encoded = json.dumps(
        policy, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    policy["policy_id"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return policy
