"""Comparison contract for two fixed-work semantic gem5 measurements."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _positive(value: Any, label: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(float(value)) or float(value) <= 0):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def compare_semantic_results(fixed: Mapping[str, Any],
                             clip: Mapping[str, Any],
                             fixed_frequency_ghz: float | None = None,
                             clip_frequency_ghz: float | None = None) -> dict:
    """Use fixed useful-work throughput; gate IPC on an identical trace vector."""
    identity_fields = (
        "instruction_window_scope", "r1_protocol_id", "workload_binary_sha256",
        "work_unit_type", "measure_work_units",
    )
    for field in identity_fields:
        if fixed.get(field) != clip.get(field):
            raise ValueError(f"semantic pair differs for {field}")
    if fixed.get("instruction_window_scope") != "semantic-work":
        raise ValueError("semantic comparison requires semantic-work results")

    vectors = []
    for label, result in (("fixed", fixed), ("clip", clip)):
        vector = result.get("instruction_vector")
        if (not isinstance(vector, list) or not vector
                or not all(isinstance(value, int) and not isinstance(value, bool)
                           and value > 0 for value in vector)):
            raise ValueError(f"{label} instruction vector is malformed")
        vectors.append(vector)
    fixed_vector, clip_vector = vectors
    if len(fixed_vector) != len(clip_vector):
        raise ValueError("semantic pair instruction-vector lengths differ")
    deltas = [right - left for left, right in zip(fixed_vector, clip_vector)]
    relative = [delta / left for left, delta in zip(fixed_vector, deltas)]
    same_trace = fixed_vector == clip_vector

    fixed_rate = _positive(fixed.get("work_units_per_cycle"), "fixed rate")
    clip_rate = _positive(clip.get("work_units_per_cycle"), "clip rate")
    fixed_score = fixed_rate
    clip_score = clip_rate
    score_unit = "work_units_per_cycle"
    if fixed_frequency_ghz is not None or clip_frequency_ghz is not None:
        fixed_score *= _positive(fixed_frequency_ghz, "fixed frequency")
        clip_score *= _positive(clip_frequency_ghz, "clip frequency")
        score_unit = "work_units_per_ns"
    return {
        "primary_metric": score_unit,
        "fixed_score": fixed_score,
        "clip3d_score": clip_score,
        "improvement_percent": 100.0 * (clip_score / fixed_score - 1.0),
        "same_trace": same_trace,
        "ipc_comparison_allowed": same_trace,
        "fixed_instruction_vector": fixed_vector,
        "clip3d_instruction_vector": clip_vector,
        "instruction_delta": deltas,
        "instruction_relative_delta": relative,
        "max_abs_instruction_relative_delta": max(abs(value) for value in relative),
        "ipc_improvement_percent": (
            100.0 * (
                _positive(clip.get("ipc2"), "clip IPC")
                / _positive(fixed.get("ipc2"), "fixed IPC") - 1.0
            ) if same_trace else None
        ),
    }
