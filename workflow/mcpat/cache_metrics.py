"""Strict parser for cache metrics emitted by McPAT's embedded CACTI-P."""

from __future__ import annotations

import hashlib
import json
import math
import re


MARKER = "CLIP_MCPAT_CACTI_P_V1"
_SCIENTIFIC = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+"
_RECORD_RE = re.compile(
    rf"^{MARKER} cache=(?P<cache>l1i|l1d|l2) core=(?P<core>-?[0-9]+) "
    rf"access_time_s=(?P<access_time_s>{_SCIENTIFIC}) "
    rf"cycle_time_s=(?P<cycle_time_s>{_SCIENTIFIC}) "
    rf"height_mm=(?P<height_mm>{_SCIENTIFIC}) "
    rf"width_mm=(?P<width_mm>{_SCIENTIFIC}) "
    r"mcpat_version=(?P<mcpat_version>1\.3) "
    r"model=(?P<model>embedded-cacti-p)$"
)
_IDENTITY_FIELDS = (
    "access_time_s", "cache", "core", "cycle_time_s", "height_mm",
    "mcpat_version", "model", "width_mm",
)


def _positive_float(value: str, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"embedded CACTI-P {field} must be finite and positive")
    return parsed


def _record_identity(record: dict) -> str:
    """Hash the sorted scientific record fields, never external file paths."""
    canonical = {
        key: (format(record[key], ".17e") if key in {
            "access_time_s", "cycle_time_s", "height_mm", "width_mm"
        } else record[key])
        for key in _IDENTITY_FIELDS
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_embedded_cacti_records(
        text: str, expected_core_count: int = 4) -> list[dict]:
    """Parse one exact, complete embedded-CACTI-P record set from McPAT text."""
    if isinstance(expected_core_count, bool) or expected_core_count <= 0:
        raise ValueError("expected core count must be a positive integer")

    records = []
    for line in text.splitlines():
        if not line.startswith(MARKER):
            continue
        match = _RECORD_RE.fullmatch(line)
        if not match:
            raise ValueError("malformed embedded CACTI-P record")
        raw = match.groupdict()
        core_index = int(raw["core"])
        cache = raw["cache"]
        if cache == "l2":
            if core_index != -1:
                raise ValueError("embedded CACTI-P shared L2 record must use core=-1")
            core = None
        else:
            core = core_index
        record = {
            "cache": cache,
            "core": core,
            "access_time_s": _positive_float(raw["access_time_s"], "access_time_s"),
            "cycle_time_s": _positive_float(raw["cycle_time_s"], "cycle_time_s"),
            "height_mm": _positive_float(raw["height_mm"], "height_mm"),
            "width_mm": _positive_float(raw["width_mm"], "width_mm"),
            "mcpat_version": raw["mcpat_version"],
            "model": raw["model"],
        }
        record["record_id"] = _record_identity(record)
        records.append(record)

    expected = {
        *(("l1i", core) for core in range(expected_core_count)),
        *(("l1d", core) for core in range(expected_core_count)),
        ("l2", None),
    }
    observed = [(record["cache"], record["core"]) for record in records]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate embedded CACTI-P cache record")
    if set(observed) != expected:
        missing = sorted(expected - set(observed), key=lambda item: (item[0], item[1] is None, item[1]))
        extra = sorted(set(observed) - expected, key=lambda item: (item[0], item[1] is None, item[1]))
        raise ValueError(
            f"embedded CACTI-P record set mismatch: missing={missing}, extra={extra}"
        )
    return sorted(
        records,
        key=lambda record: ({"l1i": 0, "l1d": 1, "l2": 2}[record["cache"]],
                            -1 if record["core"] is None else record["core"]),
    )
