#!/usr/bin/env python3
"""Shared, dependency-free helpers for the CLIP-3D reproduction workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from numbers import Real
from pathlib import Path
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def format_temperature_c(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("temperature must be finite")
    return f"{number:.6f}"


def format_temperature_csv_row(record: dict) -> dict:
    return {
        key: format_temperature_c(value)
        if key.endswith("_c") and isinstance(value, Real) and not isinstance(value, bool)
        else value
        for key, value in record.items()
    }


def parse_frequency_hz(value: str) -> float:
    """Parse the explicit frequency forms used by the gem5 workflow."""
    if not isinstance(value, str):
        raise ValueError("frequency must be a string")
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(Hz|kHz|MHz|GHz|THz)\s*", value
    )
    if match is None:
        raise ValueError(f"unsupported frequency: {value!r}")
    scale = {
        "Hz": 1.0,
        "kHz": 1.0e3,
        "MHz": 1.0e6,
        "GHz": 1.0e9,
        "THz": 1.0e12,
    }[match.group(2)]
    return float(match.group(1)) * scale


def read_json(path: Path | str) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def instruction_window_scope(metadata: dict) -> object:
    """Return an explicit scope, or the historical cpu0 default when absent."""
    if "instruction_window_scope" in metadata:
        return metadata["instruction_window_scope"]
    return "cpu0"


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Publish complete bytes by unique same-directory temp and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = 0o644
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    descriptor_open = True
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor_open = False
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_json(path: Path | str, value: Any) -> None:
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(path, payload)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_size_bytes(text: str | int | float) -> int:
    if isinstance(text, (int, float)):
        return int(text)
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b)?\s*", text, re.I)
    if not match:
        raise ValueError(f"cannot parse byte size: {text!r}")
    value = float(match.group(1))
    suffix = (match.group(2) or "B").lower()
    powers = {"b": 0, "kb": 1, "kib": 1, "mb": 2, "mib": 2,
              "gb": 3, "gib": 3, "tb": 4, "tib": 4}
    return int(value * (1024 ** powers[suffix]))


def parse_frequency_ghz(text: str | int | float) -> float:
    if isinstance(text, (int, float)):
        return float(text)
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([gm]?hz)\s*", text, re.I)
    if not match:
        raise ValueError(f"cannot parse frequency: {text!r}")
    value = float(match.group(1))
    return value if match.group(2).lower() == "ghz" else value / 1000.0


def parse_gem5_stats_text(text: str,
                          include_nonfinite: bool = False) -> dict[str, float]:
    """Parse the last gem5 statistics section from one captured text snapshot."""
    sections = text.split(
        "---------- Begin Simulation Statistics ----------"
    )
    text = sections[-1]
    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("-") or line[0].isspace():
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(value) or include_nonfinite:
            result[fields[0]] = value
    return result


def parse_gem5_stats(path: Path | str,
                     include_nonfinite: bool = False) -> dict[str, float]:
    """Read the last statistics section into a flat name -> number map.

    Normal workflow consumers retain the historical finite-only behavior.
    Validation code may request non-finite values so malformed counters are
    distinguished from counters that are absent.
    """
    return parse_gem5_stats_text(
        Path(path).read_text(encoding="utf-8"), include_nonfinite
    )


def stat(stats: dict[str, float], name: str, default: float = 0.0) -> float:
    return float(stats.get(name, default))


def aggregate_ipc(stats: dict[str, float], cores: int = 4) -> float:
    instructions = sum(
        stat(stats, f"system.cpu{i}.commitStats0.numInsts") for i in range(cores)
    )
    cycles = max(stat(stats, f"system.cpu{i}.numCycles") for i in range(cores))
    if cycles <= 0:
        raise ValueError("gem5 statistics contain no positive CPU cycle count")
    return instructions / cycles


def positive(value: float, floor: float = 0.0) -> float:
    return max(float(value), floor)
