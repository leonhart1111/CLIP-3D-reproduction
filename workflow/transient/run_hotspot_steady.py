#!/usr/bin/env python3
"""Run a case-matched steady-only HotSpot initialization solve."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import time

from workflow.common import read_json, sha256_file, write_json
from workflow.transient.run_hotspot_transient import (
    DEFAULT_HOTSPOT,
    read_named_temperatures,
)


_INPUT_FILES = (
    "hotspot.config",
    "power_transient.ptrace",
    "stack.lcf",
    "materials.txt",
)
_OUTPUT_FILES = (
    "initialization.steady.txt",
    "initialization.grid.steady.txt",
)


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _sha256(path: Path) -> str:
    return "sha256:" + sha256_file(path)


def canonical_hotspot_steady_command(hotspot: Path | str) -> list[str]:
    """Return the only command accepted for average-power initialization."""
    return [
        str(hotspot),
        "-c", "hotspot.config",
        "-p", "power_transient.ptrace",
        "-grid_layer_file", "stack.lcf",
        "-materials_file", "materials.txt",
        "-model_type", "grid",
        "-detailed_3D", "on",
        "-steady_file", "initialization.steady.txt",
        "-grid_steady_file", "initialization.grid.steady.txt",
    ]


def validate_hotspot_steady_record(
    case_dir: Path,
    hotspot: Path = DEFAULT_HOTSPOT,
    expected_record: dict | None = None,
) -> dict:
    """Rehash and validate one persisted matched steady initialization."""
    raw_case = Path(case_dir)
    if raw_case.is_symlink() or not raw_case.is_dir():
        raise FileNotFoundError(
            f"steady initialization case must be a regular directory: {raw_case}"
        )
    case_dir = raw_case.resolve()
    hotspot = _regular_file(Path(hotspot), "HotSpot binary")
    record_path = _regular_file(
        case_dir / "steady_initialization.json",
        "steady initialization record",
    )
    record = read_json(record_path)
    if not isinstance(record, dict):
        raise ValueError("steady initialization record must be a dictionary")
    if expected_record is not None and record != expected_record:
        raise ValueError("steady initialization returned and persisted records differ")
    command = record.get("command")
    if (
        command != canonical_hotspot_steady_command(hotspot)
        or record.get("return_code") != 0
    ):
        raise ValueError("steady initialization canonical command evidence differs")
    inputs = {
        name: _regular_file(
            case_dir / name, f"steady initialization input {name}"
        )
        for name in _INPUT_FILES
    }
    expected_inputs = {
        **{name: _sha256(path) for name, path in inputs.items()},
        "hotspot_binary": _sha256(hotspot),
    }
    outputs = {
        name: _regular_file(
            case_dir / name, f"steady initialization output {name}"
        )
        for name in _OUTPUT_FILES
    }
    expected_outputs = {name: _sha256(path) for name, path in outputs.items()}
    if record.get("input_sha256") != expected_inputs:
        raise ValueError("steady initialization input hashes differ")
    if record.get("output_sha256") != expected_outputs:
        raise ValueError("steady initialization output hashes differ")
    return record


def run_hotspot_steady(case_dir: Path,
                       hotspot: Path = DEFAULT_HOTSPOT) -> dict:
    """Compute average-power steady temperatures for one materialized case.

    Omitting HotSpot's ``-o`` option is intentional: the tool reads every
    power row, averages them, and solves only the steady thermal system.
    """
    raw_case = Path(case_dir)
    if raw_case.is_symlink() or not raw_case.is_dir():
        raise FileNotFoundError(
            f"steady initialization case must be a regular directory: {raw_case}"
        )
    case_dir = raw_case.resolve()
    hotspot = _regular_file(Path(hotspot), "HotSpot binary")
    inputs = {
        name: _regular_file(case_dir / name, f"steady initialization input {name}")
        for name in _INPUT_FILES
    }
    outputs = {name: case_dir / name for name in _OUTPUT_FILES}
    stale = [str(path) for path in outputs.values() if path.exists() or path.is_symlink()]
    if stale:
        raise FileExistsError(
            "steady initialization outputs already exist: " + ", ".join(stale)
        )

    command = canonical_hotspot_steady_command(hotspot)
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    (case_dir / "hotspot_steady.log").write_text(
        process.stdout or "", encoding="utf-8"
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"steady initialization HotSpot failed (rc={process.returncode}); "
            f"see {case_dir / 'hotspot_steady.log'}"
        )
    for name, path in outputs.items():
        _regular_file(path, f"steady initialization output {name}")
        if path.stat().st_size == 0:
            raise ValueError(f"steady initialization output is empty: {path}")

    block_temperatures = read_named_temperatures(
        outputs["initialization.steady.txt"]
    )
    grid_temperatures = read_named_temperatures(
        outputs["initialization.grid.steady.txt"]
    )
    if any(
        not math.isfinite(value)
        for _, value in [*block_temperatures, *grid_temperatures]
    ):
        raise ValueError("steady initialization contains non-finite temperatures")
    peak_name, peak_k = max(block_temperatures, key=lambda item: item[1])
    input_sha256 = {
        **{name: _sha256(path) for name, path in inputs.items()},
        "hotspot_binary": _sha256(hotspot),
    }
    output_sha256 = {name: _sha256(path) for name, path in outputs.items()}
    result = {
        "schema_version": 1,
        "mode": "average-power steady initialization",
        "non_formal": True,
        "paper_equivalent": False,
        "command": command,
        "return_code": process.returncode,
        "elapsed_seconds": elapsed,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "artifacts": {
            **{name: str(path.resolve()) for name, path in outputs.items()},
            "log": str((case_dir / "hotspot_steady.log").resolve()),
        },
        "initial_peak": {
            "peak_unit": peak_name,
            "tmax_k": peak_k,
            "tmax_c": peak_k - 273.15,
        },
    }
    write_json(case_dir / "steady_initialization.json", result)
    return validate_hotspot_steady_record(case_dir, hotspot, result)
