#!/usr/bin/env python3
"""Construct a validated, deterministic catalogue of R1 architecture points."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from workflow.common import parse_gem5_stats, read_json, sha256_file


@dataclass(frozen=True, order=True)
class ArchitectureKey:
    workload: str
    l1d_size: str
    l2_size: str

    def relative_path(self) -> Path:
        return Path(self.workload) / f"l1d_{self.l1d_size}" / f"l2_{self.l2_size}"


def expected_keys(experiment: dict) -> list[ArchitectureKey]:
    return [ArchitectureKey(workload, l1d, l2)
            for workload, l1d, l2 in itertools.product(
                experiment["workloads"], experiment["l1d_sizes"],
                experiment["l2_sizes"])]


def _read(path: Path, errors: list[str], name: str):
    try:
        return read_json(path)
    except (OSError, ValueError) as exception:
        errors.append(f"cannot read {name}: {exception}")
        return None


def _validate_counters(record: dict, metadata: dict, stats_path: Path,
                       expected_scope: str) -> None:
    errors = record["errors"]
    try:
        stats = parse_gem5_stats(stats_path, include_nonfinite=True)
    except (OSError, UnicodeDecodeError) as exception:
        errors.append(f"cannot read stats.txt: {exception}")
        return

    target = metadata.get("measure_insts", metadata.get("measure_insts_cpu0"))
    per_core = []
    for core in range(4):
        instructions = stats.get(f"system.cpu{core}.commitStats0.numInsts")
        cycles = stats.get(f"system.cpu{core}.numCycles")
        per_core.append({"core": core, "instructions": instructions, "cycles": cycles})
        if (not isinstance(instructions, (int, float))
                or not math.isfinite(instructions) or instructions <= 0):
            errors.append(f"CPU{core} instructions must be positive")
        if (not isinstance(cycles, (int, float))
                or not math.isfinite(cycles) or cycles <= 0):
            errors.append(f"CPU{core} cycles must be positive")
        if expected_scope == "all-cores" and target is not None:
            try:
                below_target = instructions is None or instructions < int(target)
            except (TypeError, ValueError):
                errors.append("invalid measurement target")
            else:
                if below_target:
                    errors.append(f"CPU{core} instructions below measurement target")
    record["per_core"] = per_core


def _record_for(root: Path, key: ArchitectureKey, expected_scope: str,
                validate_counters: bool) -> dict:
    directory = root / key.relative_path()
    record = {
        "key": asdict(key),
        "directory": str(directory.resolve()),
        "relative_directory": key.relative_path().as_posix(),
        "state": None,
        "valid": False,
        "errors": [],
    }
    errors = record["errors"]
    status_path = directory / "status.json"
    metadata_path = directory / "r1_metadata.json"
    stats_path = directory / "stats.txt"
    if not status_path.is_file():
        errors.append("missing status.json")
    else:
        status = _read(status_path, errors, "status.json")
        if isinstance(status, dict):
            record["state"] = status.get("state")
            if record["state"] != "success":
                errors.append(f"status state is not success: {record['state']!r}")
        elif status is not None:
            errors.append("status.json must contain an object")
    if not metadata_path.is_file():
        errors.append("missing r1_metadata.json")
        metadata = None
    else:
        metadata = _read(metadata_path, errors, "r1_metadata.json")
        if metadata is not None and not isinstance(metadata, dict):
            errors.append("r1_metadata.json must contain an object")
            metadata = None
    if not stats_path.is_file():
        errors.append("missing stats.txt")

    if metadata is not None:
        for field, expected in (("workload", key.workload),
                                ("l1d_size", key.l1d_size),
                                ("l2_size", key.l2_size)):
            if metadata.get(field) != expected:
                errors.append(
                    f"metadata {field}={metadata.get(field)!r} does not match path {expected!r}"
                )
        scope = metadata.get("instruction_window_scope", "cpu0")
        record["instruction_window_scope"] = scope
        if scope != expected_scope:
            errors.append(
                f"instruction_window_scope={scope!r} does not match profile {expected_scope!r}"
            )
        if validate_counters and stats_path.is_file():
            _validate_counters(record, metadata, stats_path, expected_scope)

    record["valid"] = not errors
    return record


def build_catalogue(root: Path, experiment_path: Path, profile: str = "paper",
                    validate_counters: bool = True) -> dict:
    """Validate only the configured R1 grid and report other status directories."""
    root = Path(root).resolve()
    experiment_path = Path(experiment_path).resolve()
    experiment = read_json(experiment_path)
    try:
        profile_data = experiment["profiles"][profile]
    except KeyError as exception:
        raise ValueError(f"unknown R1 profile: {profile}") from exception
    expected_scope = profile_data.get("instruction_window_scope", "cpu0")
    keys = expected_keys(experiment)
    canonical_paths = {key.relative_path().as_posix() for key in keys}
    records = [_record_for(root, key, expected_scope, validate_counters) for key in keys]

    excluded = []
    if root.is_dir():
        for status_path in sorted(root.rglob("status.json")):
            relative_directory = status_path.parent.relative_to(root).as_posix()
            if relative_directory not in canonical_paths:
                excluded.append({
                    "directory": str(status_path.parent.resolve()),
                    "relative_directory": relative_directory,
                    "reason": "not an expected canonical directory",
                })
    unique_key_count = len(set(keys))
    configuration_errors = []
    if unique_key_count != len(keys):
        configuration_errors.append("experiment grid contains duplicate architecture keys")
    return {
        "root": str(root),
        "experiment": str(experiment_path),
        "profile": profile,
        "instruction_window_scope": expected_scope,
        "expected_count": len(keys),
        "unique_key_count": unique_key_count,
        "valid_count": sum(record["valid"] for record in records),
        "complete": not configuration_errors and len(records) == len(keys)
                    and all(record["valid"] for record in records),
        "errors": configuration_errors,
        "canonical_records": records,
        "excluded_noncanonical": excluded,
    }


def canonical_directories(catalogue: dict) -> list[Path]:
    """Return valid canonical directories in the experiment's deterministic order."""
    return [Path(record["directory"]) for record in catalogue["canonical_records"]
            if record["valid"]]


def snapshot_canonical_artifacts(catalogue: dict) -> dict[str, str]:
    """Hash the required provenance artifacts of each valid canonical point."""
    snapshot = {}
    for record in catalogue["canonical_records"]:
        if not record["valid"]:
            continue
        directory = Path(record["directory"])
        relative_directory = record["relative_directory"]
        for name in ("status.json", "r1_metadata.json", "stats.txt"):
            snapshot[f"{relative_directory}/{name}"] = sha256_file(directory / name)
    return snapshot
