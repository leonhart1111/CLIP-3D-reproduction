#!/usr/bin/env python3
"""Construct a validated, deterministic catalogue of R1 architecture points."""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from workflow.common import (
    parse_frequency_hz,
    parse_gem5_stats,
    read_json,
    sha256_file,
)
from workflow.r1_protocol import SEMANTIC_SCOPE, canonical_protocol


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


def _positive_integer(value, label: str, errors: list[str]) -> int | None:
    if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        errors.append(f"{label} must be a positive integer")
        return None
    return value


def _validate_semantic_evidence(record: dict, directory: Path, metadata: dict,
                                status: dict | None, stats_path: Path,
                                expected_unit: dict) -> None:
    """Bind a semantic catalogue entry to its binary, markers, and stats."""
    errors = record["errors"]
    warmup = _positive_integer(
        expected_unit.get("warmup"), "semantic warmup work units", errors
    )
    measure = _positive_integer(
        expected_unit.get("measure"), "semantic measure work units", errors
    )
    unit_type = expected_unit.get("type")
    if not isinstance(unit_type, str) or not unit_type.strip():
        errors.append("semantic work-unit type must be a non-empty string")
        unit_type = None
    expected = {
        "instruction_window_scope": SEMANTIC_SCOPE,
        "warmup_work_units": warmup,
        "measure_work_units": measure,
        "work_unit_type": unit_type,
    }
    for field, value in expected.items():
        if value is not None and metadata.get(field) != value:
            errors.append(
                f"metadata {field}={metadata.get(field)!r} does not match "
                f"semantic profile {value!r}"
            )

    try:
        protocol = canonical_protocol(metadata.get("r1_protocol", {}))
    except (TypeError, ValueError) as exception:
        errors.append(f"invalid semantic r1_protocol: {exception}")
    else:
        for field, value in expected.items():
            if value is not None and protocol.get(field) != value:
                errors.append(f"semantic r1_protocol differs for {field}")

    protocol_id = metadata.get("r1_protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        errors.append("semantic metadata lacks r1_protocol_id")
    if isinstance(status, dict) and status.get("r1_protocol_id") != protocol_id:
        errors.append("semantic status r1_protocol_id differs from metadata")

    digest = metadata.get("binary_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        errors.append("semantic metadata lacks a lowercase binary SHA-256")
    binary = metadata.get("binary")
    if not isinstance(binary, str) or not binary:
        errors.append("semantic metadata lacks workload binary path")
    elif not Path(binary).is_file():
        errors.append("semantic workload binary path does not exist")
    elif isinstance(digest, str):
        if sha256_file(Path(binary)) != digest:
            errors.append("semantic workload binary SHA-256 differs from metadata")

    evidence_path = directory / "roi_events.json"
    if not evidence_path.is_file():
        errors.append("missing roi_events.json")
        return
    if isinstance(status, dict):
        for field, path in (
                ("stats_sha256", stats_path),
                ("r1_metadata_sha256", directory / "r1_metadata.json"),
                ("roi_events_sha256", evidence_path)):
            if path.is_file() and status.get(field) != sha256_file(path):
                errors.append(f"semantic status {field} differs from live artifact")
    evidence = _read(evidence_path, errors, "roi_events.json")
    if not isinstance(evidence, dict):
        errors.append("roi_events.json must contain an object")
        return
    marker_expected = {
        "scope": SEMANTIC_SCOPE,
        "work_unit_type": unit_type,
        "warmup_work_units": warmup,
        "measure_work_units": measure,
        "marker_work_id": 1,
    }
    for field, value in marker_expected.items():
        if value is not None and evidence.get(field) != value:
            errors.append(f"semantic ROI evidence differs for {field}")
    events = evidence.get("events")
    valid_events = (
        isinstance(events, list)
        and len(events) == 2
        and all(isinstance(event, dict) for event in events)
        and [event.get("cause") for event in events] == ["workbegin", "workend"]
        and all(event.get("work_id") == 1 for event in events)
        and all(isinstance(event.get("tick"), int)
                and not isinstance(event.get("tick"), bool) for event in events)
        and events[1]["tick"] > events[0]["tick"]
    )
    if not valid_events:
        errors.append("semantic ROI marker sequence is incomplete or unordered")
        return
    completion_ticks = events[1]["tick"] - events[0]["tick"]
    if evidence.get("completion_ticks") != completion_ticks:
        errors.append("semantic ROI completion_ticks differs from marker ticks")
    if not stats_path.is_file():
        return
    try:
        stats = parse_gem5_stats(stats_path, include_nonfinite=True)
    except (OSError, UnicodeDecodeError):
        return
    sim_ticks = stats.get("simTicks")
    sim_freq = stats.get("simFreq")
    if (not isinstance(sim_ticks, (int, float))
            or not math.isfinite(sim_ticks)
            or not float(sim_ticks).is_integer()
            or int(sim_ticks) != completion_ticks):
        errors.append("semantic ROI marker ticks differ from stats simTicks")
    if (not isinstance(sim_freq, (int, float)) or not math.isfinite(sim_freq)
            or sim_freq <= 0 or not float(sim_freq).is_integer()):
        errors.append("semantic stats simFreq must be a positive integer")
        return
    try:
        cpu_frequency = parse_frequency_hz(metadata["cpu_clock"])
    except (KeyError, TypeError, ValueError) as exception:
        errors.append(f"invalid semantic CPU clock: {exception}")
        return
    completion_cycles = completion_ticks * cpu_frequency / int(sim_freq)
    record.update({
        "work_unit_type": unit_type,
        "warmup_work_units": warmup,
        "measure_work_units": measure,
        "completion_ticks": completion_ticks,
        "completion_cycles": completion_cycles,
        "work_units_per_cycle": (
            measure / completion_cycles
            if measure is not None and completion_cycles > 0 else None
        ),
    })


def _record_for(root: Path, key: ArchitectureKey, expected_scope: str,
                validate_counters: bool, profile_data: dict) -> dict:
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
    status = None
    if not status_path.is_file():
        errors.append("missing status.json")
    else:
        status = _read(status_path, errors, "status.json")
        if isinstance(status, dict):
            record["state"] = status.get("state")
            if record["state"] != "success":
                errors.append(f"status state is not success: {record['state']!r}")
        else:
            errors.append("status.json must contain an object")
    if not metadata_path.is_file():
        errors.append("missing r1_metadata.json")
        metadata = None
    else:
        metadata = _read(metadata_path, errors, "r1_metadata.json")
        if not isinstance(metadata, dict):
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
        if expected_scope == SEMANTIC_SCOPE:
            units = profile_data.get("semantic_work_units")
            expected_unit = units.get(key.workload) if isinstance(units, dict) else None
            if not isinstance(expected_unit, dict):
                errors.append(
                    f"semantic profile lacks work-unit declaration for {key.workload}"
                )
            else:
                _validate_semantic_evidence(
                    record, directory, metadata, status, stats_path, expected_unit
                )

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
    records = [
        _record_for(root, key, expected_scope, validate_counters, profile_data)
        for key in keys
    ]

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


def validate_canonical_plan(root: Path, catalogue: dict,
                            planned_path: Path | None = None) -> dict:
    """Bind one plan exactly to the catalogue's ordered canonical job list."""
    root = Path(root).resolve()
    planned_path = Path(planned_path or root / "planned_jobs.json").resolve()
    errors: list[str] = []
    if not planned_path.is_file():
        return {
            "valid": False,
            "path": str(planned_path),
            "job_count": 0,
            "errors": ["missing planned_jobs.json"],
        }
    try:
        plan = read_json(planned_path)
    except (OSError, ValueError) as error:
        return {
            "valid": False,
            "path": str(planned_path),
            "job_count": 0,
            "errors": [f"cannot read planned_jobs.json: {error}"],
        }
    if not isinstance(plan, dict):
        return {
            "valid": False,
            "path": str(planned_path),
            "job_count": 0,
            "errors": ["planned_jobs.json must contain an object"],
        }

    expected_profile = catalogue["profile"]
    if plan.get("profile") != expected_profile:
        errors.append(
            f"plan profile {plan.get('profile')!r} does not match {expected_profile!r}"
        )
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        errors.append("planned jobs must be a list")
        jobs = []
    job_count = plan.get("job_count")
    if not isinstance(job_count, int) or isinstance(job_count, bool) or job_count < 0:
        errors.append("plan job_count must be a non-boolean non-negative integer")
        reported_count = 0
    else:
        reported_count = job_count
        if job_count != len(jobs):
            errors.append(
                f"plan job_count {job_count} does not match jobs length {len(jobs)}"
            )

    expected_records = catalogue["canonical_records"]
    if len(jobs) != len(expected_records):
        errors.append(
            f"planned jobs length {len(jobs)} does not match canonical length "
            f"{len(expected_records)}"
        )
    seen_keys: set[tuple[object, object, object]] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            errors.append(f"job {index} must contain an object")
            continue
        actual_key = tuple(job.get(field) for field in (
            "workload", "l1d_size", "l2_size"
        ))
        if actual_key in seen_keys:
            errors.append(f"job {index} duplicates canonical key {actual_key!r}")
        seen_keys.add(actual_key)
        if index >= len(expected_records):
            errors.append(f"job {index} is an extra noncanonical job")
            continue
        record = expected_records[index]
        expected_key = tuple(record["key"][field] for field in (
            "workload", "l1d_size", "l2_size"
        ))
        if actual_key != expected_key:
            errors.append(
                f"job {index} does not match ordered canonical job: "
                f"{actual_key!r} != {expected_key!r}"
            )
        if job.get("profile") != expected_profile:
            errors.append(
                f"job {index} profile {job.get('profile')!r} does not match "
                f"{expected_profile!r}"
            )
        output_dir = job.get("output_dir")
        expected_output = Path(record["directory"]).resolve()
        if (not isinstance(output_dir, str) or not output_dir
                or Path(output_dir).resolve() != expected_output):
            errors.append(
                f"job {index} output_dir does not match canonical output "
                f"{expected_output}"
            )

    return {
        "valid": not errors,
        "path": str(planned_path),
        "job_count": reported_count,
        "jobs_length": len(jobs),
        "errors": errors,
    }


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
