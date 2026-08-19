#!/usr/bin/env python3
"""Run a resumable gem5 R2 and validate all measured cores."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import time
from pathlib import Path

from workflow.common import (
    PROJECT_ROOT,
    aggregate_ipc,
    parse_frequency_hz,
    parse_gem5_stats,
    read_json,
    sha256_file,
    write_json,
)
from workflow.r1_protocol import SEMANTIC_SCOPE, canonical_protocol


DEFAULT_GEM5 = PROJECT_ROOT / "tools/src/gem5/build/X86/gem5.opt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/gem5/clip_r1.py"
GEM5_OVERRIDE_KEYS = (
    "l1i_tag_latency", "l1i_data_latency", "l1i_response_latency",
    "l1d_tag_latency", "l1d_data_latency", "l1d_response_latency",
    "l2_tag_latency", "l2_data_latency", "l2_response_latency",
    "xbar_frontend_latency", "xbar_forward_latency",
    "xbar_response_latency", "xbar_snoop_response_latency",
)
MAX_GEM5_LATENCY = (1 << 64) - 1


def validate_gem5_overrides(overrides: dict) -> dict[str, int]:
    """Validate the exact positive uint64 domain consumed by gem5 Cycles."""
    if not isinstance(overrides, dict) or set(overrides) != set(GEM5_OVERRIDE_KEYS):
        raise ValueError(
            "gem5_overrides must contain the complete canonical override key set"
        )
    for key in GEM5_OVERRIDE_KEYS:
        value = overrides[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"gem5_overrides {key} must be a non-boolean integer"
            )
        if value < 1 or value > MAX_GEM5_LATENCY:
            raise ValueError(
                f"gem5_overrides {key} must be in [1, {MAX_GEM5_LATENCY}]"
            )
    return overrides


def canonical_gem5_args(overrides: dict) -> list[str]:
    """Render the complete latency override mapping in one immutable order."""
    overrides = validate_gem5_overrides(overrides)
    arguments: list[str] = []
    for key in GEM5_OVERRIDE_KEYS:
        arguments.extend((f"--{key.replace('_', '-')}", str(overrides[key])))
    return arguments


def validate_latency_vector(vector: dict) -> dict[str, int]:
    """Validate overrides and their exact ordered command-line rendering."""
    if not isinstance(vector, dict):
        raise ValueError("latency vector must contain an object")
    overrides = validate_gem5_overrides(vector.get("gem5_overrides"))
    expected_args = canonical_gem5_args(overrides)
    if vector.get("gem5_args") != expected_args:
        raise ValueError(
            "latency vector gem5_args are not the canonical rendering of "
            "gem5_overrides"
        )
    return overrides


def strict_latency_vectors_equal(left: dict, right: dict) -> bool:
    """Compare only vectors that both satisfy the exact override contract."""
    left_overrides = validate_latency_vector(left)
    right_overrides = validate_latency_vector(right)
    return all(left_overrides[key] == right_overrides[key]
               for key in GEM5_OVERRIDE_KEYS)


def _semantic_contract(metadata: dict) -> None:
    """Require the live fixed-work identity consumed by a semantic R2."""
    for field in ("warmup_work_units", "measure_work_units"):
        value = metadata.get(field)
        if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"semantic R2 metadata requires positive {field}")
    unit_type = metadata.get("work_unit_type")
    if not isinstance(unit_type, str) or not unit_type.strip():
        raise ValueError("semantic R2 metadata requires work_unit_type")
    binary = metadata.get("binary")
    if not isinstance(binary, str) or not binary:
        raise ValueError("semantic R2 metadata requires binary")
    binary_path = Path(binary)
    if not binary_path.is_file():
        raise ValueError("semantic R2 workload binary does not exist")
    binary_sha256 = metadata.get("binary_sha256")
    if (not isinstance(binary_sha256, str) or len(binary_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in binary_sha256)):
        raise ValueError("semantic R2 metadata requires binary_sha256")
    if sha256_file(binary_path) != binary_sha256:
        raise ValueError("semantic R2 workload binary differs from binary_sha256")
    try:
        protocol = canonical_protocol(metadata.get("r1_protocol", {}))
    except (TypeError, ValueError) as error:
        raise ValueError(f"semantic R2 metadata has invalid r1_protocol: {error}") from error
    for field in (
            "instruction_window_scope", "warmup_work_units",
            "measure_work_units", "work_unit_type"):
        if protocol.get(field) != metadata.get(field):
            raise ValueError(f"semantic R2 r1_protocol differs for {field}")
    identity = metadata.get("r1_protocol_id")
    if not isinstance(identity, str) or not identity:
        raise ValueError("semantic R2 metadata requires r1_protocol_id")


def _command_tail(metadata: dict, vector: dict,
                  scope: str | None = None) -> list[str]:
    scope = scope or metadata.get("instruction_window_scope", "cpu0")
    overrides = validate_latency_vector(vector)
    gem5_args = canonical_gem5_args(overrides)
    tail = [
        "--stage", "R2", "--workload", metadata["workload"],
    ]
    if scope == SEMANTIC_SCOPE:
        _semantic_contract(metadata)
        tail.extend(("--binary", metadata["binary"]))
    tail.extend((
        "--l1i-size", metadata["l1i_size"], "--l1d-size", metadata["l1d_size"],
        "--l2-size", metadata["l2_size"],
    ))
    if scope == SEMANTIC_SCOPE:
        tail.extend((
            "--warmup-work-units", str(metadata["warmup_work_units"]),
            "--measure-work-units", str(metadata["measure_work_units"]),
            "--work-unit-type", metadata["work_unit_type"],
        ))
    else:
        tail.extend((
            "--warmup-insts", str(metadata["warmup_insts_cpu0"]),
            "--measure-insts", str(metadata["measure_insts_cpu0"]),
        ))
    tail.extend(("--instruction-window-scope", scope, *gem5_args))
    if scope == SEMANTIC_SCOPE:
        tail.extend(("--workload-binary-sha256", metadata["binary_sha256"]))
        protocol = metadata.get("r1_protocol")
        if not isinstance(protocol, dict):
            raise ValueError("semantic R2 metadata requires r1_protocol")
        if not protocol.get("family") or not protocol.get("profile"):
            raise ValueError("semantic R2 protocol requires family and profile")
        tail.extend((
            "--r1-protocol-family", protocol["family"],
            "--r1-profile", protocol["profile"],
        ))
        protocol_identity = metadata.get("r1_protocol_id")
        if not isinstance(protocol_identity, str) or not protocol_identity:
            raise ValueError("semantic R2 metadata requires r1_protocol_id")
        tail.extend(("--r1-protocol-id", protocol_identity))
    options = metadata.get("command", [])[1:]
    if options:
        tail.extend(("--options", " ".join(options)))
    if metadata.get("stdin"):
        tail.extend(("--stdin", metadata["stdin"]))
    return tail


def _provenance(r1_dir: Path, latency_path: Path,
                scope: str | None = None) -> dict:
    metadata = read_json(r1_dir / "r1_metadata.json")
    effective_scope = scope or metadata.get("instruction_window_scope", "cpu0")
    provenance = {
        "r1_directory": str(r1_dir.resolve()),
        "r1_metadata_sha256": sha256_file(r1_dir / "r1_metadata.json"),
        "r1_stats_sha256": sha256_file(r1_dir / "stats.txt"),
        "latency_vector": str(latency_path.resolve()),
        "latency_sha256": sha256_file(latency_path),
        "instruction_window_scope": effective_scope,
    }
    if effective_scope == SEMANTIC_SCOPE:
        _semantic_contract(metadata)
        roi_path = r1_dir / "roi_events.json"
        provenance.update({
            "r1_protocol_id": metadata["r1_protocol_id"],
            "workload_binary_sha256": metadata["binary_sha256"],
            "warmup_work_units": metadata["warmup_work_units"],
            "measure_work_units": metadata["measure_work_units"],
            "work_unit_type": metadata["work_unit_type"],
            "r1_roi_events_sha256": sha256_file(roi_path),
        })
    else:
        provenance.update({
            "warmup_insts_cpu0": metadata["warmup_insts_cpu0"],
            "measure_insts_cpu0": metadata["measure_insts_cpu0"],
        })
    return provenance


def _validated_measurement(stats: dict[str, float], metadata: dict,
                           scope: str | None = None) -> tuple[list[dict], float]:
    """Validate finite integral counters once for both fresh and cached R2 paths."""
    cores = int(metadata["num_cores"])
    scope = scope or metadata.get("instruction_window_scope", "cpu0")
    minimum = int(metadata["measure_insts_cpu0"]) if scope == "all-cores" else 1
    per_core = []
    for core in range(cores):
        values = []
        for label, name in (
                ("instructions", f"system.cpu{core}.commitStats0.numInsts"),
                ("cycles", f"system.cpu{core}.numCycles")):
            value = stats.get(name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0
                    or not float(value).is_integer()):
                raise ValueError(f"R2 CPU{core} {label} must be a finite positive integer")
            values.append(int(value))
        instructions, cycles = values
        if instructions < minimum:
            raise ValueError(
                f"R2 CPU{core} instructions are below measurement minimum {minimum}"
            )
        per_core.append({
            "core": core,
            "instructions": instructions,
            "cycles": cycles,
            "ipc": instructions / cycles,
        })
    return per_core, aggregate_ipc(stats, cores)


def _semantic_metrics(stats: dict[str, float], metadata: dict,
                      output_dir: Path) -> dict:
    """Validate marker evidence and derive global fixed-work completion rate."""
    evidence_path = output_dir / "roi_events.json"
    evidence = read_json(evidence_path)
    if not isinstance(evidence, dict):
        raise ValueError("R2 semantic marker evidence must contain an object")
    for field in ("warmup_work_units", "measure_work_units"):
        value = metadata.get(field)
        if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"R2 semantic metadata {field} must be positive")
    if (not isinstance(metadata.get("work_unit_type"), str)
            or not metadata["work_unit_type"].strip()):
        raise ValueError("R2 semantic metadata work_unit_type is invalid")
    expected = {
        "scope": SEMANTIC_SCOPE,
        "work_unit_type": metadata["work_unit_type"],
        "warmup_work_units": metadata["warmup_work_units"],
        "measure_work_units": metadata["measure_work_units"],
        "marker_work_id": 1,
    }
    for field, wanted in expected.items():
        if evidence.get(field) != wanted:
            raise ValueError(f"R2 semantic marker evidence differs for {field}")
    events = evidence.get("events")
    if (not isinstance(events, list) or len(events) != 2
            or not all(isinstance(event, dict) for event in events)
            or [event.get("cause") for event in events]
            != ["workbegin", "workend"]
            or not all(event.get("work_id") == 1 for event in events)
            or not all(isinstance(event.get("tick"), int)
                       and not isinstance(event.get("tick"), bool)
                       for event in events)
            or events[1]["tick"] <= events[0]["tick"]):
        raise ValueError("R2 semantic marker sequence is incomplete or unordered")
    completion_ticks = events[1]["tick"] - events[0]["tick"]
    if evidence.get("completion_ticks") != completion_ticks:
        raise ValueError("R2 semantic completion_ticks differs from marker ticks")
    for name in ("simTicks", "simFreq"):
        value = stats.get(name)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0
                or not float(value).is_integer()):
            raise ValueError(f"R2 {name} must be a finite positive integer")
    if completion_ticks != int(stats["simTicks"]):
        raise ValueError("R2 semantic marker ticks differ from stats simTicks")
    completion_cycles = (
        completion_ticks * parse_frequency_hz(metadata["cpu_clock"])
        / int(stats["simFreq"])
    )
    measure_units = int(metadata["measure_work_units"])
    return {
        "primary_performance_metric": "work_units_per_cycle",
        "work_unit_type": metadata["work_unit_type"],
        "measure_work_units": measure_units,
        "completion_ticks": completion_ticks,
        "completion_cycles": completion_cycles,
        "work_units_per_cycle": measure_units / completion_cycles,
        "roi_events": str(evidence_path.resolve()),
        "roi_events_sha256": sha256_file(evidence_path),
    }


def _stats_snapshot(path: Path) -> tuple[dict[str, float], str]:
    """Parse and hash the same captured gem5 statistics bytes."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    sections = data.decode("utf-8").split(
        "---------- Begin Simulation Statistics ----------"
    )
    stats: dict[str, float] = {}
    for line in sections[-1].splitlines():
        if not line or line.startswith("-") or line[0].isspace():
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            stats[fields[0]] = float(fields[1])
        except ValueError:
            continue
    return stats, digest


def _validate_measurement(result: dict, status: dict, metadata: dict,
                          output_dir: Path, reasons: list[str],
                          scope: str | None = None) -> None:
    """Bind recorded IPC/per-core values to the measured R2 stats file."""
    stats_path = output_dir / "stats.txt"
    if not stats_path.is_file():
        reasons.append(f"missing R2 stats: {stats_path}")
        return
    try:
        stats, stats_sha256 = _stats_snapshot(stats_path)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        reasons.append(f"cannot validate R2 stats: {error}")
        return
    for label, record in (("result", result), ("status", status)):
        value = record.get("stats")
        if (not isinstance(value, str)
                or Path(value).resolve() != stats_path.resolve()):
            reasons.append(f"R2 {label} stats does not identify the cached stats")
        if record.get("stats_sha256") != stats_sha256:
            reasons.append(f"R2 {label} stats_sha256 does not match cached stats")

    try:
        effective_scope = scope or metadata.get("instruction_window_scope", "cpu0")
        measured_per_core, measured_ipc = _validated_measurement(
            stats, metadata, effective_scope
        )
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"cannot validate R2 stats: {error}")
        return
    if sha256_file(stats_path) != stats_sha256:
        reasons.append("R2 stats changed during validation")
    per_core = result.get("per_core")
    if not isinstance(per_core, list) or len(per_core) != len(measured_per_core):
        reasons.append("R2 result per_core does not cover every canonical core")
        per_core = []
    for measured in measured_per_core:
        core = measured["core"]
        if core >= len(per_core) or not isinstance(per_core[core], dict):
            continue
        recorded = per_core[core]
        if (recorded.get("core") != core
                or recorded.get("instructions") != measured["instructions"]
                or recorded.get("cycles") != measured["cycles"]):
            reasons.append(f"R2 result CPU{core} counters differ from cached stats")
        recorded_ipc = recorded.get("ipc")
        if (not isinstance(recorded_ipc, (int, float))
                or isinstance(recorded_ipc, bool)
                or not math.isclose(
                    float(recorded_ipc), measured["ipc"],
                    rel_tol=1e-12, abs_tol=0.0,
                )):
            reasons.append(f"R2 result CPU{core} IPC differs from cached stats")
    recorded_ipc = result.get("ipc2")
    if (not isinstance(recorded_ipc, (int, float))
            or isinstance(recorded_ipc, bool)
            or not math.isclose(
                float(recorded_ipc), measured_ipc, rel_tol=1e-12, abs_tol=0.0
            )):
        reasons.append("R2 result IPC2 differs from cached stats")
    if status.get("ipc2") != result.get("ipc2"):
        reasons.append("R2 status IPC2 differs from cached result")
    if effective_scope == SEMANTIC_SCOPE:
        instruction_vector = [
            measured["instructions"] for measured in measured_per_core
        ]
        if result.get("instruction_vector") != instruction_vector:
            reasons.append(
                "R2 result instruction vector differs from cached stats"
            )
        try:
            semantic = _semantic_metrics(stats, metadata, output_dir)
        except (KeyError, OSError, TypeError, ValueError) as error:
            reasons.append(f"cannot validate R2 semantic metrics: {error}")
            return
        for field, wanted in semantic.items():
            if isinstance(result.get(field), bool) or result.get(field) != wanted:
                reasons.append(f"R2 result {field} differs from semantic evidence")
            if isinstance(status.get(field), bool) or status.get(field) != wanted:
                reasons.append(f"R2 status {field} differs from semantic evidence")


def _validate_command(result: dict, status: dict, metadata: dict, vector: dict,
                      output_dir: Path, reasons: list[str],
                      scope: str | None = None) -> None:
    result_command = result.get("command")
    status_command = status.get("command")
    if status.get("return_code") != 0:
        reasons.append("R2 success status return_code is not zero")
    if not isinstance(result_command, list) or not all(
            isinstance(item, str) for item in result_command):
        reasons.append("R2 result command is malformed")
        return
    if status_command != result_command:
        reasons.append("R2 status command differs from cached result command")
    try:
        expected_tail = _command_tail(metadata, vector, scope)
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f"cannot establish canonical R2 command: {error}")
        return
    expected_prefix = [
        "--listener-mode=off", f"--outdir={output_dir.resolve()}",
    ]
    if (len(result_command) < 4
            or not Path(result_command[0]).is_absolute()
            or result_command[1:3] != expected_prefix
            or not Path(result_command[3]).is_absolute()
            or result_command[4:] != expected_tail):
        reasons.append("R2 command does not match canonical metadata and latency arguments")


def validate_local_result(r1_dir: Path, latency_path: Path, output_dir: Path,
                          instruction_window_scope: str | None = None) -> dict:
    """Decide whether an existing successful R2 is exactly reusable in place."""
    r1_dir = Path(r1_dir).resolve()
    latency_path = Path(latency_path).resolve()
    output_dir = Path(output_dir).resolve()
    result_path = output_dir / "r2_result.json"
    status_path = output_dir / "status.json"
    reasons: list[str] = []
    result = None
    status = None
    for path, label in ((result_path, "result"), (status_path, "status")):
        if not path.is_file():
            reasons.append(f"missing R2 {label}: {path}")
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError) as error:
            reasons.append(f"cannot read R2 {label} {path}: {error}")
            continue
        if not isinstance(value, dict):
            reasons.append(f"R2 {label} must contain an object: {path}")
            continue
        if label == "result":
            result = value
        else:
            status = value
    if result is None or status is None:
        return {
            "accepted": False,
            "reasons": reasons,
            "result_path": str(result_path),
            "status_path": str(status_path),
        }

    try:
        metadata = read_json(r1_dir / "r1_metadata.json")
        vector = read_json(latency_path)
        expected = _provenance(
            r1_dir, latency_path, instruction_window_scope
        )
    except (KeyError, OSError, ValueError) as error:
        reasons.append(f"cannot establish requested R2 provenance: {error}")
        expected = {}
    if result.get("schema_version") != 3:
        reasons.append("R2 result schema_version is not 3")
    if status.get("schema_version") != 3:
        reasons.append("R2 status schema_version is not 3")
    if status.get("state") != "success":
        reasons.append("R2 status state is not success")
    for field, wanted in expected.items():
        if result.get(field) != wanted:
            reasons.append(f"R2 result {field} does not match requested provenance")
        if status.get(field) != wanted:
            reasons.append(f"R2 status {field} does not match requested provenance")
    if status.get("r2_result") != str(result_path):
        reasons.append("R2 status r2_result does not identify the cached result")
    try:
        result_sha256 = sha256_file(result_path)
    except OSError as error:
        reasons.append(f"cannot hash R2 result {result_path}: {error}")
    else:
        if status.get("r2_result_sha256") != result_sha256:
            reasons.append("R2 status r2_result_sha256 does not match the cached result")
    if "metadata" in locals() and "vector" in locals():
        _validate_command(
            result, status, metadata, vector, output_dir, reasons,
            instruction_window_scope,
        )
        _validate_measurement(
            result, status, metadata, output_dir, reasons,
            instruction_window_scope,
        )
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "result_path": str(result_path),
        "status_path": str(status_path),
        "result": result,
        "status": status,
    }


def run(r1_dir: Path, latency_path: Path, output_dir: Path,
        gem5: Path = DEFAULT_GEM5, config: Path = DEFAULT_CONFIG,
        rerun: bool = False,
        instruction_window_scope: str | None = None) -> dict:
    if instruction_window_scope not in (None, "cpu0", "all-cores", SEMANTIC_SCOPE):
        raise ValueError(
            "instruction_window_scope must be cpu0, all-cores, "
            "semantic-work, or None"
        )
    result_path = output_dir / "r2_result.json"
    status_path = output_dir / "status.json"
    status_declares_success = False
    if not rerun and status_path.is_file():
        status_declares_success = read_json(status_path).get("state") == "success"
    if not rerun and (result_path.is_file() or status_declares_success):
        decision = validate_local_result(
            r1_dir, latency_path, output_dir, instruction_window_scope
        )
        if decision["accepted"]:
            return decision["result"]
        details = "; ".join(decision["reasons"])
        raise ValueError(
            f"existing R2 cache is incompatible ({details}); "
            "use --rerun to replace it"
        )

    metadata = read_json(r1_dir / "r1_metadata.json")
    vector = read_json(latency_path)
    effective_scope = instruction_window_scope or metadata.get(
        "instruction_window_scope", "cpu0"
    )
    provenance = _provenance(r1_dir, latency_path, effective_scope)
    command = [
        str(gem5.resolve()), "--listener-mode=off", f"--outdir={output_dir.resolve()}",
        str(config.resolve()),
        *_command_tail(metadata, vector, effective_scope),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(status_path, {
        "schema_version": 3, "state": "running", "started_unix": started,
        "command": command, **provenance,
    })
    stats_path = output_dir / "stats.txt"
    stats_path.unlink(missing_ok=True)
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
    (output_dir / "gem5.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        write_json(status_path, {
            "schema_version": 3, "state": "failed", "started_unix": started,
            "finished_unix": time.time(), "return_code": process.returncode,
            "error": f"gem5 exited with status {process.returncode}", "command": command,
            **provenance,
        })
        raise RuntimeError(f"gem5 R2 failed; see {output_dir / 'gem5.log'}")

    try:
        stats_path = stats_path.resolve()
        if not stats_path.is_file():
            raise ValueError("gem5 R2 completed without producing fresh stats")
        stats, stats_sha256 = _stats_snapshot(stats_path)
        per_core, ipc2 = _validated_measurement(stats, metadata, effective_scope)
        semantic = (
            _semantic_metrics(stats, metadata, output_dir)
            if effective_scope == SEMANTIC_SCOPE else {}
        )
        if sha256_file(stats_path) != stats_sha256:
            raise ValueError("R2 stats changed during validation")
        result = {
            "schema_version": 3, "command": command, **provenance,
            "ipc2": ipc2, "per_core": per_core,
            "instruction_vector": [item["instructions"] for item in per_core],
            "stats": str(stats_path), "stats_sha256": stats_sha256,
            "elapsed_seconds": time.time() - started,
            **semantic,
        }
        write_json(result_path, result)
        if sha256_file(stats_path) != stats_sha256:
            raise ValueError("R2 stats changed during validation")
        write_json(status_path, {
            "schema_version": 3, "state": "success", "started_unix": started,
            "finished_unix": time.time(), "return_code": 0,
            "ipc2": result["ipc2"], "stats": result["stats"],
            "stats_sha256": stats_sha256,
            "command": command,
            "r2_result": str(result_path.resolve()),
            "r2_result_sha256": sha256_file(result_path),
            **semantic,
            **provenance,
        })
        return result
    except Exception as error:
        write_json(status_path, {
            "schema_version": 3, "state": "failed", "started_unix": started,
            "finished_unix": time.time(), "return_code": 0,
            "error": f"{type(error).__name__}: {error}",
            **provenance,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gem5", type=Path, default=DEFAULT_GEM5)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--instruction-window-scope",
        choices=("cpu0", "all-cores", SEMANTIC_SCOPE),
        default=None,
        help="override the R1-recorded measurement anchor (cpu0 = same instruction stream)",
    )
    args = parser.parse_args()
    result = run(args.r1_dir.resolve(), args.latency.resolve(), args.output_dir.resolve(),
                 args.gem5, args.config, args.rerun,
                 args.instruction_window_scope)
    print(f"gem5 R2 IPC2 = {result['ipc2']:.6f}")


if __name__ == "__main__":
    main()
