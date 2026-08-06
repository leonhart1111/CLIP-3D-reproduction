#!/usr/bin/env python3
"""Run a resumable gem5 R2 and validate all measured cores."""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from pathlib import Path

from workflow.common import (
    PROJECT_ROOT,
    aggregate_ipc,
    parse_gem5_stats,
    read_json,
    sha256_file,
    write_json,
)


DEFAULT_GEM5 = PROJECT_ROOT / "tools/src/gem5/build/X86/gem5.opt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/gem5/clip_r1.py"


def _provenance(r1_dir: Path, latency_path: Path) -> dict:
    metadata = read_json(r1_dir / "r1_metadata.json")
    return {
        "r1_directory": str(r1_dir.resolve()),
        "r1_metadata_sha256": sha256_file(r1_dir / "r1_metadata.json"),
        "r1_stats_sha256": sha256_file(r1_dir / "stats.txt"),
        "latency_vector": str(latency_path.resolve()),
        "latency_sha256": sha256_file(latency_path),
        "instruction_window_scope": metadata.get("instruction_window_scope", "cpu0"),
        "warmup_insts_cpu0": metadata["warmup_insts_cpu0"],
        "measure_insts_cpu0": metadata["measure_insts_cpu0"],
    }


def _validate_measurement(result: dict, status: dict, r1_dir: Path,
                          output_dir: Path, reasons: list[str]) -> None:
    """Bind recorded IPC/per-core values to the measured R2 stats file."""
    stats_path = output_dir / "stats.txt"
    if not stats_path.is_file():
        reasons.append(f"missing R2 stats: {stats_path}")
        return
    for label, record in (("result", result), ("status", status)):
        value = record.get("stats")
        if (not isinstance(value, str)
                or Path(value).resolve() != stats_path.resolve()):
            reasons.append(f"R2 {label} stats does not identify the cached stats")
        if record.get("stats_sha256") != sha256_file(stats_path):
            reasons.append(f"R2 {label} stats_sha256 does not match cached stats")

    metadata = read_json(r1_dir / "r1_metadata.json")
    try:
        cores = int(metadata["num_cores"])
        scope = metadata.get("instruction_window_scope", "cpu0")
        minimum = int(metadata["measure_insts_cpu0"]) if scope == "all-cores" else 1
        stats = parse_gem5_stats(stats_path, include_nonfinite=True)
    except (KeyError, OSError, TypeError, ValueError) as error:
        reasons.append(f"cannot validate R2 stats: {error}")
        return
    per_core = result.get("per_core")
    if not isinstance(per_core, list) or len(per_core) != cores:
        reasons.append("R2 result per_core does not cover every canonical core")
        per_core = []
    for core in range(cores):
        instructions_value = stats.get(f"system.cpu{core}.commitStats0.numInsts")
        cycles_value = stats.get(f"system.cpu{core}.numCycles")
        if (not isinstance(instructions_value, (int, float))
                or not math.isfinite(instructions_value)
                or instructions_value <= 0
                or not float(instructions_value).is_integer()):
            reasons.append(f"R2 stats CPU{core} instructions are not positive integers")
            continue
        if instructions_value < minimum:
            reasons.append(
                f"R2 stats CPU{core} instructions are below measurement minimum "
                f"{minimum}"
            )
        if (not isinstance(cycles_value, (int, float))
                or not math.isfinite(cycles_value)
                or cycles_value <= 0
                or not float(cycles_value).is_integer()):
            reasons.append(f"R2 stats CPU{core} cycles are not positive integers")
            continue
        if core >= len(per_core) or not isinstance(per_core[core], dict):
            continue
        recorded = per_core[core]
        instructions = int(instructions_value)
        cycles = int(cycles_value)
        if (recorded.get("core") != core
                or recorded.get("instructions") != instructions
                or recorded.get("cycles") != cycles):
            reasons.append(f"R2 result CPU{core} counters differ from cached stats")
        recorded_ipc = recorded.get("ipc")
        if (not isinstance(recorded_ipc, (int, float))
                or isinstance(recorded_ipc, bool)
                or not math.isclose(
                    float(recorded_ipc), instructions / cycles,
                    rel_tol=1e-12, abs_tol=0.0,
                )):
            reasons.append(f"R2 result CPU{core} IPC differs from cached stats")
    try:
        measured_ipc = aggregate_ipc(stats, cores)
    except (TypeError, ValueError) as error:
        reasons.append(f"cannot aggregate R2 IPC from cached stats: {error}")
        return
    recorded_ipc = result.get("ipc2")
    if (not isinstance(recorded_ipc, (int, float))
            or isinstance(recorded_ipc, bool)
            or not math.isclose(
                float(recorded_ipc), measured_ipc, rel_tol=1e-12, abs_tol=0.0
            )):
        reasons.append("R2 result IPC2 differs from cached stats")
    if status.get("ipc2") != result.get("ipc2"):
        reasons.append("R2 status IPC2 differs from cached result")


def validate_local_result(r1_dir: Path, latency_path: Path, output_dir: Path) -> dict:
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
        expected = _provenance(r1_dir, latency_path)
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
    _validate_measurement(result, status, r1_dir, output_dir, reasons)
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
        rerun: bool = False) -> dict:
    result_path = output_dir / "r2_result.json"
    status_path = output_dir / "status.json"
    status_declares_success = False
    if not rerun and status_path.is_file():
        status_declares_success = read_json(status_path).get("state") == "success"
    if not rerun and (result_path.is_file() or status_declares_success):
        decision = validate_local_result(r1_dir, latency_path, output_dir)
        if decision["accepted"]:
            return decision["result"]
        details = "; ".join(decision["reasons"])
        raise ValueError(
            f"existing R2 cache is incompatible ({details}); "
            "use --rerun to replace it"
        )

    metadata = read_json(r1_dir / "r1_metadata.json")
    vector = read_json(latency_path)
    provenance = _provenance(r1_dir, latency_path)
    scope = metadata.get("instruction_window_scope", "cpu0")
    command = [
        str(gem5.resolve()), "--listener-mode=off", f"--outdir={output_dir.resolve()}",
        str(config.resolve()), "--stage", "R2", "--workload", metadata["workload"],
        "--l1i-size", metadata["l1i_size"], "--l1d-size", metadata["l1d_size"],
        "--l2-size", metadata["l2_size"], "--warmup-insts",
        str(metadata["warmup_insts_cpu0"]), "--measure-insts",
        str(metadata["measure_insts_cpu0"]), "--instruction-window-scope", scope,
        *vector["gem5_args"],
    ]
    options = metadata.get("command", [])[1:]
    if options:
        command.extend(("--options", " ".join(options)))
    if metadata.get("stdin"):
        command.extend(("--stdin", metadata["stdin"]))

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
        stats = parse_gem5_stats(stats_path)
        stats_sha256 = sha256_file(stats_path)
        cores = int(metadata["num_cores"])
        minimum = int(metadata["measure_insts_cpu0"]) if scope == "all-cores" else 1
        per_core = []
        for core in range(cores):
            instructions = int(stats.get(f"system.cpu{core}.commitStats0.numInsts", 0))
            cycles = int(stats.get(f"system.cpu{core}.numCycles", 0))
            if instructions < minimum or cycles <= 0:
                raise ValueError(
                    f"invalid R2 CPU{core}: instructions={instructions}, cycles={cycles}, "
                    f"required_instructions={minimum}"
                )
            per_core.append({"core": core, "instructions": instructions,
                             "cycles": cycles, "ipc": instructions / cycles})
        result = {
            "schema_version": 3, "command": command, **provenance,
            "ipc2": aggregate_ipc(stats, cores), "per_core": per_core,
            "stats": str(stats_path), "stats_sha256": stats_sha256,
            "elapsed_seconds": time.time() - started,
        }
        write_json(result_path, result)
        write_json(status_path, {
            "schema_version": 3, "state": "success", "started_unix": started,
            "finished_unix": time.time(), "return_code": 0,
            "ipc2": result["ipc2"], "stats": result["stats"],
            "stats_sha256": stats_sha256,
            "r2_result": str(result_path.resolve()),
            "r2_result_sha256": sha256_file(result_path),
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
    args = parser.parse_args()
    result = run(args.r1_dir.resolve(), args.latency.resolve(), args.output_dir.resolve(),
                 args.gem5, args.config, args.rerun)
    print(f"gem5 R2 IPC2 = {result['ipc2']:.6f}")


if __name__ == "__main__":
    main()
