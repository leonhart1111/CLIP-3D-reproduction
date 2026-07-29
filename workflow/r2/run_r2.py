#!/usr/bin/env python3
"""Run a resumable gem5 R2 and validate all measured cores."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from workflow.common import PROJECT_ROOT, aggregate_ipc, parse_gem5_stats, read_json, write_json


DEFAULT_GEM5 = PROJECT_ROOT / "tools/src/gem5/build/X86/gem5.opt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/gem5/clip_r1.py"


def run(r1_dir: Path, latency_path: Path, output_dir: Path,
        gem5: Path = DEFAULT_GEM5, config: Path = DEFAULT_CONFIG,
        rerun: bool = False) -> dict:
    result_path = output_dir / "r2_result.json"
    status_path = output_dir / "status.json"
    if not rerun and result_path.is_file() and status_path.is_file():
        if read_json(status_path).get("state") == "success":
            return read_json(result_path)

    metadata = read_json(r1_dir / "r1_metadata.json")
    vector = read_json(latency_path)
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
        "schema_version": 1, "state": "running", "started_unix": started,
        "command": command, "latency_vector": str(latency_path.resolve()),
    })
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
    (output_dir / "gem5.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        write_json(status_path, {
            "schema_version": 1, "state": "failed", "started_unix": started,
            "finished_unix": time.time(), "return_code": process.returncode,
            "error": f"gem5 exited with status {process.returncode}", "command": command,
        })
        raise RuntimeError(f"gem5 R2 failed; see {output_dir / 'gem5.log'}")

    try:
        stats = parse_gem5_stats(output_dir / "stats.txt")
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
            "schema_version": 2, "command": command,
            "latency_vector": str(latency_path.resolve()),
            "instruction_window_scope": scope,
            "ipc2": aggregate_ipc(stats, cores), "per_core": per_core,
            "stats": str((output_dir / "stats.txt").resolve()),
            "elapsed_seconds": time.time() - started,
        }
        write_json(result_path, result)
        write_json(status_path, {
            "schema_version": 1, "state": "success", "started_unix": started,
            "finished_unix": time.time(), "return_code": 0,
            "ipc2": result["ipc2"], "stats": result["stats"],
        })
        return result
    except Exception as error:
        write_json(status_path, {
            "schema_version": 1, "state": "failed", "started_unix": started,
            "finished_unix": time.time(), "return_code": 0,
            "error": f"{type(error).__name__}: {error}",
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
