#!/usr/bin/env python3
"""Re-run one completed R1 point with periodic statistics in a separate directory."""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import time
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, write_json


DEFAULT_GEM5 = PROJECT_ROOT / "tools/src/gem5/build/X86/gem5.opt"
TRANSIENT_CONFIG = PROJECT_ROOT / "configs/gem5/clip_r1_transient.py"


def completed(output_dir: Path, sample_ms: float) -> bool:
    status = output_dir / "status.json"
    metadata = output_dir / "r1_metadata.json"
    stats = output_dir / "stats.txt"
    if not status.is_file() or not metadata.is_file() or not stats.is_file():
        return False
    recorded = read_json(metadata)
    return (
        read_json(status).get("state") == "success"
        and recorded.get("transient_statistics") is True
        and abs(float(recorded.get("sample_interval_ms", -1)) - sample_ms) < 1e-12
    )


def command_from_metadata(source_r1_dir: Path, output_dir: Path, gem5: Path,
                          sample_ms: float) -> list[str]:
    metadata = read_json(source_r1_dir / "r1_metadata.json")
    command = [
        str(gem5.resolve()),
        "--listener-mode=off",
        f"--outdir={output_dir.resolve()}",
        str(TRANSIENT_CONFIG.resolve()),
        "--sample-ms", str(sample_ms),
        "--workload", metadata["workload"],
        "--binary", metadata["binary"],
        "--l1i-size", metadata["l1i_size"],
        "--l1d-size", metadata["l1d_size"],
        "--l2-size", metadata["l2_size"],
        "--mem-size", metadata["memory_size"],
        "--cpu-clock", metadata["cpu_clock"],
        "--warmup-insts", str(metadata["warmup_insts"]),
        "--measure-insts", str(metadata["measure_insts"]),
        "--instruction-window-scope",
        metadata.get("instruction_window_scope", "cpu0"),
    ]
    workload_command = list(metadata.get("command", []))
    if len(workload_command) > 1:
        command.extend(("--options", shlex.join(workload_command[1:])))
    if metadata.get("stdin"):
        command.extend(("--stdin", metadata["stdin"]))
    latencies = metadata.get("latencies", {})
    for level in ("l1i", "l1d", "l2"):
        values = latencies.get(level, {})
        for field in ("tag", "data", "response"):
            if field in values:
                command.extend((f"--{level}-{field}-latency", str(values[field])))
    xbar = latencies.get("xbar", {})
    for field in ("frontend", "forward", "response", "snoop_response"):
        if field in xbar:
            option = field.replace("_", "-")
            command.extend((f"--xbar-{option}-latency", str(xbar[field])))
    return command


def run(source_r1_dir: Path, output_dir: Path, sample_ms: float = 10.0,
        gem5: Path = DEFAULT_GEM5, rerun: bool = False) -> dict:
    source_r1_dir = source_r1_dir.resolve()
    output_dir = output_dir.resolve()
    if not math.isfinite(sample_ms) or sample_ms <= 0:
        raise ValueError("sample_ms must be positive")
    for required in (source_r1_dir / "r1_metadata.json", source_r1_dir / "stats.txt"):
        if not required.is_file():
            raise FileNotFoundError(required)
    source_status = source_r1_dir / "status.json"
    if source_status.is_file() and read_json(source_status).get("state") != "success":
        raise ValueError(f"source R1 is not complete: {source_status}")
    if not gem5.is_file():
        raise FileNotFoundError(gem5)
    if not TRANSIENT_CONFIG.is_file():
        raise FileNotFoundError(TRANSIENT_CONFIG)
    if not rerun and completed(output_dir, sample_ms):
        return read_json(output_dir / "status.json")
    if not rerun and (output_dir / "status.json").is_file():
        state = read_json(output_dir / "status.json").get("state")
        raise RuntimeError(
            f"transient R1 is not reusable (state={state} or sampling mismatch); "
            "use a new directory or --rerun"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    command = command_from_metadata(source_r1_dir, output_dir, gem5, sample_ms)
    write_json(output_dir / "command.json", {
        "schema_version": 1,
        "source_r1": str(source_r1_dir),
        "argv": command,
        "shell_command": shlex.join(command),
    })
    started = time.time()
    write_json(output_dir / "status.json", {
        "state": "running",
        "started_unix": started,
        "source_r1": str(source_r1_dir),
        "sample_interval_ms": sample_ms,
        "command": command,
    })
    try:
        with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            output_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr)
        if process.returncode != 0:
            raise RuntimeError(
                f"transient R1 failed with rc={process.returncode}; "
                f"see {output_dir / 'stderr.log'}"
            )
        metadata = read_json(output_dir / "r1_metadata.json")
        if metadata.get("transient_statistics") is not True:
            raise RuntimeError("transient R1 did not record periodic-statistics metadata")
        result = {
            "state": "success",
            "started_unix": started,
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "source_r1": str(source_r1_dir),
            "sample_interval_ms": sample_ms,
            "command": command,
        }
    except BaseException as error:
        write_json(output_dir / "status.json", {
            "state": "failed",
            "started_unix": started,
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "source_r1": str(source_r1_dir),
            "sample_interval_ms": sample_ms,
            "command": command,
            "error": f"{type(error).__name__}: {error}",
        })
        raise
    write_json(output_dir / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-r1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ms", type=float, default=10.0)
    parser.add_argument("--gem5", type=Path, default=DEFAULT_GEM5)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source_r1_dir, args.output_dir, args.sample_ms,
        args.gem5.resolve(), args.rerun,
    )
    print(
        f"Transient R1 complete: sample={result['sample_interval_ms']:.6g} ms, "
        f"elapsed={result.get('elapsed_seconds', 0):.1f} s"
    )


if __name__ == "__main__":
    main()
