#!/usr/bin/env python3
"""Run and resume the CLIP-3D gem5 R1 L1D x L2 cache sweep."""

import argparse
import concurrent.futures
import csv
import hashlib
import itertools
import json
import math
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow.r1_protocol import (
    PROTOCOL_FAMILY,
    SEMANTIC_SCOPE,
    canonical_protocol,
    protocol_id,
)
from workflow.common import parse_frequency_hz
DEFAULT_EXPERIMENT = PROJECT_ROOT / "configs/experiments/r1_cache_sweep.json"
DEFAULT_GEM5 = PROJECT_ROOT / "tools/src/gem5/build/X86/gem5.opt"
DEFAULT_R1_CONFIG = PROJECT_ROOT / "configs/gem5/clip_r1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/architecture_sweep/r1"


@dataclass(frozen=True)
class Job:
    workload: str
    l1d_size: str
    l2_size: str
    profile: str
    family: str
    warmup_insts: int | None
    measure_insts: int | None
    instruction_window_scope: str
    options: str | None
    protocol_id: str
    output_dir: Path
    warmup_work_units: int | None = None
    measure_work_units: int | None = None
    work_unit_type: str | None = None
    workload_binary_sha256: str | None = None

    @property
    def job_id(self):
        return (
            f"{self.workload}__l1d_{self.l1d_size}__l2_{self.l2_size}"
        )

    @property
    def protocol(self) -> dict:
        value = {
            "family": self.family,
            "profile": self.profile,
            "instruction_window_scope": self.instruction_window_scope,
        }
        if self.instruction_window_scope == SEMANTIC_SCOPE:
            value.update({
                "warmup_work_units": self.warmup_work_units,
                "measure_work_units": self.measure_work_units,
                "work_unit_type": self.work_unit_type,
            })
        else:
            value.update({
                "warmup_insts": self.warmup_insts,
                "measure_insts": self.measure_insts,
            })
        return canonical_protocol(value)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create or execute the 5 workload x 4 L1D x 5 L2 CLIP-3D "
            "R1 experiment grid. Without --execute, only the plan is written."
        )
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--profile")
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--l1d-sizes", nargs="+")
    parser.add_argument("--l2-sizes", nargs="+")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gem5", type=Path, default=DEFAULT_GEM5)
    parser.add_argument("--r1-config", type=Path, default=DEFAULT_R1_CONFIG)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run gem5; otherwise generate the 100-point plan only",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="rerun jobs whose status.json already records success",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least one")
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be non-negative")
    return args


def load_experiment(path):
    with path.resolve().open() as stream:
        experiment = json.load(stream)
    if experiment.get("schema_version") != 1:
        raise ValueError("unsupported experiment schema")
    return experiment


def select_values(requested, available, label):
    if requested is None:
        return list(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    return requested


def safe_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def make_jobs(args, experiment, r1_config_sha256, gem5_binary_sha256,
              workload_binary_sha256=None):
    workloads = select_values(
        args.workloads, experiment["workloads"], "workloads"
    )
    l1d_sizes = select_values(
        args.l1d_sizes, experiment["l1d_sizes"], "L1D sizes"
    )
    l2_sizes = select_values(
        args.l2_sizes, experiment["l2_sizes"], "L2 sizes"
    )
    profiles = experiment.get("profiles")
    if not isinstance(profiles, dict) or args.profile not in profiles:
        raise ValueError(
            f"unknown profile {args.profile!r}; available: "
            + ", ".join(sorted(profiles))
        )
    profile = profiles[args.profile]
    options = profile.get("workload_options", {})
    scope = profile.get("instruction_window_scope", "cpu0")
    semantic_units = profile.get("semantic_work_units", {})
    workload_binary_sha256 = workload_binary_sha256 or {}
    root = args.output_root.resolve() / args.profile

    jobs = []
    for workload, l1d_size, l2_size in itertools.product(
        workloads, l1d_sizes, l2_sizes
    ):
        protocol_value = {
            "family": PROTOCOL_FAMILY,
            "profile": args.profile,
            "instruction_window_scope": scope,
        }
        if scope == SEMANTIC_SCOPE:
            unit = semantic_units.get(workload)
            if not isinstance(unit, dict):
                raise ValueError(
                    f"semantic profile lacks work-unit declaration for {workload}"
                )
            protocol_value.update({
                "warmup_work_units": unit.get("warmup"),
                "measure_work_units": unit.get("measure"),
                "work_unit_type": unit.get("type"),
            })
            binary_digest = workload_binary_sha256.get(workload)
            if (not isinstance(binary_digest, str) or len(binary_digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in binary_digest)):
                raise ValueError(
                    f"semantic profile requires a built, hashed {workload} binary"
                )
        else:
            protocol_value.update({
                "warmup_insts": profile.get("warmup_insts"),
                "measure_insts": profile.get("measure_insts"),
            })
            binary_digest = None
        canonical = canonical_protocol(protocol_value)
        output_dir = (
            root
            / safe_component(workload)
            / f"l1d_{safe_component(l1d_size)}"
            / f"l2_{safe_component(l2_size)}"
        )
        jobs.append(
            Job(
                workload=workload,
                l1d_size=l1d_size,
                l2_size=l2_size,
                profile=args.profile,
                family=canonical["family"],
                warmup_insts=canonical.get("warmup_insts"),
                measure_insts=canonical.get("measure_insts"),
                warmup_work_units=canonical.get("warmup_work_units"),
                measure_work_units=canonical.get("measure_work_units"),
                work_unit_type=canonical.get("work_unit_type"),
                instruction_window_scope=scope,
                options=options.get(workload),
                workload_binary_sha256=binary_digest,
                protocol_id=protocol_id(
                    canonical,
                    workload_options=options.get(workload),
                    gem5_config_sha256=r1_config_sha256,
                    gem5_binary_sha256=gem5_binary_sha256,
                    workload_binary_sha256=binary_digest,
                ),
                output_dir=output_dir,
            )
        )
    return jobs


def command_for(job, args):
    command = [
        str(args.gem5.resolve()),
        "--listener-mode=off",
        f"--outdir={job.output_dir}",
        str(args.r1_config.resolve()),
        "--workload",
        job.workload,
        "--l1d-size",
        job.l1d_size,
        "--l2-size",
        job.l2_size,
        "--instruction-window-scope",
        job.instruction_window_scope,
        "--r1-protocol-family",
        job.family,
        "--r1-profile",
        job.profile,
        "--r1-protocol-id",
        job.protocol_id,
    ]
    if job.instruction_window_scope == SEMANTIC_SCOPE:
        command.extend((
            "--warmup-work-units", str(job.warmup_work_units),
            "--measure-work-units", str(job.measure_work_units),
            "--work-unit-type", str(job.work_unit_type),
        ))
    else:
        command.extend((
            "--warmup-insts", str(job.warmup_insts),
            "--measure-insts", str(job.measure_insts),
        ))
    if job.workload_binary_sha256 is not None:
        command.extend((
            "--workload-binary-sha256", job.workload_binary_sha256,
        ))
    if job.options is not None:
        command.extend(("--options", job.options))
    return command


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def read_status(job):
    path = job.output_dir / "status.json"
    if not path.is_file():
        return None
    try:
        with path.open() as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def is_reusable(existing, job) -> bool:
    """A successful output is reusable only under an identical protocol."""
    return bool(
        existing is not None
        and existing.get("state") == "success"
        and existing.get("r1_protocol_id") == job.protocol_id
    )


def artifacts_reusable(existing, job) -> bool:
    """Require semantic marker/binary evidence before accepting a cached R1."""
    if not is_reusable(existing, job):
        return False
    stats_path = job.output_dir / "stats.txt"
    if not stats_path.is_file():
        return False
    if job.instruction_window_scope != SEMANTIC_SCOPE:
        return True
    try:
        statistics = extract_stats(stats_path)
        if any(statistics[f"cpu{core}_insts"] <= 0 for core in range(4)):
            return False
        semantic_evidence(job.output_dir, job)
        metadata_path = job.output_dir / "r1_metadata.json"
        evidence_path = job.output_dir / "roi_events.json"
        with metadata_path.open() as stream:
            metadata = json.load(stream)
        if (metadata.get("r1_protocol_id") != job.protocol_id
                or canonical_protocol(metadata.get("r1_protocol", {})) != job.protocol
                or metadata.get("binary_sha256") != job.workload_binary_sha256
                or existing.get("stats_sha256") != hashlib.sha256(
                    stats_path.read_bytes()).hexdigest()
                or existing.get("r1_metadata_sha256") != hashlib.sha256(
                    metadata_path.read_bytes()).hexdigest()
                or existing.get("roi_events_sha256") != hashlib.sha256(
                    evidence_path.read_bytes()).hexdigest()):
            return False
        binary = Path(metadata["binary"])
        return (
            binary.is_file()
            and hashlib.sha256(binary.read_bytes()).hexdigest()
            == job.workload_binary_sha256
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def parse_number(text):
    value = float(text)
    return value if math.isfinite(value) else None


def extract_stats(path):
    text = path.read_text()
    result = {}
    for core in range(4):
        inst_match = re.search(
            rf"^system\.cpu{core}\.commitStats0\.numInsts\s+([0-9.eE+-]+)",
            text,
            re.M,
        )
        ipc_match = re.search(
            rf"^system\.cpu{core}\.commitStats0\.ipc\s+([0-9.eE+.-]+)",
            text,
            re.M,
        )
        cycle_match = re.search(
            rf"^system\.cpu{core}\.numCycles\s+([0-9.eE+-]+)",
            text,
            re.M,
        )
        if not inst_match or not ipc_match or not cycle_match:
            raise ValueError(f"missing CPU{core} statistics in {path}")
        result[f"cpu{core}_insts"] = int(float(inst_match.group(1)))
        result[f"cpu{core}_ipc"] = parse_number(ipc_match.group(1))
        result[f"cpu{core}_cycles"] = int(float(cycle_match.group(1)))

    total_insts = sum(result[f"cpu{core}_insts"] for core in range(4))
    wall_cycles = max(result[f"cpu{core}_cycles"] for core in range(4))
    result["total_insts"] = total_insts
    result["wall_cycles"] = wall_cycles
    result["aggregate_ipc"] = (
        total_insts / wall_cycles if wall_cycles else None
    )
    for name in ("simTicks", "simFreq"):
        match = re.search(rf"^{name}\s+([0-9.eE+-]+)", text, re.M)
        if match is None:
            raise ValueError(f"missing {name} in {path}")
        value = float(match.group(1))
        if (not math.isfinite(value) or value <= 0
                or not value.is_integer()):
            raise ValueError(f"{name} must be a finite positive integer in {path}")
        result[name] = int(value)
    return result


def semantic_evidence(output_dir: Path, job: Job) -> dict:
    """Require the two synchronized benchmark markers for a semantic ROI."""
    path = output_dir / "roi_events.json"
    if not path.is_file():
        raise ValueError(f"semantic ROI lacks marker evidence: {path}")
    with path.open() as stream:
        evidence = json.load(stream)
    if not isinstance(evidence, dict):
        raise ValueError("semantic ROI evidence must contain an object")
    expected = {
        "scope": SEMANTIC_SCOPE,
        "work_unit_type": job.work_unit_type,
        "warmup_work_units": job.warmup_work_units,
        "measure_work_units": job.measure_work_units,
        "marker_work_id": 1,
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise ValueError(
                f"semantic ROI evidence {field} differs: "
                f"expected {value!r}, observed {evidence.get(field)!r}"
            )
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
        raise ValueError("semantic ROI marker sequence is incomplete or unordered")
    statistics = extract_stats(output_dir / "stats.txt")
    completion_ticks = events[1]["tick"] - events[0]["tick"]
    if (evidence.get("completion_ticks") != completion_ticks
            or completion_ticks != statistics["simTicks"]):
        raise ValueError("semantic marker ticks differ from measured stats simTicks")
    return evidence


def run_job(job, args):
    existing = read_status(job)
    if not args.rerun and artifacts_reusable(existing, job):
        print(f"[skip]  {job.job_id}", flush=True)
        return existing

    job.output_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(job, args)
    atomic_json(
        job.output_dir / "command.json",
        {
            "job": {**asdict(job), "output_dir": str(job.output_dir)},
            "r1_protocol": job.protocol,
            "r1_protocol_id": job.protocol_id,
            "argv": command,
            "shell_command": shlex.join(command),
        },
    )
    started = time.time()
    running = {
        "state": "running",
        "job_id": job.job_id,
        "r1_protocol": job.protocol,
        "r1_protocol_id": job.protocol_id,
        "started_unix": started,
        "command": command,
    }
    atomic_json(job.output_dir / "status.json", running)
    print(f"[start] {job.job_id}", flush=True)

    state = "failed"
    return_code = None
    error = None
    try:
        with (job.output_dir / "stdout.log").open("w") as stdout, (
            job.output_dir / "stderr.log"
        ).open("w") as stderr:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout_seconds or None,
                check=False,
            )
        return_code = completed.returncode
        if return_code == 0:
            statistics = extract_stats(job.output_dir / "stats.txt")
            inactive = [
                core
                for core in range(4)
                if statistics[f"cpu{core}_insts"] <= 0
            ]
            if inactive:
                raise ValueError(f"inactive measured cores: {inactive}")
            if job.instruction_window_scope == "all-cores":
                short = [core for core in range(4)
                         if statistics[f"cpu{core}_insts"] < job.measure_insts]
                if short:
                    raise ValueError(f"cores below measurement target: {short}")
            if job.instruction_window_scope == SEMANTIC_SCOPE:
                semantic_evidence(job.output_dir, job)
            state = "success"
        else:
            error = f"gem5 exited with status {return_code}"
    except subprocess.TimeoutExpired:
        state = "timeout"
        error = f"exceeded {args.timeout_seconds} seconds"
    except Exception as exception:
        error = str(exception)

    finished = time.time()
    status = {
        "state": state,
        "job_id": job.job_id,
        "r1_protocol": job.protocol,
        "r1_protocol_id": job.protocol_id,
        "return_code": return_code,
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "error": error,
        "stats_file": str(job.output_dir / "stats.txt"),
    }
    if state == "success":
        statistics = extract_stats(job.output_dir / "stats.txt")
        status.update({
            "stats_sha256": hashlib.sha256(
                (job.output_dir / "stats.txt").read_bytes()
            ).hexdigest(),
            "r1_metadata_sha256": hashlib.sha256(
                (job.output_dir / "r1_metadata.json").read_bytes()
            ).hexdigest(),
        })
        status.update(statistics)
        if job.instruction_window_scope == SEMANTIC_SCOPE:
            evidence = semantic_evidence(job.output_dir, job)
            with (job.output_dir / "r1_metadata.json").open() as stream:
                metadata = json.load(stream)
            cpu_clock_hz = parse_frequency_hz(metadata["cpu_clock"])
            completion_cycles = (
                evidence["completion_ticks"] * cpu_clock_hz
                / statistics["simFreq"]
            )
            status.update({
                "work_unit_type": job.work_unit_type,
                "warmup_work_units": job.warmup_work_units,
                "measure_work_units": job.measure_work_units,
                "completion_ticks": evidence["completion_ticks"],
                "completion_cycles": completion_cycles,
                "work_units_per_cycle": (
                    job.measure_work_units / completion_cycles
                ),
                "instruction_vector": [
                    statistics[f"cpu{core}_insts"] for core in range(4)
                ],
                "primary_performance_metric": "work_units_per_cycle",
                "roi_events": str(job.output_dir / "roi_events.json"),
                "roi_events_sha256": hashlib.sha256(
                    (job.output_dir / "roi_events.json").read_bytes()
                ).hexdigest(),
                "workload_binary_sha256": job.workload_binary_sha256,
            })
    atomic_json(job.output_dir / "status.json", status)
    print(
        f"[{state:7}] {job.job_id} ({status['elapsed_seconds']:.1f}s)",
        flush=True,
    )
    return status


def write_plan(jobs, args, experiment):
    root = args.output_root.resolve() / args.profile
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "experiment": experiment["name"],
        "profile": args.profile,
        "default_profile": experiment.get("default_profile", "paper"),
        "job_count": len(jobs),
        "jobs": [
            {
                **asdict(job),
                "output_dir": str(job.output_dir),
                "argv": command_for(job, args),
            }
            for job in jobs
        ],
    }
    atomic_json(root / "planned_jobs.json", plan)
    return root


def write_summary(jobs, root):
    rows = []
    for job in jobs:
        status = read_status(job) or {"state": "not_run"}
        row = {
            "job_id": job.job_id,
            "workload": job.workload,
            "l1d_size": job.l1d_size,
            "l2_size": job.l2_size,
            "profile": job.profile,
            "instruction_window_scope": job.instruction_window_scope,
            "work_unit_type": job.work_unit_type,
            "warmup_work_units": job.warmup_work_units,
            "measure_work_units": job.measure_work_units,
            "state": status.get("state"),
            "return_code": status.get("return_code"),
            "elapsed_seconds": status.get("elapsed_seconds"),
            "error": status.get("error"),
            "stats_file": status.get("stats_file"),
        }
        for key in (
            "cpu0_insts", "cpu1_insts", "cpu2_insts", "cpu3_insts",
            "cpu0_ipc", "cpu1_ipc", "cpu2_ipc", "cpu3_ipc",
            "total_insts", "wall_cycles", "aggregate_ipc",
            "completion_cycles", "work_units_per_cycle",
        ):
            row[key] = status.get(key)
        rows.append(row)

    atomic_json(root / "summary.json", rows)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    args = parse_arguments()
    args.experiment = args.experiment.resolve()
    args.gem5 = args.gem5.resolve()
    args.r1_config = args.r1_config.resolve()
    args.output_root = args.output_root.resolve()
    if not args.gem5.is_file():
        raise FileNotFoundError(f"gem5 binary not found: {args.gem5}")
    if not args.r1_config.is_file():
        raise FileNotFoundError(f"R1 configuration not found: {args.r1_config}")

    experiment = load_experiment(args.experiment)
    if args.profile is None:
        args.profile = experiment.get("default_profile", "paper")
    r1_config_sha256 = hashlib.sha256(
        args.r1_config.read_bytes()
    ).hexdigest()
    gem5_binary_sha256 = hashlib.sha256(
        args.gem5.read_bytes()
    ).hexdigest()
    workload_hashes = {}
    for workload in experiment["workloads"]:
        binary = PROJECT_ROOT / "benchmarks/bin" / workload
        if binary.is_file():
            workload_hashes[workload] = hashlib.sha256(binary.read_bytes()).hexdigest()
        elif args.execute and (
                args.workloads is None or workload in args.workloads):
            raise FileNotFoundError(f"workload binary not found: {binary}")
    jobs = make_jobs(
        args, experiment, r1_config_sha256, gem5_binary_sha256,
        workload_hashes,
    )
    root = write_plan(jobs, args, experiment)
    print(
        f"Planned {len(jobs)} jobs for profile '{args.profile}' in {root}",
        flush=True,
    )
    if not args.execute:
        write_summary(jobs, root)
        print("Plan only; pass --execute to run gem5.", flush=True)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_job, job, args) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    rows = write_summary(jobs, root)
    counts = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    print(f"Sweep summary: {counts}", flush=True)
    if counts.get("failed", 0) or counts.get("timeout", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
