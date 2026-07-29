#!/usr/bin/env python3
"""Run and resume the CLIP-3D gem5 R1 L1D x L2 cache sweep."""

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    warmup_insts: int
    measure_insts: int
    instruction_window_scope: str
    options: str | None
    output_dir: Path

    @property
    def job_id(self):
        return (
            f"{self.workload}__l1d_{self.l1d_size}__l2_{self.l2_size}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create or execute the 5 workload x 4 L1D x 5 L2 CLIP-3D "
            "R1 experiment grid. Without --execute, only the plan is written."
        )
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--profile", choices=("paper", "paper_all_cores", "smoke"), default="paper")
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


def make_jobs(args, experiment):
    workloads = select_values(
        args.workloads, experiment["workloads"], "workloads"
    )
    l1d_sizes = select_values(
        args.l1d_sizes, experiment["l1d_sizes"], "L1D sizes"
    )
    l2_sizes = select_values(
        args.l2_sizes, experiment["l2_sizes"], "L2 sizes"
    )
    profile = experiment["profiles"][args.profile]
    options = profile.get("workload_options", {})
    root = args.output_root.resolve() / args.profile

    jobs = []
    for workload, l1d_size, l2_size in itertools.product(
        workloads, l1d_sizes, l2_sizes
    ):
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
                warmup_insts=int(profile["warmup_insts"]),
                measure_insts=int(profile["measure_insts"]),
                instruction_window_scope=profile.get("instruction_window_scope", "cpu0"),
                options=options.get(workload),
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
        "--warmup-insts",
        str(job.warmup_insts),
        "--measure-insts",
        str(job.measure_insts),
        "--instruction-window-scope",
        job.instruction_window_scope,
    ]
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
    return result


def run_job(job, args):
    existing = read_status(job)
    if (
        not args.rerun
        and existing is not None
        and existing.get("state") == "success"
        and (job.output_dir / "stats.txt").is_file()
    ):
        print(f"[skip]  {job.job_id}", flush=True)
        return existing

    job.output_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(job, args)
    atomic_json(
        job.output_dir / "command.json",
        {
            "job": {**asdict(job), "output_dir": str(job.output_dir)},
            "argv": command,
            "shell_command": shlex.join(command),
        },
    )
    started = time.time()
    running = {
        "state": "running",
        "job_id": job.job_id,
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
        "return_code": return_code,
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "error": error,
        "stats_file": str(job.output_dir / "stats.txt"),
    }
    if state == "success":
        status.update(extract_stats(job.output_dir / "stats.txt"))
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
    jobs = make_jobs(args, experiment)
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
