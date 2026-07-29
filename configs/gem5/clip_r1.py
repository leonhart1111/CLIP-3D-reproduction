#!/usr/bin/env python3
"""CLIP-3D R1: four-core X86 O3 syscall-emulation configuration.

This configuration intentionally assigns one Process object to four one-thread
CPU contexts. Only the first context starts at initialization; Linux clone()
system calls place the three pthread children on the remaining halted cores.
Assigning four different Process objects would incorrectly run four independent
copies of the benchmark.
"""

import argparse
import json
import re
import shlex
from pathlib import Path

import m5
from m5.objects import (
    AddrRange,
    Cache,
    DDR3_1600_8x8,
    L2XBar,
    MemCtrl,
    Process,
    Root,
    SEWorkload,
    SrcClockDomain,
    System,
    SystemXBar,
    VoltageDomain,
    X86O3CPU,
)


NUM_CORES = 4
WARMUP_CAUSE = "CLIP-3D R1 warmup complete"
MEASUREMENT_CAUSE = "CLIP-3D R1 measurement complete"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class L1InstructionCache(Cache):
    assoc = 2
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 8
    tgts_per_mshr = 20
    is_read_only = True
    writeback_clean = True


class L1DataCache(Cache):
    assoc = 2
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 16
    tgts_per_mshr = 20
    write_buffers = 8


class SharedL2Cache(Cache):
    assoc = 8
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 32
    tgts_per_mshr = 16
    write_buffers = 16


def nonnegative_integer(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run one CLIP-3D workload on four 2 GHz X86 O3 cores in "
            "gem5 syscall-emulation mode."
        )
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=("fft", "cholesky", "stream", "matmul", "stencil"),
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="override benchmarks/bin/<workload>",
    )
    parser.add_argument(
        "--options",
        help="override the workload arguments, expressed as one shell string",
    )
    parser.add_argument(
        "--stdin",
        type=Path,
        help="redirect benchmark stdin (defaults to tk14.O for cholesky)",
    )
    parser.add_argument("--l1i-size", default="32kB")
    parser.add_argument("--l1d-size", default="32kB")
    parser.add_argument("--l2-size", default="512kB")
    parser.add_argument("--mem-size", default="2GiB")
    parser.add_argument("--cpu-clock", default="2GHz")
    parser.add_argument("--system-clock", default="1GHz")
    parser.add_argument("--stage", choices=("R1", "R2"), default="R1")
    parser.add_argument(
        "--instruction-window-scope",
        choices=("cpu0", "all-cores"),
        default="cpu0",
        help=("cpu0 preserves the in-progress reproduction protocol; all-cores "
              "waits until every core reaches each instruction target"),
    )
    for level in ("l1i", "l1d", "l2"):
        parser.add_argument(f"--{level}-tag-latency", type=positive_integer, default=1)
        parser.add_argument(f"--{level}-data-latency", type=positive_integer, default=1)
        parser.add_argument(f"--{level}-response-latency", type=positive_integer, default=1)
    parser.add_argument("--xbar-frontend-latency", type=positive_integer, default=1)
    parser.add_argument("--xbar-forward-latency", type=positive_integer, default=1)
    parser.add_argument("--xbar-response-latency", type=positive_integer, default=1)
    parser.add_argument("--xbar-snoop-response-latency", type=positive_integer, default=1)
    parser.add_argument(
        "--warmup-insts",
        type=nonnegative_integer,
        default=100_000_000,
        help="CPU0 committed instructions before statistics reset",
    )
    parser.add_argument(
        "--measure-insts",
        type=positive_integer,
        default=500_000_000,
        help="CPU0 committed instructions in the measured region",
    )
    parser.add_argument(
        "--max-ticks-per-phase",
        type=nonnegative_integer,
        default=0,
        help="optional deadlock guard; zero means m5.MaxTick",
    )
    return parser.parse_args()


def workload_defaults(name):
    defaults = {
        "fft": ["-m16", "-p4", "-r100"],
        "cholesky": ["-p4", "-r100"],
        "stream": [],
        "matmul": ["-n", "1024", "-r", "1", "-t", "4"],
        "stencil": ["-n", "2048", "-i", "500", "-t", "4"],
    }
    return defaults[name]


def resolve_path(path):
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return resolved


def build_process(args):
    binary = resolve_path(
        args.binary if args.binary is not None
        else Path("benchmarks/bin") / args.workload
    )
    if not binary.is_file():
        raise FileNotFoundError(f"workload binary does not exist: {binary}")

    options = (
        shlex.split(args.options)
        if args.options is not None
        else workload_defaults(args.workload)
    )
    stdin_path = args.stdin
    if stdin_path is None and args.workload == "cholesky":
        stdin_path = Path("benchmarks/inputs/cholesky/tk14.O")
    if stdin_path is not None:
        stdin_path = resolve_path(stdin_path)
        if not stdin_path.is_file():
            raise FileNotFoundError(f"stdin file does not exist: {stdin_path}")

    process = Process()
    process.executable = str(binary)
    process.cmd = [str(binary), *options]
    process.cwd = str(PROJECT_ROOT)
    if stdin_path is not None:
        process.input = str(stdin_path)
    # gem5 23.1 SE exposes no usable shared-cache size to this glibc build,
    # leaving __x86_shared_non_temporal_threshold at zero.  glibc then sends
    # sub-16-KiB memcpy calls into a non-temporal loop that assumes at least
    # one 16-KiB block and accesses unmapped memory.  Keep normal benchmark
    # copies on the same SSE2 path selected on a native host.
    environment = [
        "GLIBC_TUNABLES=glibc.cpu.x86_non_temporal_threshold=1073741824"
    ]
    if args.workload == "stream":
        environment.extend([
            "OMP_NUM_THREADS=4",
            "OMP_DYNAMIC=FALSE",
            "OMP_PROC_BIND=FALSE",
        ])
    process.env = environment
    return process, binary, options, stdin_path


def configure_o3_cpu(cpu):
    cpu.fetchWidth = 4
    cpu.decodeWidth = 4
    cpu.renameWidth = 4
    cpu.dispatchWidth = 4
    cpu.issueWidth = 4
    # Writeback is wider than issue because several variable-latency units can
    # complete together. A four-entry writeback queue can overflow gem5 23.1's
    # default five-cycle forward time buffer even with four-wide issue.
    cpu.wbWidth = 8
    cpu.commitWidth = 4
    cpu.squashWidth = 4
    cpu.numROBEntries = 192
    cpu.forwardComSize = 10


def build_system(args, process):
    system = System()
    system.mem_mode = "timing"
    system.mem_ranges = [AddrRange(args.mem_size)]
    system.cache_line_size = 64

    system.voltage_domain = VoltageDomain()
    system.clk_domain = SrcClockDomain(
        clock=args.system_clock,
        voltage_domain=system.voltage_domain,
    )
    system.cpu_voltage_domain = VoltageDomain()
    system.cpu_clk_domain = SrcClockDomain(
        clock=args.cpu_clock,
        voltage_domain=system.cpu_voltage_domain,
    )

    system.cpu = [
        X86O3CPU(cpu_id=index, clk_domain=system.cpu_clk_domain)
        for index in range(NUM_CORES)
    ]
    system.to_l2_bus = L2XBar(
        clk_domain=system.cpu_clk_domain,
        frontend_latency=args.xbar_frontend_latency,
        forward_latency=args.xbar_forward_latency,
        response_latency=args.xbar_response_latency,
        snoop_response_latency=args.xbar_snoop_response_latency,
        width=64,
    )
    system.l2 = SharedL2Cache(
        size=args.l2_size,
        clk_domain=system.cpu_clk_domain,
        tag_latency=args.l2_tag_latency,
        data_latency=args.l2_data_latency,
        response_latency=args.l2_response_latency,
    )
    system.memory_bus = SystemXBar(width=64)
    system.l2.cpu_side = system.to_l2_bus.mem_side_ports
    system.l2.mem_side = system.memory_bus.cpu_side_ports
    system.system_port = system.memory_bus.cpu_side_ports

    for cpu in system.cpu:
        configure_o3_cpu(cpu)
        cpu.addPrivateSplitL1Caches(
            L1InstructionCache(
                size=args.l1i_size,
                tag_latency=args.l1i_tag_latency,
                data_latency=args.l1i_data_latency,
                response_latency=args.l1i_response_latency,
            ),
            L1DataCache(
                size=args.l1d_size,
                tag_latency=args.l1d_tag_latency,
                data_latency=args.l1d_data_latency,
                response_latency=args.l1d_response_latency,
            ),
        )
        cpu.createInterruptController()
        cpu.connectAllPorts(
            system.to_l2_bus.cpu_side_ports,
            system.memory_bus.cpu_side_ports,
            system.memory_bus.mem_side_ports,
        )

        # One shared Process is deliberate: CPU0 starts it and clone() claims
        # the other three halted contexts for pthread children.
        cpu.workload = process
        cpu.createThreads()

    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR3_1600_8x8()
    system.mem_ctrl.dram.range = system.mem_ranges[0]
    system.mem_ctrl.port = system.memory_bus.mem_side_ports
    system.workload = SEWorkload.init_compatible(process.executable)
    return system


def write_metadata(args, binary, options, stdin_path, environment):
    output_directory = Path(m5.options.outdir)
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": f"CLIP-3D {args.stage}",
        "workload": args.workload,
        "binary": str(binary),
        "command": [str(binary), *options],
        "environment": list(environment),
        "stdin": str(stdin_path) if stdin_path is not None else None,
        "num_cores": NUM_CORES,
        "cpu_type": "X86O3CPU",
        "cpu_clock": args.cpu_clock,
        "issue_width": 4,
        "rob_entries": 192,
        "l1i_size": args.l1i_size,
        "l1d_size": args.l1d_size,
        "l1_associativity": 2,
        "l2_size": args.l2_size,
        "l2_associativity": 8,
        "cache_line_bytes": 64,
        "memory_size": args.mem_size,
        "latencies": {
            "l1i": {"tag": args.l1i_tag_latency, "data": args.l1i_data_latency,
                    "response": args.l1i_response_latency},
            "l1d": {"tag": args.l1d_tag_latency, "data": args.l1d_data_latency,
                    "response": args.l1d_response_latency},
            "l2": {"tag": args.l2_tag_latency, "data": args.l2_data_latency,
                   "response": args.l2_response_latency},
            "xbar": {"frontend": args.xbar_frontend_latency,
                     "forward": args.xbar_forward_latency,
                     "response": args.xbar_response_latency,
                     "snoop_response": args.xbar_snoop_response_latency},
        },
        "warmup_insts_cpu0": args.warmup_insts,
        "measure_insts_cpu0": args.measure_insts,
        "warmup_insts": args.warmup_insts,
        "measure_insts": args.measure_insts,
        "instruction_window_scope": args.instruction_window_scope,
        "stop_anchor": ("CPU0 thread 0" if args.instruction_window_scope == "cpu0"
                        else "all four CPU thread-0 contexts"),
        "thread_mapping": "one process; pthread clone into four CPU contexts",
    }
    with (output_directory / "r1_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


def simulate_phase(expected_cause, max_ticks):
    event = m5.simulate(max_ticks if max_ticks else m5.MaxTick)
    cause = event.getCause()
    print(f"Exit @ tick {m5.curTick()}: {cause}")
    if cause != expected_cause:
        m5.stats.dump()
        raise RuntimeError(
            f"simulation ended before '{expected_cause}'; actual cause: {cause}"
        )


def simulate_all_core_phase(base_cause, max_ticks):
    expected = {f"{base_cause} CPU{core}" for core in range(NUM_CORES)}
    observed = set()
    while observed != expected:
        event = m5.simulate(max_ticks if max_ticks else m5.MaxTick)
        cause = event.getCause()
        print(f"Exit @ tick {m5.curTick()}: {cause}")
        if cause not in expected:
            m5.stats.dump()
            raise RuntimeError(
                f"simulation ended before all cores reached '{base_cause}'; actual cause: {cause}"
            )
        if cause in observed:
            raise RuntimeError(f"duplicate instruction-stop event: {cause}")
        observed.add(cause)


def stats_path():
    configured = Path(m5.options.stats_file)
    if configured.is_absolute():
        return configured
    return Path(m5.options.outdir) / configured


def validate_four_core_activity(path, minimum_instructions=1):
    text = path.read_text()
    patterns = (
        re.compile(r"^system\.cpu([0-3])\.commitStats0\.numInsts\s+([0-9.eE+-]+)", re.M),
        re.compile(r"^system\.cpu([0-3])\.thread_0\.numInsts\s+([0-9.eE+-]+)", re.M),
    )
    counts = {}
    for pattern in patterns:
        for match in pattern.finditer(text):
            counts[int(match.group(1))] = int(float(match.group(2)))
        if len(counts) == NUM_CORES:
            break

    if len(counts) != NUM_CORES:
        raise RuntimeError(
            f"could not find committed-instruction statistics for four cores in {path}"
        )
    print("Measured committed instructions:")
    for core in range(NUM_CORES):
        print(f"  cpu{core}: {counts[core]}")
    inactive = [core for core, count in counts.items() if count < minimum_instructions]
    if inactive:
        raise RuntimeError(
            f"cores below {minimum_instructions} measured instructions: {inactive}"
        )


def main():
    args = parse_arguments()
    process, binary, options, stdin_path = build_process(args)
    system = build_system(args, process)
    root = Root(full_system=False, system=system)
    write_metadata(args, binary, options, stdin_path, process.env)

    m5.instantiate()
    print(
        f"CLIP-3D {args.stage}: {args.workload}, four X86 O3 cores, "
        f"L1D={args.l1d_size}, L2={args.l2_size}"
    )

    if args.warmup_insts:
        if args.instruction_window_scope == "all-cores":
            for core, cpu in enumerate(system.cpu):
                cpu.scheduleInstStop(0, args.warmup_insts, f"{WARMUP_CAUSE} CPU{core}")
            simulate_all_core_phase(WARMUP_CAUSE, args.max_ticks_per_phase)
        else:
            system.cpu[0].scheduleInstStop(0, args.warmup_insts, WARMUP_CAUSE)
            simulate_phase(WARMUP_CAUSE, args.max_ticks_per_phase)

    m5.stats.reset()
    if args.instruction_window_scope == "all-cores":
        for core, cpu in enumerate(system.cpu):
            cpu.scheduleInstStop(0, args.measure_insts, f"{MEASUREMENT_CAUSE} CPU{core}")
        simulate_all_core_phase(MEASUREMENT_CAUSE, args.max_ticks_per_phase)
    else:
        system.cpu[0].scheduleInstStop(0, args.measure_insts, MEASUREMENT_CAUSE)
        simulate_phase(MEASUREMENT_CAUSE, args.max_ticks_per_phase)
    m5.stats.dump()
    validate_four_core_activity(
        stats_path(), args.measure_insts if args.instruction_window_scope == "all-cores" else 1
    )
    print(f"{args.stage} measurement completed; statistics: {stats_path()}")


main()
