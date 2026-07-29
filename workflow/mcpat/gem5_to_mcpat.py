#!/usr/bin/env python3
"""Convert one CLIP-3D gem5 R1 result into a four-core McPAT XML file.

The original paper does not publish its gem5-to-McPAT converter.  This module
therefore keeps every inferred activity mapping in ``mapping_report.json``.
Static structures match configs/gem5/clip_r1.py; dynamic values come from the
last measurement section in stats.txt.
"""

from __future__ import annotations

import argparse
import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from workflow.common import (
    PROJECT_ROOT,
    parse_frequency_ghz,
    parse_gem5_stats,
    parse_size_bytes,
    read_json,
    stat,
    write_json,
)


DEFAULT_TEMPLATE = (
    PROJECT_ROOT / "tools/src/mcpat/ProcessorDescriptionFiles/ARM_A9_2GHz.xml"
)
FP_CLASSES = (
    "FloatAdd", "FloatCmp", "FloatCvt", "FloatMult", "FloatMultAcc",
    "FloatDiv", "FloatMisc", "FloatSqrt", "SimdFloatAdd", "SimdFloatAlu",
    "SimdFloatCmp", "SimdFloatCvt", "SimdFloatDiv", "SimdFloatMisc",
    "SimdFloatMult", "SimdFloatMultAcc", "SimdFloatMatMultAcc",
    "SimdFloatSqrt", "VectorFloatArith", "VectorFloatConvert",
    "VectorFloatReduce",
)
MUL_CLASSES = ("IntMult", "IntDiv", "FloatMult", "FloatMultAcc", "FloatDiv",
               "SimdMult", "SimdMultAcc", "SimdDiv", "SimdFloatMult",
               "SimdFloatMultAcc", "SimdFloatDiv")

# McPAT accepts temperatures only on a 10 K grid.  The paper does not publish
# its XML, but its FFT 64-kB/1-MB anchor (gamma=0.446) is much closer to the
# 320 K HP model than to the former 370 K setting (which had incorrectly been
# inferred from T_safe).  T_safe is a DVFS threshold, not the McPAT operating
# temperature.
DEFAULT_MCPAT_SETTINGS = {
    "temperature_k": 320,
    "device_type": 0,
    "longer_channel_device": 1,
    "interconnect_projection_type": 1,
}


def named_child(component: ET.Element, tag: str, name: str) -> ET.Element:
    for child in component.findall(tag):
        if child.get("name") == name:
            return child
    raise KeyError(f"{component.get('id')} has no {tag} named {name}")


def set_named(component: ET.Element, tag: str, name: str, value: object) -> None:
    named_child(component, tag, name).set("value", str(value))


def component_by_id(root: ET.Element, component_id: str) -> ET.Element:
    for component in root.iter("component"):
        if component.get("id") == component_id:
            return component
    raise KeyError(f"missing McPAT component {component_id}")


def replace_core_identity(component: ET.Element, core: int) -> None:
    for item in component.iter("component"):
        identifier = item.get("id", "")
        item.set("id", re.sub(r"system\.core0", f"system.core{core}", identifier))
    component.set("name", f"core{core}")


def instruction_class(stats: dict[str, float], core: int, names: tuple[str, ...]) -> int:
    prefix = f"system.cpu{core}.commitStats0.committedInstType::"
    return int(sum(stat(stats, prefix + name) for name in names))


def cache_counts(stats: dict[str, float], core: int, cache: str) -> dict[str, int]:
    prefix = f"system.cpu{core}.{cache}."
    reads = stat(stats, prefix + "ReadReq.accesses::total")
    writes = stat(stats, prefix + "WriteReq.accesses::total")
    accesses = stat(stats, prefix + "overallAccesses::total")
    if reads + writes == 0:
        reads = accesses if cache == "icache" else accesses * 0.7
        writes = 0 if cache == "icache" else accesses - reads
    read_misses = stat(stats, prefix + "ReadReq.misses::total")
    write_misses = stat(stats, prefix + "WriteReq.misses::total")
    misses = stat(stats, prefix + "overallMisses::total")
    if read_misses + write_misses == 0 and misses:
        denom = reads + writes
        read_misses = misses * reads / denom if denom else misses
        write_misses = misses - read_misses
    return {"reads": int(reads), "writes": int(writes),
            "read_misses": int(read_misses), "write_misses": int(write_misses)}


def update_core(core_xml: ET.Element, core: int, metadata: dict, stats: dict) -> dict:
    cycles = int(stat(stats, f"system.cpu{core}.numCycles"))
    committed = int(stat(stats, f"system.cpu{core}.commitStats0.numInsts"))
    # gem5 v23 calls this counter numOps.  The old opsCommitted spelling never
    # existed in these R1 files and silently fell back to macro-instructions.
    micro_ops = int(stat(stats, f"system.cpu{core}.commitStats0.numOps", committed))
    if micro_ops <= 0:
        micro_ops = int(stat(stats, f"system.cpu{core}.commitStats0.committedInstType::total", committed))
    loads = instruction_class(stats, core, ("MemRead", "FloatMemRead", "VectorUnitStrideLoad",
                                              "VectorStridedLoad", "VectorIndexedLoad"))
    stores = instruction_class(stats, core, ("MemWrite", "FloatMemWrite", "VectorUnitStrideStore",
                                               "VectorStridedStore", "VectorIndexedStore"))
    fp = instruction_class(stats, core, FP_CLASSES)
    mul = instruction_class(stats, core, MUL_CLASSES)
    branches = int(stat(stats, f"system.cpu{core}.commitStats0.committedControl::total",
                        stat(stats, f"system.cpu{core}.branchPred.condPredicted")))
    branch_misses = int(stat(stats, f"system.cpu{core}.iew.branchMispredicts",
                             stat(stats, f"system.cpu{core}.branchPred.condIncorrect")))
    integer = max(micro_ops - fp, 0)
    ialu = instruction_class(
        stats, core,
        ("IntAlu", "SimdAdd", "SimdAddAcc", "SimdAlu", "SimdCmp",
         "SimdCvt", "SimdMisc", "SimdShift", "SimdShiftAcc"),
    )

    # Prefer counters emitted directly by gem5's O3CPU.  Formula fallbacks are
    # retained for other compatible gem5 releases and are listed in the
    # mapping report below rather than being hidden assumptions.
    rob_reads = int(stat(stats, f"system.cpu{core}.rob.reads", micro_ops))
    rob_writes = int(stat(stats, f"system.cpu{core}.rob.writes", micro_ops))
    rename_total_reads = int(stat(
        stats, f"system.cpu{core}.rename.lookups", micro_ops * 2
    ))
    fp_rename_reads = int(stat(
        stats, f"system.cpu{core}.rename.fpLookups", fp * 2
    ))
    rename_reads = max(rename_total_reads - fp_rename_reads, 0)
    rename_total_writes = int(stat(
        stats, f"system.cpu{core}.rename.renamedOperands", micro_ops
    ))
    fp_rename_writes = int(stat(
        stats, f"system.cpu{core}.executeStats0.numFpRegWrites", fp
    ))
    rename_writes = max(rename_total_writes - fp_rename_writes, 0)
    dispatched = int(stat(stats, f"system.cpu{core}.iew.dispatchedInsts", micro_ops))
    wakeups = int(stat(stats, f"system.cpu{core}.iew.consumerInst", micro_ops * 2))
    fp_dispatch = min(fp, dispatched)
    fp_wakeups = min(fp * 2, wakeups)
    int_reg_reads = int(stat(
        stats, f"system.cpu{core}.executeStats0.numIntRegReads", integer * 2
    ))
    fp_reg_reads = int(stat(
        stats, f"system.cpu{core}.executeStats0.numFpRegReads", fp * 2
    ))
    int_reg_writes = int(stat(
        stats, f"system.cpu{core}.executeStats0.numIntRegWrites", integer
    ))
    fp_reg_writes = int(stat(
        stats, f"system.cpu{core}.executeStats0.numFpRegWrites", fp
    ))
    busy = max(cycles - int(stat(stats, f"system.cpu{core}.idleCycles")), 0)

    static_params = {
        "clock_rate": int(round(parse_frequency_ghz(metadata["cpu_clock"]) * 1000)),
        "instruction_length": 32, "opcode_width": 16, "x86": 1,
        "machine_type": 0, "number_hardware_threads": 1,
        "fetch_width": metadata.get("issue_width", 4),
        "decode_width": metadata.get("issue_width", 4),
        "issue_width": metadata.get("issue_width", 4),
        "peak_issue_width": metadata.get("issue_width", 4),
        "commit_width": metadata.get("issue_width", 4),
        "instruction_window_size": 64, "fp_instruction_window_size": 64,
        "ROB_size": metadata.get("rob_entries", 192),
        "phy_Regs_IRF_size": 256, "phy_Regs_FRF_size": 256,
        "store_buffer_size": 96, "load_buffer_size": 48, "memory_ports": 2,
    }
    for name, value in static_params.items():
        set_named(core_xml, "param", name, value)

    core_stats = {
        "total_instructions": micro_ops, "int_instructions": integer,
        "fp_instructions": fp, "branch_instructions": branches,
        "branch_mispredictions": branch_misses, "load_instructions": loads,
        "store_instructions": stores, "committed_instructions": micro_ops,
        "committed_int_instructions": integer, "committed_fp_instructions": fp,
        "pipeline_duty_cycle": min(micro_ops / max(cycles * metadata.get("issue_width", 4), 1), 1.0),
        "total_cycles": cycles, "idle_cycles": max(cycles - busy, 0), "busy_cycles": busy,
        "ROB_reads": rob_reads, "ROB_writes": rob_writes,
        "rename_reads": rename_reads, "rename_writes": rename_writes,
        "fp_rename_reads": fp_rename_reads, "fp_rename_writes": fp_rename_writes,
        "inst_window_reads": max(dispatched - fp_dispatch, 0),
        "inst_window_writes": max(dispatched - fp_dispatch, 0),
        "inst_window_wakeup_accesses": max(wakeups - fp_wakeups, 0),
        "fp_inst_window_reads": fp_dispatch, "fp_inst_window_writes": fp_dispatch,
        "fp_inst_window_wakeup_accesses": fp_wakeups,
        "int_regfile_reads": int_reg_reads, "float_regfile_reads": fp_reg_reads,
        "int_regfile_writes": int_reg_writes, "float_regfile_writes": fp_reg_writes,
        "function_calls": int(stat(stats, f"system.cpu{core}.commitStats0.committedControl::IsCall")),
        "context_switches": 0, "ialu_accesses": ialu,
        "fpu_accesses": fp, "mul_accesses": mul,
        "cdb_alu_accesses": ialu, "cdb_mul_accesses": mul,
        "cdb_fpu_accesses": fp,
    }
    for name, value in core_stats.items():
        set_named(core_xml, "stat", name, value)

    # McPAT's CACTI-P rejects a one-cycle 45-nm cache organization at 2 GHz.
    # These fields constrain physical synthesis, not the R1 ideal gem5 timing;
    # CACTI supplies the timing that is later back-annotated into R2.
    throughput_cycles, latency_cycles = 10, 10
    line = int(metadata.get("cache_line_bytes", 64))
    for cache_name, size_key, assoc_key, xml_id in (
        ("icache", "l1i_size", "l1_associativity", "icache"),
        ("dcache", "l1d_size", "l1_associativity", "dcache"),
    ):
        cache = component_by_id(core_xml, f"system.core{core}.{xml_id}")
        cfg_name = f"{xml_id}_config"
        size = parse_size_bytes(metadata[size_key])
        assoc = int(metadata[assoc_key])
        policy = 0 if cache_name == "icache" else 1
        set_named(cache, "param", cfg_name,
                  f"{size},{line},{assoc},1,{throughput_cycles},{latency_cycles},32,{policy}")
        counts = cache_counts(stats, core, cache_name)
        set_named(cache, "stat", "read_accesses", counts["reads"])
        set_named(cache, "stat", "read_misses", counts["read_misses"])
        if cache_name == "dcache":
            set_named(cache, "stat", "write_accesses", counts["writes"])
            set_named(cache, "stat", "write_misses", counts["write_misses"])

    itlb = component_by_id(core_xml, f"system.core{core}.itlb")
    dtlb = component_by_id(core_xml, f"system.core{core}.dtlb")
    icounts = cache_counts(stats, core, "icache")
    dcounts = cache_counts(stats, core, "dcache")
    set_named(itlb, "stat", "total_accesses", icounts["reads"])
    set_named(itlb, "stat", "total_misses", 0)
    set_named(dtlb, "stat", "total_accesses", dcounts["reads"] + dcounts["writes"])
    set_named(dtlb, "stat", "total_misses", 0)
    btb = component_by_id(core_xml, f"system.core{core}.BTB")
    set_named(btb, "stat", "read_accesses", branches)
    set_named(btb, "stat", "write_accesses", branch_misses)
    return {"core": core, "cycles": cycles, "instructions": committed,
            "micro_ops": micro_ops, "loads": loads, "stores": stores,
            "fp_ops": fp, "branch_instructions": branches,
            "branch_mispredictions": branch_misses,
            "direct_o3_counters": {
                "rob_reads": rob_reads, "rob_writes": rob_writes,
                "rename_total_reads": rename_total_reads,
                "rename_fp_reads": fp_rename_reads,
                "rename_total_writes": rename_total_writes,
                "dispatched": dispatched, "window_wakeups": wakeups,
                "int_reg_reads": int_reg_reads, "fp_reg_reads": fp_reg_reads,
                "int_reg_writes": int_reg_writes, "fp_reg_writes": fp_reg_writes,
                "ialu_accesses": ialu,
            },
            "mapping_quality": "direct where gem5 exposes a counter; otherwise documented estimate"}


def l2_counts(stats: dict[str, float]) -> dict[str, int]:
    reads = (stat(stats, "system.l2.ReadReq.accesses::total") +
             stat(stats, "system.l2.ReadSharedReq.accesses::total"))
    writes = (stat(stats, "system.l2.ReadExReq.accesses::total") +
              stat(stats, "system.l2.WriteReq.accesses::total") +
              stat(stats, "system.l2.WritebackDirty.accesses::total"))
    accesses = stat(stats, "system.l2.overallAccesses::total")
    if reads + writes == 0:
        reads, writes = accesses * 0.7, accesses * 0.3
    total_misses = stat(stats, "system.l2.overallMisses::total")
    read_misses = (stat(stats, "system.l2.ReadReq.misses::total") +
                   stat(stats, "system.l2.ReadSharedReq.misses::total"))
    write_misses = max(total_misses - read_misses, 0)
    return {"reads": int(reads), "writes": int(writes),
            "read_misses": int(read_misses), "write_misses": int(write_misses)}


def convert(r1_dir: Path, output_xml: Path, template: Path = DEFAULT_TEMPLATE,
            report_path: Path | None = None, settings: dict | None = None) -> dict:
    mcpat_settings = {**DEFAULT_MCPAT_SETTINGS, **(settings or {})}
    temperature = int(mcpat_settings["temperature_k"])
    if temperature % 10 or not 300 <= temperature <= 400:
        raise ValueError("McPAT temperature_k must be 300..400 K in 10 K steps")
    metadata = read_json(r1_dir / "r1_metadata.json")
    stats = parse_gem5_stats(r1_dir / "stats.txt")
    cores = int(metadata.get("num_cores", 4))
    if cores != 4:
        raise ValueError(f"CLIP-3D reproduction requires four cores, got {cores}")
    tree = ET.parse(template)
    root = tree.getroot()
    system = component_by_id(root, "system")
    system_values = {
        "number_of_cores": cores, "number_of_L1Directories": 0,
        "number_of_L2Directories": 0, "number_of_L2s": 1,
        "Private_L2": 0, "number_of_L3s": 0, "number_of_NoCs": 1,
        "homogeneous_cores": 0, "homogeneous_L2s": 1,
        "core_tech_node": 45,
        "target_core_clockrate": int(round(parse_frequency_ghz(metadata["cpu_clock"]) * 1000)),
        "temperature": temperature, "number_cache_levels": 2,
        "device_type": int(mcpat_settings["device_type"]),
        "longer_channel_device": int(mcpat_settings["longer_channel_device"]),
        "interconnect_projection_type": int(mcpat_settings["interconnect_projection_type"]),
        "Embedded": 0, "machine_bits": 32, "virtual_address_width": 32,
        "physical_address_width": 32,
    }
    for name, value in system_values.items():
        set_named(system, "param", name, value)
    cycles = max(int(stat(stats, f"system.cpu{i}.numCycles")) for i in range(cores))
    set_named(system, "stat", "total_cycles", cycles)
    set_named(system, "stat", "idle_cycles", 0)
    set_named(system, "stat", "busy_cycles", cycles)

    original_core = component_by_id(system, "system.core0")
    insertion_index = list(system).index(original_core)
    system.remove(original_core)
    mappings = []
    for core_index in range(cores):
        core_xml = copy.deepcopy(original_core)
        replace_core_identity(core_xml, core_index)
        mappings.append(update_core(core_xml, core_index, metadata, stats))
        system.insert(insertion_index + core_index, core_xml)

    l2 = component_by_id(system, "system.L20")
    line = int(metadata.get("cache_line_bytes", 64))
    l2_size = parse_size_bytes(metadata["l2_size"])
    assoc = int(metadata["l2_associativity"])
    set_named(l2, "param", "L2_config",
              f"{l2_size},{line},{assoc},8,8,23,32,1")
    set_named(l2, "param", "clockrate", system_values["target_core_clockrate"])
    counts = l2_counts(stats)
    for xml_name, key in (("read_accesses", "reads"), ("write_accesses", "writes"),
                          ("read_misses", "read_misses"), ("write_misses", "write_misses")):
        set_named(l2, "stat", xml_name, counts[key])
    noc = component_by_id(system, "system.NoC0")
    set_named(noc, "param", "clockrate", system_values["target_core_clockrate"])
    for name in ("total_accesses",):
        try:
            set_named(noc, "stat", name, counts["reads"] + counts["writes"])
        except KeyError:
            pass

    ET.indent(tree, space="  ")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    report = {
        "schema_version": 1,
        "source_r1": str(r1_dir.resolve()),
        "output_xml": str(output_xml.resolve()),
        "technology_nm": 45,
        "mcpat_settings": mcpat_settings,
        "cores": mappings,
        "l2": counts,
        "paper_parameters": ["4 cores", "2 GHz", "45 nm", "L1 assoc=2", "L2 assoc=8"],
        "reproduction_assumptions": [
            "ARM_A9 XML is used only as a complete McPAT schema and is rewritten to an x86 four-wide OOO core.",
            "x86 instruction activity uses gem5 committed numOps, as required by the McPAT XML schema.",
            "ROB, rename, dispatch, wakeup, and physical-register activity prefer direct O3CPU counters.",
            "gem5 counters without a one-to-one McPAT field are estimated from committed micro-op classes.",
            "TLB misses are zero because the current SE statistics do not expose a stable per-TLB miss counter.",
            "The schema's 32-bit address widths are retained: this McPAT/CACTI-P build reports no valid array organization with 64-bit widths.",
            "McPAT temperature is an explicitly configured power-model operating point and is not T_safe.",
        ],
    }
    write_json(report_path or output_xml.with_name("mapping_report.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--temperature-k", type=int, default=DEFAULT_MCPAT_SETTINGS["temperature_k"])
    parser.add_argument("--device-type", type=int, default=DEFAULT_MCPAT_SETTINGS["device_type"])
    parser.add_argument("--longer-channel-device", type=int,
                        default=DEFAULT_MCPAT_SETTINGS["longer_channel_device"])
    parser.add_argument("--interconnect-projection-type", type=int,
                        default=DEFAULT_MCPAT_SETTINGS["interconnect_projection_type"])
    args = parser.parse_args()
    settings = {
        "temperature_k": args.temperature_k, "device_type": args.device_type,
        "longer_channel_device": args.longer_channel_device,
        "interconnect_projection_type": args.interconnect_projection_type,
    }
    convert(args.r1_dir.resolve(), args.output.resolve(), args.template.resolve(),
            args.report.resolve() if args.report else None, settings)
    print(f"McPAT XML written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
