#!/usr/bin/env python3
"""Parse detailed McPAT text into CLIP-3D module power/area JSON."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from workflow.common import write_json


METRIC_RE = re.compile(
    r"^\s*(Area|Runtime Dynamic|Subthreshold Leakage|Gate Leakage)\s*=\s*"
    r"([0-9.eE+-]+)\s*(mm\^2|W)", re.M
)


def resolve_power_calibration(mcpat_config: dict, workload: str) -> dict:
    """Resolve a documented default calibration plus an optional workload fit.

    McPAT XML details are not published by CLIP-3D.  A single global scale is
    therefore the preferred default, while Table-III anchor experiments may
    supply an explicit per-workload override.  Keeping the merge here makes
    the selected provenance visible in every mcpat.json instead of silently
    changing power later in the thermal flow.
    """
    base = dict(mcpat_config.get("power_calibration", {}))
    overrides = base.pop("by_workload", {})
    if overrides and not isinstance(overrides, dict):
        raise ValueError("mcpat.power_calibration.by_workload must be an object")
    selected = overrides.get(workload, {}) if overrides else {}
    if selected and not isinstance(selected, dict):
        raise ValueError(f"power calibration for {workload!r} must be an object")
    result = {**base, **selected}
    selection = {
        "workload": workload,
        "used_workload_override": bool(selected),
        "available_workload_overrides": sorted(overrides),
    }
    provenance = dict(result.get("provenance") or {"kind": "unspecified"})
    provenance["selection"] = selection
    result["provenance"] = provenance
    result["selection"] = selection
    return result


def metrics(text: str) -> dict[str, float]:
    found = {}
    for name, value, _unit in METRIC_RE.findall(text):
        found.setdefault(name, float(value))
    required = {"Area", "Runtime Dynamic", "Subthreshold Leakage", "Gate Leakage"}
    missing = required - found.keys()
    if missing:
        raise ValueError(f"McPAT block lacks metrics: {sorted(missing)}")
    leakage = found["Subthreshold Leakage"] + found["Gate Leakage"]
    return {
        "area_mm2": found["Area"],
        "dynamic_power_w": found["Runtime Dynamic"],
        "subthreshold_leakage_w": found["Subthreshold Leakage"],
        "gate_leakage_w": found["Gate Leakage"],
        "leakage_power_w": leakage,
        "total_power_w": found["Runtime Dynamic"] + leakage,
    }


def top_level_metrics(text: str) -> dict[str, float]:
    """Parse a McPAT functional block, treating omitted zero leakage as zero."""
    found = {}
    for name, value, _unit in METRIC_RE.findall(text):
        found.setdefault(name, float(value))
    required = {"Area", "Runtime Dynamic"}
    missing = required - found.keys()
    if missing:
        raise ValueError(f"McPAT block lacks metrics: {sorted(missing)}")
    subthreshold = found.get("Subthreshold Leakage", 0.0)
    gate = found.get("Gate Leakage", 0.0)
    leakage = subthreshold + gate
    return {
        "area_mm2": found["Area"],
        "dynamic_power_w": found["Runtime Dynamic"],
        "subthreshold_leakage_w": subthreshold,
        "gate_leakage_w": gate,
        "leakage_power_w": leakage,
        "total_power_w": found["Runtime Dynamic"] + leakage,
    }


def first_heading_block(text: str, heading: str) -> str:
    match = re.search(rf"^\s*{re.escape(heading)}:\s*$", text, re.M)
    if not match:
        raise ValueError(f"McPAT output lacks {heading!r}")
    tail = text[match.end():]
    next_heading = re.search(r"^\s*[A-Za-z][^=\n]*:\s*$", tail, re.M)
    return tail[:next_heading.start()] if next_heading else tail


def subtract(base: dict[str, float], children: list[dict[str, float]]) -> dict[str, float]:
    """Subtract printed McPAT components without breaking power conservation."""
    result = dict(base)
    original_fields = (
        "area_mm2", "dynamic_power_w", "subthreshold_leakage_w",
        "gate_leakage_w", "leakage_power_w", "total_power_w",
    )
    primitive_fields = (
        "area_mm2", "dynamic_power_w", "subthreshold_leakage_w",
        "gate_leakage_w",
    )
    raw_residuals = {
        field: base[field] - sum(child[field] for child in children)
        for field in original_fields
    }
    clipped_negative_magnitudes = {}
    for field in primitive_fields:
        raw = raw_residuals[field]
        result[field] = max(raw, 0.0)
        clipped_negative_magnitudes[field] = max(-raw, 0.0)
    result["leakage_power_w"] = (
        result["subthreshold_leakage_w"] + result["gate_leakage_w"]
    )
    result["total_power_w"] = (
        result["dynamic_power_w"] + result["leakage_power_w"]
    )
    result["subtraction_diagnostics"] = {
        "raw_residuals": raw_residuals,
        "clipped_negative_magnitudes": clipped_negative_magnitudes,
    }
    return result


def granular_core_logic(chunk: str, core_total: dict[str, float],
                        icache: dict[str, float], dcache: dict[str, float],
                        core_index: int) -> list[dict] | None:
    """Return non-overlapping top-level McPAT core blocks when available.

    Detailed McPAT output contains workload-dependent power for IFU, rename,
    LSU, MMU, and execution blocks.  Collapsing them into one uniform rectangle
    erases the spatial power density consumed by HotSpot.  Cache power is
    subtracted from IFU/LSU because L1I/L1D remain separate CACTI-sized blocks.
    Synthetic/minimal McPAT text lacks these headings and retains the legacy
    one-block fallback.
    """
    headings = (
        ("ifu", "core_ifu", "Instruction Fetch Unit"),
        ("rename", "core_rename", "Renaming Unit"),
        ("lsu", "core_lsu", "Load Store Unit"),
        ("mmu", "core_mmu", "Memory Management Unit"),
        ("exec", "core_exec", "Execution Unit"),
    )
    try:
        values = {
            short: top_level_metrics(first_heading_block(chunk, heading))
            for short, _kind, heading in headings
        }
    except ValueError:
        return None
    values["ifu"] = subtract(values["ifu"], [icache])
    values["lsu"] = subtract(values["lsu"], [dcache])
    pieces = []
    for short, kind, _heading in headings:
        pieces.append({
            "name": f"core{core_index}_{short}", "kind": kind,
            "core": core_index, **values[short],
        })
    logic_total = subtract(core_total, [icache, dcache])
    remainder = subtract(logic_total, [values[short] for short, _, _ in headings])
    if remainder["area_mm2"] > 1e-12 or remainder["total_power_w"] > 1e-12:
        pieces.append({
            "name": f"core{core_index}_other", "kind": "core_other",
            "core": core_index, **remainder,
        })
    return pieces


def apply_power_calibration(result: dict, dynamic_scale: float = 1.0,
                            leakage_scale: float = 1.0,
                            provenance: dict | None = None) -> dict:
    """Apply one documented global calibration to every parsed module.

    The paper does not publish its McPAT XML.  A calibration is therefore
    allowed only as an explicit experiment setting; raw values are retained
    next to calibrated values so it cannot silently turn into a fitting knob.
    """
    if dynamic_scale <= 0 or leakage_scale <= 0:
        raise ValueError("McPAT power calibration scales must be positive")

    def scale(item: dict) -> None:
        item["raw_power"] = {
            "dynamic_power_w": item["dynamic_power_w"],
            "subthreshold_leakage_w": item["subthreshold_leakage_w"],
            "gate_leakage_w": item["gate_leakage_w"],
            "leakage_power_w": item["leakage_power_w"],
            "total_power_w": item["total_power_w"],
        }
        item["dynamic_power_w"] *= dynamic_scale
        item["subthreshold_leakage_w"] *= leakage_scale
        item["gate_leakage_w"] *= leakage_scale
        item["leakage_power_w"] = (
            item["subthreshold_leakage_w"] + item["gate_leakage_w"]
        )
        item["total_power_w"] = item["dynamic_power_w"] + item["leakage_power_w"]

    scale(result["processor"])
    for module in result["modules"]:
        scale(module)
    result["raw_module_totals"] = dict(result["module_totals"])
    result["module_totals"] = {
        key: sum(module[key] for module in result["modules"])
        for key in ("area_mm2", "dynamic_power_w", "leakage_power_w", "total_power_w")
    }
    result["checks"]["module_power_minus_processor_w"] = (
        result["module_totals"]["total_power_w"] - result["processor"]["total_power_w"]
    )
    result["power_calibration"] = {
        "dynamic_scale": dynamic_scale,
        "leakage_scale": leakage_scale,
        "provenance": provenance or {"kind": "unspecified"},
        "raw_values_retained": True,
    }
    return result


def parse_mcpat_text(text: str) -> dict:
    technology = re.search(r"Technology\s+([0-9.]+)\s+nm", text)
    clock = re.search(r"Core clock Rate\(MHz\)\s+([0-9.]+)", text)
    processor_match = re.search(r"^Processor:\s*$", text, re.M)
    if not processor_match:
        raise ValueError("not a complete McPAT result")
    processor = metrics(text[processor_match.end():])
    chunks = re.split(r"^\*{20,}\s*$", text, flags=re.M)
    core_chunks = [chunk for chunk in chunks if re.search(r"^Core:\s*$", chunk, re.M)]
    if not core_chunks:
        raise ValueError("McPAT must be run with -print_level 5 to expose core submodules")

    modules = []
    for index, chunk in enumerate(core_chunks):
        core_total = metrics(chunk[re.search(r"^Core:\s*$", chunk, re.M).end():])
        icache = metrics(first_heading_block(chunk, "Instruction Cache"))
        dcache = metrics(first_heading_block(chunk, "Data Cache"))
        logic_blocks = granular_core_logic(chunk, core_total, icache, dcache, index)
        if logic_blocks is None:
            logic_blocks = [{
                "name": f"core{index}_logic", "kind": "core_logic", "core": index,
                **subtract(core_total, [icache, dcache]),
            }]
        modules.extend(logic_blocks)
        modules.extend((
            {"name": f"core{index}_l1i", "kind": "l1i", "core": index, **icache},
            {"name": f"core{index}_l1d", "kind": "l1d", "core": index, **dcache},
        ))

    l2_chunks = [chunk for chunk in chunks if re.search(r"^L2\s*$", chunk, re.M)]
    if not l2_chunks:
        raise ValueError("detailed McPAT output lacks shared L2 section")
    modules.append({"name": "shared_l2", "kind": "l2", **metrics(l2_chunks[0])})
    bus_chunks = [chunk for chunk in chunks if re.search(r"^BUSES\s*$", chunk, re.M)]
    if bus_chunks:
        modules.append({"name": "noc", "kind": "interconnect", **metrics(bus_chunks[0])})

    totals = {
        key: sum(module[key] for module in modules)
        for key in ("area_mm2", "dynamic_power_w", "leakage_power_w", "total_power_w")
    }
    return {
        "schema_version": 1,
        "technology_nm": float(technology.group(1)) if technology else None,
        "clock_mhz": float(clock.group(1)) if clock else None,
        "processor": processor,
        "modules": modules,
        "module_totals": totals,
        "checks": {
            "core_count": len(core_chunks),
            "core_logic_granularity": (
                "McPAT top-level functional blocks"
                if any(module["kind"] == "core_exec" for module in modules)
                else "legacy aggregate core_logic fallback"
            ),
            "module_power_minus_processor_w": totals["total_power_w"] - processor["total_power_w"],
            "note": "Processor totals may contain components intentionally omitted from the physical module list.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = parse_mcpat_text(args.input.read_text(encoding="utf-8"))
    write_json(args.output, result)
    print(f"Parsed {len(result['modules'])} physical modules: {args.output}")


if __name__ == "__main__":
    main()
