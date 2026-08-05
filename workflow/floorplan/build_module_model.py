#!/usr/bin/env python3
"""Combine gem5 metadata/statistics and parsed McPAT modules."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from workflow.common import (
    aggregate_ipc,
    parse_gem5_stats,
    parse_size_bytes,
    read_json,
    write_json,
)


# The reference pre-scale area combines McPAT non-cache logic with local CACTI
# cache areas for the paper reference architecture (4 cores, 32-kB L1I/L1D,
# 512-kB shared L2).  Cache area and delay must come from the local CACTI run,
# not from paper tables.
DEFAULT_REFERENCE_RAW_AREA_MM2 = 45.7538495872
DEFAULT_AREA_SCALE = 150.0 / DEFAULT_REFERENCE_RAW_AREA_MM2


def extract_communication_profile(stats: dict[str, float], num_cores: int,
                                  stats_path: Path,
                                  instruction_window_scope: str,
                                  required: bool = False) -> dict:
    """Build an auditable shared-L2 demand-access profile for represented CPUs."""
    if num_cores <= 0:
        raise ValueError("communication profile requires at least one core")
    per_core: dict[str, dict] = {}
    diagnostics: list[str] = []
    raw_counts: list[float | None] = []
    missing_cores: list[int] = []
    for core in range(num_cores):
        candidates = [
            f"system.l2.demandAccesses::cpu{core}.data",
            f"system.l2.demandAccesses::cpu{core}.inst",
        ]
        matched = [name for name in candidates if name in stats]
        if not matched:
            missing_cores.append(core)
            diagnostics.append(f"missing shared-L2 demand counter for core {core}")
            raw_count: float | None = 0.0
        else:
            values = [float(stats[name]) for name in matched]
            nonfinite = [name for name, value in zip(matched, values)
                         if not math.isfinite(value)]
            negative = [name for name, value in zip(matched, values) if value < 0]
            if nonfinite:
                diagnostics.extend(
                    f"non-finite shared-L2 demand counter: {name}"
                    for name in nonfinite
                )
                raw_count = None
            else:
                raw_count = sum(values)
            diagnostics.extend(
                f"negative shared-L2 demand counter: {name}" for name in negative
            )
        raw_counts.append(raw_count)
        per_core[str(core)] = {
            "matched_counters": matched,
            "raw_demand_accesses": raw_count,
            "normalized_weight": None,
        }

    total = None if any(value is None for value in raw_counts) else sum(
        float(value) for value in raw_counts if value is not None
    )
    if total is not None and total <= 0:
        diagnostics.append("total demand accesses must be positive")
    status = "unavailable" if diagnostics else "available"
    if status == "available":
        assert total is not None and total > 0
        for record in per_core.values():
            record["normalized_weight"] = record["raw_demand_accesses"] / total

    profile = {
        "status": status,
        "source_stats": str(stats_path.resolve()),
        "instruction_window_scope": instruction_window_scope,
        "counter_family": "system.l2.demandAccesses per CPU requestor",
        "per_core": per_core,
        "total_demand_accesses": total,
        "missing_cores": missing_cores,
        "diagnostics": diagnostics,
    }
    if required and status != "available":
        raise ValueError(
            "communication profile unavailable: " + "; ".join(diagnostics)
        )
    return profile


def cache_record(cacti: dict, level: str, size: str) -> dict:
    wanted = parse_size_bytes(size)
    matches = [
        record for record in cacti["records"]
        if record["level"] == level and int(record["size_bytes"]) == wanted
    ]
    if len(matches) != 1:
        raise KeyError(f"CACTI table has {len(matches)} matches for {level} {size}")
    return matches[0]


def apply_physical_areas(modules: list[dict], metadata: dict, cacti: dict,
                         area_scale: float) -> list[dict]:
    """Use CACTI cache geometry and McPAT non-cache area, then scale globally."""
    if area_scale <= 0:
        raise ValueError("area scale must be positive")
    cache_geometry = {
        "l1i": cache_record(cacti, "l1d", metadata["l1i_size"]),
        "l1d": cache_record(cacti, "l1d", metadata["l1d_size"]),
        "l2": cache_record(cacti, "l2", metadata["l2_size"]),
    }
    dimension_scale = area_scale ** 0.5
    result = []
    for source in modules:
        module = dict(source)
        module["raw_area_mm2"] = float(source["area_mm2"])
        geometry = cache_geometry.get(module["kind"])
        if geometry is None:
            pre_scale_area = module["raw_area_mm2"]
            module["area_source"] = "McPAT"
        else:
            pre_scale_area = float(geometry["area_mm2"])
            module["area_source"] = geometry.get("value_source", "CACTI")
            module["cacti_level"] = geometry["level"]
            module["cacti_size"] = geometry["size"]
            module["preferred_width_mm"] = (
                float(geometry["width_mm"]) * dimension_scale
            )
            module["preferred_height_mm"] = (
                float(geometry["height_mm"]) * dimension_scale
            )
        module["area_before_global_scale_mm2"] = pre_scale_area
        module["area_mm2"] = pre_scale_area * area_scale
        module["power_density_w_per_mm2"] = (
            module["total_power_w"] / module["area_mm2"]
        )
        result.append(module)
    return result


def build_model(r1_dir: Path, mcpat_json: Path, cacti_json: Path, output: Path,
                area_scale: float = DEFAULT_AREA_SCALE,
                require_communication_profile: bool = False) -> dict:
    metadata = read_json(r1_dir / "r1_metadata.json")
    stats_path = r1_dir / "stats.txt"
    stats = parse_gem5_stats(stats_path)
    communication_stats = parse_gem5_stats(stats_path, include_nonfinite=True)
    num_cores = int(metadata.get("num_cores", 4))
    communication_profile = extract_communication_profile(
        communication_stats, num_cores, stats_path,
        str(metadata.get("instruction_window_scope", "not recorded")),
        require_communication_profile,
    )
    mcpat = read_json(mcpat_json)
    cacti = read_json(cacti_json)
    modules = apply_physical_areas(mcpat["modules"], metadata, cacti, area_scale)
    totals = {
        "area_mm2": sum(module["area_mm2"] for module in modules),
        "dynamic_power_w": sum(module["dynamic_power_w"] for module in modules),
        "leakage_power_w": sum(module["leakage_power_w"] for module in modules),
        "total_power_w": sum(module["total_power_w"] for module in modules),
    }
    kinds = sorted({module["kind"] for module in modules})
    by_kind = {
        kind: {
            "count": sum(module["kind"] == kind for module in modules),
            "area_mm2": sum(
                module["area_mm2"] for module in modules if module["kind"] == kind
            ),
            "total_power_w": sum(
                module["total_power_w"] for module in modules if module["kind"] == kind
            ),
        }
        for kind in kinds
    }
    for values in by_kind.values():
        values["power_fraction"] = values["total_power_w"] / totals["total_power_w"]
    result = {
        "schema_version": 1,
        "source_r1": str(r1_dir.resolve()),
        "source_mcpat": str(mcpat_json.resolve()),
        "source_cacti": str(cacti_json.resolve()),
        "architecture": metadata,
        "ipc1": aggregate_ipc(stats, num_cores),
        "communication_profile": communication_profile,
        "power_provenance": mcpat["power_provenance"],
        "area_calibration": {
            "scale_factor": area_scale,
            "reference_target_mm2": 150.0,
            "reference_raw_mm2": 150.0 / area_scale,
            "reference_architecture": "4 cores, L1D=32kB, L2=512kB, 45nm",
            "pre_scale_sources": {
                "core_logic_and_interconnect": "McPAT",
                "l1i_l1d_l2": "local CACTI run",
            },
            "scope": "global area scale only; power is not scaled",
        },
        "power_distribution": {
            "by_kind": by_kind,
            "movable_kinds": ["l2"],
            "movable_power_w": by_kind.get("l2", {}).get("total_power_w", 0.0),
            "movable_power_fraction": by_kind.get("l2", {}).get("power_fraction", 0.0),
        },
        "modules": modules,
        "totals": totals,
        "gamma": totals["leakage_power_w"] / totals["total_power_w"],
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--mcpat-json", type=Path, required=True)
    parser.add_argument("--cacti-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--area-scale", type=float, default=DEFAULT_AREA_SCALE)
    args = parser.parse_args()
    result = build_model(
        args.r1_dir.resolve(), args.mcpat_json.resolve(),
        args.cacti_json.resolve(), args.output.resolve(), args.area_scale
    )
    print(
        f"Module model: {len(result['modules'])} modules, "
        f"{result['totals']['area_mm2']:.3f} mm^2, "
        f"{result['totals']['total_power_w']:.3f} W"
    )


if __name__ == "__main__":
    main()
