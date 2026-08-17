#!/usr/bin/env python3
"""Combine gem5 metadata/statistics and parsed McPAT modules."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from workflow.common import (
    aggregate_ipc,
    instruction_window_scope,
    parse_gem5_stats,
    parse_size_bytes,
    read_json,
    write_json,
)
from workflow.cache_contract import (
    mcpat_embedded_cache_record_identity,
    validate_cache_contract,
)
from workflow.mcpat.run_mcpat import (
    MCPAT_PROVENANCE_AUTHORITY,
    MCPAT_PROVENANCE_SCHEMA_VERSION,
)


def extract_communication_profile(stats: dict[str, float], num_cores: int,
                                  stats_path: Path,
                                  instruction_window_scope: object,
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


def apply_physical_areas(modules: list[dict], metadata: dict,
                         cacti: dict) -> list[dict]:
    """Use measured CACTI cache geometry and unmodified McPAT logic area."""
    cache_geometry = {
        "l1i": cache_record(cacti, "l1i", metadata["l1i_size"]),
        "l1d": cache_record(cacti, "l1d", metadata["l1d_size"]),
        "l2": cache_record(cacti, "l2", metadata["l2_size"]),
    }
    result = []
    for source in modules:
        module = dict(source)
        mcpat_area = float(source["area_mm2"])
        geometry = cache_geometry.get(module["kind"])
        if geometry is None:
            module["area_mm2"] = mcpat_area
            module["area_source"] = "McPAT"
        else:
            module["mcpat_reported_area_mm2"] = mcpat_area
            module["area_mm2"] = float(geometry["area_mm2"])
            module["area_source"] = geometry.get("value_source", "CACTI")
            module["cacti_level"] = geometry["level"]
            module["cacti_size"] = geometry["size"]
            module["cacti_record_id"] = geometry.get("cacti_record_id")
            module["preferred_width_mm"] = float(geometry["width_mm"])
            module["preferred_height_mm"] = float(geometry["height_mm"])
        module["power_density_w_per_mm2"] = (
            module["total_power_w"] / module["area_mm2"]
        )
        result.append(module)
    return result


def apply_mcpat_cache_geometry(modules: list[dict], cache_records: list[dict]) -> list[dict]:
    """Normalize embedded CACTI-P aspect ratios to McPAT aggregate areas."""
    records = {}
    for record in cache_records:
        if not isinstance(record, dict):
            raise ValueError("McPAT embedded CACTI-P record must be an object")
        try:
            key = (record["cache"], record["core"])
        except KeyError as error:
            raise ValueError("incomplete McPAT embedded CACTI-P record") from error
        if key in records:
            raise ValueError(f"duplicate McPAT embedded CACTI-P cache geometry for {key}")
        identity = mcpat_embedded_cache_record_identity(record)
        if not identity or record.get("record_id") != identity:
            raise ValueError(
                f"McPAT embedded CACTI-P record identity differs for {key}"
            )
        records[key] = record
    result = []
    for source in modules:
        module = dict(source)
        for legacy_field in (
            "source_cacti", "cacti_characterization_id",
            "mcpat_reported_area_mm2",
        ):
            module.pop(legacy_field, None)
        if module.get("kind") not in {"l1i", "l1d", "l2"}:
            module["area_source"] = "McPAT"
            result.append(module)
            continue
        key = (module["kind"], None if module["kind"] == "l2" else module.get("core"))
        try:
            record = records[key]
        except KeyError as error:
            raise ValueError(f"McPAT embedded CACTI-P lacks cache geometry for {key}") from error
        area = float(module["area_mm2"])
        width, height = float(record["width_mm"]), float(record["height_mm"])
        if not all(math.isfinite(value) and value > 0 for value in (area, width, height)):
            raise ValueError(f"invalid McPAT cache geometry for {key}")
        ratio = width / height
        normalized_width = math.sqrt(area * ratio)
        normalized_height = area / normalized_width
        module.update({
            "area_source": "McPAT aggregate Area",
            "raw_array_dimensions_mm": {"width": width, "height": height},
            "normalized_block_dimensions_mm": {
                "width": normalized_width, "height": normalized_height,
            },
            "preferred_width_mm": normalized_width,
            "preferred_height_mm": normalized_height,
            "aspect_ratio": ratio,
            "geometry_formula": (
                "width=sqrt(McPAT_area_mm2*embedded_width_mm/embedded_height_mm); "
                "height=McPAT_area_mm2/width"
            ),
            "embedded_cacti_record_id": record["record_id"],
        })
        result.append(module)
    return result


def validate_mobility_contract(modules: list[dict], expected_cores: int = 4) -> dict:
    """Validate granular fixed modules and identify the sole movable shared L2."""
    if isinstance(expected_cores, bool) or not isinstance(expected_cores, int) \
            or expected_cores <= 0:
        raise ValueError("expected core count must be a positive integer")
    if not isinstance(modules, list) or not modules:
        raise ValueError("module model must contain physical modules")

    names = [module.get("name") for module in modules]
    if any(not isinstance(name, str) or not name for name in names) \
            or len(names) != len(set(names)):
        raise ValueError("physical module names must be non-empty and unique")
    if any(module.get("kind") == "core_logic" for module in modules) or any(
        re.fullmatch(r"core[0-9]+_logic", name) for name in names
    ):
        raise ValueError("aggregate core_logic fallback is not a granular module")

    required_kinds = {
        "core_ifu", "core_rename", "core_lsu", "core_mmu", "core_exec",
        "l1i", "l1d",
    }
    observed_cores = {
        module.get("core") for module in modules if module.get("core") is not None
    }
    required_cores = set(range(expected_cores))
    if observed_cores != required_cores:
        raise ValueError(
            "granular module core count mismatch: "
            f"expected {sorted(required_cores)}, observed {sorted(observed_cores)}"
        )
    for core in sorted(required_cores):
        core_modules = [module for module in modules if module.get("core") == core]
        counts = {
            kind: sum(module.get("kind") == kind for module in core_modules)
            for kind in required_kinds
        }
        invalid = {
            kind: count for kind, count in counts.items() if count != 1
        }
        if invalid:
            raise ValueError(
                f"granular core {core} required-kind count mismatch: {invalid}"
            )
        other_count = sum(
            module.get("kind") == "core_other" for module in core_modules
        )
        if other_count > 1:
            raise ValueError(f"granular core {core} has multiple residual blocks")

    l2_modules = [module for module in modules if module.get("kind") == "l2"]
    if len(l2_modules) != 1 or l2_modules[0].get("name") != "shared_l2":
        raise ValueError("module model requires exactly one shared_l2 cache")

    movable_names = ["shared_l2"]
    fixed_names = [name for name in names if name != "shared_l2"]
    for module in modules:
        module["movable"] = module["name"] == "shared_l2"
    return {
        "schema_version": 1,
        "policy": "fixed granular core/NoC modules; movable shared L2 only",
        "expected_core_count": expected_cores,
        "observed_core_indices": sorted(observed_cores),
        "movable_names": movable_names,
        "fixed_names": fixed_names,
    }


_MCPAT_HASH_FIELDS = (
    "xml_sha256", "mapping_sha256", "output_sha256", "binary_sha256",
    "patch_sha256",
)
_CONSERVATION_FIELDS = (
    "area_mm2", "dynamic_power_w", "subthreshold_leakage_w",
    "gate_leakage_w", "leakage_power_w", "total_power_w",
)
# McPAT prints parent blocks with roughly six significant digits, so a
# child-block sum reconstructed from the same print can differ by ~1e-6
# relative.  Use a print-rounding-aware tolerance that still rejects any real
# conservation break (a missing functional block is >=1% of its parent).
MCPAT_PARENT_REL_TOL = 1e-4
MCPAT_PARENT_ABS_TOL = 1e-9


def _validated_mcpat_provenance(mcpat: dict) -> dict:
    provenance = mcpat.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "schema_version", "authority", "hashes",
    }:
        raise ValueError("McPAT artifact lacks versioned hash provenance")
    if provenance.get("schema_version") != MCPAT_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("McPAT provenance schema_version must be 1")
    if provenance.get("authority") != MCPAT_PROVENANCE_AUTHORITY:
        raise ValueError("McPAT provenance authority is invalid")
    hashes = provenance.get("hashes")
    required_hashes = set(_MCPAT_HASH_FIELDS)
    allowed_hash_sets = (
        required_hashes,
        required_hashes | {"build_provenance_sha256"},
    )
    if not isinstance(hashes, dict) or set(hashes) not in allowed_hash_sets:
        raise ValueError("McPAT provenance hash schema is invalid")
    for field in hashes:
        value = hashes.get(field)
        try:
            valid = (
                isinstance(value, str) and len(value) == 64
                and int(value, 16) >= 0 and set(value) != {"0"}
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"McPAT provenance {field} must be a nonzero SHA-256 hash")
    return {
        "schema_version": MCPAT_PROVENANCE_SCHEMA_VERSION,
        "authority": MCPAT_PROVENANCE_AUTHORITY,
        "hashes": dict(hashes),
    }


def _metric_totals(modules: list[dict]) -> dict:
    return {
        field: sum(float(module[field]) for module in modules)
        for field in _CONSERVATION_FIELDS
    }


def _validate_parent_metrics(metrics: object, label: str) -> dict:
    if not isinstance(metrics, dict) or set(metrics) != set(_CONSERVATION_FIELDS):
        raise ValueError(f"McPAT parent {label} has an invalid metric schema")
    validated = {}
    for field in _CONSERVATION_FIELDS:
        value = metrics[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"McPAT parent {label}.{field} must be finite and nonnegative")
        validated[field] = float(value)
    if not math.isclose(
        validated["leakage_power_w"],
        validated["subthreshold_leakage_w"] + validated["gate_leakage_w"],
        rel_tol=MCPAT_PARENT_REL_TOL, abs_tol=MCPAT_PARENT_ABS_TOL,
    ) or not math.isclose(
        validated["total_power_w"],
        validated["dynamic_power_w"] + validated["leakage_power_w"],
        rel_tol=MCPAT_PARENT_REL_TOL, abs_tol=MCPAT_PARENT_ABS_TOL,
    ):
        raise ValueError(f"McPAT parent {label} has inconsistent derived power")
    return validated


def _compare_parent_to_children(parent: dict, children: list[dict],
                                label: str) -> dict:
    observed = _metric_totals(children)
    residuals = {}
    for field in _CONSERVATION_FIELDS:
        residuals[field] = observed[field] - parent[field]
        if not math.isclose(
            observed[field], parent[field],
            rel_tol=MCPAT_PARENT_REL_TOL, abs_tol=MCPAT_PARENT_ABS_TOL,
        ):
            raise ValueError(
                f"McPAT parent conservation failed for {label}.{field}: "
                f"parent={parent[field]}, children={observed[field]}"
            )
    return residuals


def _validate_core_parent_conservation(mcpat: dict, modules: list[dict]) -> dict:
    evidence = mcpat.get("core_parent_metrics")
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema_version", "authority", "records",
    } or evidence.get("schema_version") != 1 or evidence.get("authority") != (
        "McPAT print-level-5 parent blocks before subtraction"
    ) or not isinstance(evidence.get("records"), list):
        raise ValueError("McPAT artifact lacks versioned parent subtraction evidence")
    records = evidence["records"]
    if len(records) != 4 or any(
        not isinstance(record, dict)
        or isinstance(record.get("core"), bool)
        or not isinstance(record.get("core"), int)
        for record in records
    ) or {record["core"] for record in records} != set(range(4)):
        raise ValueError("McPAT parent subtraction evidence must cover four cores")

    residuals = []
    for core in range(4):
        record = next(record for record in records if record.get("core") == core)
        if set(record) != {
            "core", "core_total", "instruction_fetch_unit", "load_store_unit",
        }:
            raise ValueError(f"McPAT core {core} parent evidence schema is invalid")
        parents = {
            label: _validate_parent_metrics(record[key], f"core{core}.{label}")
            for label, key in (
                ("core_total", "core_total"),
                ("instruction_fetch_unit", "instruction_fetch_unit"),
                ("load_store_unit", "load_store_unit"),
            )
        }
        core_modules = [module for module in modules if module.get("core") == core]
        ifu_children = [
            module for module in core_modules
            if module.get("kind") in {"core_ifu", "l1i"}
        ]
        lsu_children = [
            module for module in core_modules
            if module.get("kind") in {"core_lsu", "l1d"}
        ]
        residuals.append({
            "core": core,
            "core_total": _compare_parent_to_children(
                parents["core_total"], core_modules, f"core{core}.core_total"
            ),
            "instruction_fetch_unit": _compare_parent_to_children(
                parents["instruction_fetch_unit"], ifu_children,
                f"core{core}.instruction_fetch_unit",
            ),
            "load_store_unit": _compare_parent_to_children(
                parents["load_store_unit"], lsu_children,
                f"core{core}.load_store_unit",
            ),
        })
    return {
        "authority": evidence["authority"],
        "serialized_children_minus_parent": residuals,
    }


def _validate_module_conservation(
        mcpat: dict, modules: list[dict], *, require_granular_cores: bool,
) -> tuple[dict, dict]:
    totals = _metric_totals(modules)
    expected = mcpat.get("module_totals")
    if not isinstance(expected, dict):
        raise ValueError("McPAT artifact lacks module_totals conservation evidence")
    residuals = {}
    for field, observed in totals.items():
        try:
            source_value = float(expected[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"McPAT module_totals lacks finite {field}") from error
        if not math.isfinite(source_value):
            raise ValueError(f"McPAT module_totals lacks finite {field}")
        residuals[field] = observed - source_value
        if not math.isclose(observed, source_value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"McPAT module conservation failed for {field}: "
                f"source={source_value}, serialized={observed}"
            )
    result = {
        "status": "passed",
        "source": "McPAT parsed module_totals",
        "relative_tolerance": 1e-12,
        "absolute_tolerance": 1e-12,
        "source_totals": {field: float(expected[field]) for field in _CONSERVATION_FIELDS},
        "serialized_minus_source": residuals,
    }
    if require_granular_cores:
        result["parent_subtraction"] = _validate_core_parent_conservation(
            mcpat, modules
        )
    return totals, result


def _validate_module_power(module: dict) -> None:
    name = module.get("name", "<unnamed>")
    for field in _CONSERVATION_FIELDS:
        try:
            value = float(module[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"McPAT module {name}.{field} must be finite") from error
        if not math.isfinite(value) or value < 0 or (field == "area_mm2" and value <= 0):
            raise ValueError(f"McPAT module {name}.{field} must be finite and nonnegative")
    if not math.isclose(
        float(module["total_power_w"]),
        float(module["dynamic_power_w"]) + float(module["leakage_power_w"]),
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ValueError(f"McPAT module {name} does not conserve total power")
    if "subthreshold_leakage_w" in module or "gate_leakage_w" in module:
        try:
            primitive_leakage = (
                float(module["subthreshold_leakage_w"])
                + float(module["gate_leakage_w"])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"McPAT module {name} has incomplete leakage components"
            ) from error
        if not math.isclose(
            float(module["leakage_power_w"]), primitive_leakage,
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(f"McPAT module {name} does not conserve leakage power")


def construct_model(r1_dir: Path, mcpat_json: Path,
                    require_communication_profile: bool = False,
                    require_granular_cores: bool = True) -> dict:
    """Construct one McPAT-authoritative model without publishing it."""
    r1_dir = Path(r1_dir)
    mcpat_json = Path(mcpat_json)
    metadata = dict(read_json(r1_dir / "r1_metadata.json"))
    metadata["instruction_window_scope"] = instruction_window_scope(metadata)
    stats_path = r1_dir / "stats.txt"
    stats = parse_gem5_stats(stats_path)
    communication_stats = parse_gem5_stats(stats_path, include_nonfinite=True)
    num_cores = int(metadata.get("num_cores", 4))
    communication_profile = extract_communication_profile(
        communication_stats, num_cores, stats_path,
        metadata["instruction_window_scope"],
        require_communication_profile,
    )
    mcpat = read_json(mcpat_json)
    cache_contract = validate_cache_contract(
        mcpat.get("cache_contract"), expected_core_count=4,
    )
    embedded = mcpat.get("embedded_cacti_p") or {}
    if embedded.get("schema_version") != 1 or embedded.get(
            "authority") != "McPAT 1.3 embedded CACTI-P":
        raise ValueError("McPAT artifact lacks embedded CACTI-P authority")
    cache_records = embedded.get("records")
    if not isinstance(cache_records, list):
        raise ValueError("McPAT embedded CACTI-P records must be a list")
    expected_record_keys = {
        *(("l1i", core) for core in range(4)),
        *(("l1d", core) for core in range(4)),
        ("l2", None),
    }
    observed_record_keys = [
        (record.get("cache"), record.get("core"))
        for record in cache_records if isinstance(record, dict)
    ]
    if len(observed_record_keys) != len(cache_records) \
            or len(observed_record_keys) != len(set(observed_record_keys)) \
            or set(observed_record_keys) != expected_record_keys:
        raise ValueError("McPAT embedded CACTI-P record set is not the exact four-core set")
    provenance = _validated_mcpat_provenance(mcpat)
    modules = apply_mcpat_cache_geometry(mcpat.get("modules") or [], cache_records)
    for module in modules:
        _validate_module_power(module)
        module["power_density_w_per_mm2"] = (
            float(module["total_power_w"]) / float(module["area_mm2"])
        )
    if require_granular_cores:
        if num_cores != 4:
            raise ValueError(
                f"formal granular model requires four cores, observed {num_cores}"
            )
        checks = mcpat.get("checks")
        if not isinstance(checks, dict) or checks.get("core_count") != 4 \
                or checks.get("core_logic_granularity") != (
                    "McPAT top-level functional blocks"
                ):
            raise ValueError(
                "strict McPAT artifact lacks four-core granular parser checks"
            )
        mobility_contract = validate_mobility_contract(modules, expected_cores=4)
    else:
        l2_names = [module["name"] for module in modules if module.get("kind") == "l2"]
        if l2_names != ["shared_l2"]:
            raise ValueError("module model requires exactly one shared_l2 cache")
        for module in modules:
            module["movable"] = module["name"] == "shared_l2"
        mobility_contract = {
            "schema_version": 1,
            "policy": "legacy diagnostic; movable shared L2 only",
            "expected_core_count": num_cores,
            "observed_core_indices": sorted({
                module["core"] for module in modules if module.get("core") is not None
            }),
            "movable_names": ["shared_l2"],
            "fixed_names": [
                module["name"] for module in modules if module["name"] != "shared_l2"
            ],
        }
    totals, conservation = _validate_module_conservation(
        mcpat, modules, require_granular_cores=require_granular_cores,
    )
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
        "schema_version": 3,
        "source_r1": str(r1_dir.resolve()),
        "source_mcpat": str(mcpat_json.resolve()),
        "architecture": metadata,
        "ipc1": aggregate_ipc(stats, num_cores),
        "communication_profile": communication_profile,
        "power_provenance": mcpat["power_provenance"],
        "cache_contract": cache_contract,
        "cache_authority": "McPAT 1.3 embedded CACTI-P",
        "embedded_cacti_p": embedded,
        "mcpat_provenance": provenance,
        "module_schema": {
            "schema_version": 1,
            "module_count": len(modules),
            "core_count": num_cores,
            "core_logic_granularity": mcpat.get("checks", {}).get(
                "core_logic_granularity"
            ),
            "requires_granular_cores": require_granular_cores,
            "cache_area_authority": "McPAT aggregate Area",
            "cache_shape_authority": "McPAT 1.3 embedded CACTI-P aspect ratio",
        },
        "mobility_contract": mobility_contract,
        "conservation": conservation,
        "area_provenance": {
            "core_logic_and_interconnect": "unmodified McPAT area",
            "l1i_l1d_l2": "McPAT aggregate area with embedded CACTI-P aspect ratio",
            "global_scaling": "none",
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
    return result


def build_model(r1_dir: Path, mcpat_json: Path, output: Path,
                require_communication_profile: bool = False,
                require_granular_cores: bool = True) -> dict:
    """Construct and publish one McPAT-authoritative physical module model."""
    result = construct_model(
        r1_dir, mcpat_json,
        require_communication_profile=require_communication_profile,
        require_granular_cores=require_granular_cores,
    )
    write_json(Path(output), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--mcpat-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-aggregate-cores", action="store_true",
                        help="legacy diagnostic only; formal models stay granular")
    args = parser.parse_args()
    result = build_model(
        args.r1_dir.resolve(), args.mcpat_json.resolve(),
        args.output.resolve(),
        require_granular_cores=not args.allow_aggregate_cores,
    )
    print(
        f"Module model: {len(result['modules'])} modules, "
        f"{result['totals']['area_mm2']:.3f} mm^2, "
        f"{result['totals']['total_power_w']:.3f} W"
    )


if __name__ == "__main__":
    main()
