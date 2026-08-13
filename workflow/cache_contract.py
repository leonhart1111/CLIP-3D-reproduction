#!/usr/bin/env python3
"""One auditable cache-organization contract shared by McPAT and CACTI."""

from __future__ import annotations

import hashlib
import json
import math

from workflow.common import parse_size_bytes


DEVICE_TYPES = {0: "itrs-hp", 1: "itrs-lstp", 2: "itrs-lop"}
INTERCONNECT_PROJECTIONS = {0: "aggressive", 1: "conservative"}


def stable_identity(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_access_cycles(access_time_ns: float, frequency_ghz: float) -> int:
    """Convert a physical access time to an integral synchronous latency."""
    access = float(access_time_ns)
    frequency = float(frequency_ghz)
    if not math.isfinite(access) or access <= 0:
        raise ValueError("CACTI access time must be a finite positive value")
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("cache frequency must be a finite positive value")
    cycles = access * frequency
    nearest_integer = round(cycles)
    if math.isclose(cycles, nearest_integer, rel_tol=0.0, abs_tol=1e-12):
        cycles = float(nearest_integer)
    return max(1, math.ceil(cycles))


def build_cache_contract(
        metadata: dict, *, technology_nm: int, temperature_k: int,
        device_type: int, interconnect_projection_type: int) -> dict:
    """Derive the physical cache organization represented by one gem5 R1."""
    technology = int(technology_nm)
    temperature = int(temperature_k)
    if technology <= 0:
        raise ValueError("cache technology_nm must be positive")
    if temperature % 10 or not 300 <= temperature <= 400:
        raise ValueError("cache temperature_k must be 300..400 K in 10 K steps")
    try:
        cell_type = DEVICE_TYPES[int(device_type)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("cache device_type must be 0, 1, or 2") from error
    try:
        projection = INTERCONNECT_PROJECTIONS[int(interconnect_projection_type)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "cache interconnect_projection_type must be 0 or 1"
        ) from error

    defaults = {
        "l1_cache_banks": 1,
        "l2_cache_banks": 1,
        "l1_cache_output_width_bits": 512,
        "l2_cache_output_width_bits": 512,
    }
    applied = {key: value for key, value in defaults.items() if key not in metadata}
    resolved = {key: int(metadata.get(key, value)) for key, value in defaults.items()}
    for key, value in resolved.items():
        if value <= 0:
            raise ValueError(f"{key} must be positive")

    common = {
        "technology_nm": technology,
        "line_size_bytes": int(metadata.get("cache_line_bytes", 64)),
        "temperature_k": temperature,
        "device_type": cell_type,
        "interconnect_projection": projection,
        "access_mode": "normal",
        "ecc": True,
        "core_count": int(metadata.get("num_cores", 4)),
        "cacti_model": "UCA",
        # CACTI's input grammar only distinguishes L2/L3.  This field does
        # not affect UCA organization, so L2 is used for all on-chip caches.
        "cacti_cache_level": "L2",
    }
    if common["line_size_bytes"] <= 0 or common["core_count"] <= 0:
        raise ValueError("cache line size and core count must be positive")

    records = []
    for level, size_key, assoc_key, bank_key, width_key in (
        ("l1i", "l1i_size", "l1_associativity", "l1_cache_banks",
         "l1_cache_output_width_bits"),
        ("l1d", "l1d_size", "l1_associativity", "l1_cache_banks",
         "l1_cache_output_width_bits"),
        ("l2", "l2_size", "l2_associativity", "l2_cache_banks",
         "l2_cache_output_width_bits"),
    ):
        record = {
            "level": level,
            "size": metadata[size_key],
            "size_bytes": parse_size_bytes(metadata[size_key]),
            "associativity": int(metadata[assoc_key]),
            "bank_count": resolved[bank_key],
            "output_width_bits": resolved[width_key],
            **common,
        }
        for field in ("size_bytes", "associativity"):
            if record[field] <= 0:
                raise ValueError(f"{level}.{field} must be positive")
        record["contract_record_id"] = stable_identity(record)
        records.append(record)

    contract = {
        "schema_version": 1,
        "source": "gem5 R1 metadata plus explicit physical-model settings",
        "legacy_defaults_applied": applied,
        "records": records,
    }
    contract["contract_id"] = stable_identity(contract)
    return contract
