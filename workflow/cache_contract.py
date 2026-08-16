#!/usr/bin/env python3
"""One auditable cache-organization contract shared by McPAT and CACTI."""

from __future__ import annotations

import hashlib
import json
import math

from workflow.common import parse_size_bytes


DEVICE_TYPES = {0: "itrs-hp", 1: "itrs-lstp", 2: "itrs-lop"}
INTERCONNECT_PROJECTIONS = {0: "aggressive", 1: "conservative"}
CACHE_CONTRACT_SCHEMA_VERSION = 2
CACHE_CONTRACT_AUTHORITY = "gem5 R1 metadata for McPAT cache organization"
CACHE_CONTRACT_SOURCE = "gem5 R1 metadata plus explicit physical-model settings"
MCPAT_EMBEDDED_CACTI_IDENTITY_FIELDS = (
    "access_time_s", "cache", "core", "cycle_time_s", "height_mm",
    "mcpat_version", "model", "width_mm",
)


def stable_identity(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mcpat_embedded_cache_record_identity(record: dict) -> str:
    """Rebuild the content identity emitted for one embedded CACTI-P record."""
    try:
        numeric = {
            key: float(record[key]) for key in (
                "access_time_s", "cycle_time_s", "height_mm", "width_mm",
            )
        }
        cache = record["cache"]
        core = record["core"]
        mcpat_version = record["mcpat_version"]
        model = record["model"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("incomplete McPAT embedded CACTI-P record") from error
    if not all(math.isfinite(value) and value > 0 for value in numeric.values()):
        raise ValueError("nonphysical McPAT embedded CACTI-P record")
    if cache not in {"l1i", "l1d", "l2"}:
        raise ValueError("invalid McPAT embedded CACTI-P cache identity")
    if cache == "l2":
        if core is not None:
            raise ValueError("McPAT embedded CACTI-P L2 must be shared")
    elif isinstance(core, bool) or not isinstance(core, int) or core < 0:
        raise ValueError("McPAT embedded CACTI-P L1 requires a core index")
    if mcpat_version != "1.3" or model != "embedded-cacti-p":
        raise ValueError("invalid McPAT embedded CACTI-P model identity")
    canonical = {
        key: (
            format(numeric[key], ".17e")
            if key in numeric else record[key]
        )
        for key in MCPAT_EMBEDDED_CACTI_IDENTITY_FIELDS
    }
    return stable_identity(canonical)


def characterization_identity(characterization: dict) -> str:
    """Return a content identity while excluding machine-local evidence paths."""
    normalized = {
        key: value for key, value in characterization.items()
        if key != "characterization_id"
    }
    normalized["records"] = [
        {
            key: value for key, value in record.items()
            if key not in ("config", "raw_output")
        }
        for record in characterization.get("records", [])
    ]
    provenance = dict(characterization.get("provenance", {}))
    provenance.pop("cacti_executable", None)
    provenance.pop("base_config", None)
    normalized["provenance"] = provenance
    return stable_identity(normalized)


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
        "schema_version": CACHE_CONTRACT_SCHEMA_VERSION,
        "authority": CACHE_CONTRACT_AUTHORITY,
        "source": CACHE_CONTRACT_SOURCE,
        "legacy_defaults_applied": applied,
        "records": records,
    }
    contract["contract_id"] = stable_identity(contract)
    return contract


CONTRACT_FIELDS = (
    "level", "size_bytes", "associativity", "bank_count",
    "output_width_bits", "technology_nm", "line_size_bytes",
    "temperature_k", "device_type", "interconnect_projection",
    "access_mode", "ecc", "core_count", "cacti_model",
    "cacti_cache_level",
)
_CACHE_COMMON_FIELDS = (
    "technology_nm", "line_size_bytes", "temperature_k", "device_type",
    "interconnect_projection", "access_mode", "ecc", "core_count",
    "cacti_model", "cacti_cache_level",
)


def validate_cache_contract(contract: object,
                            expected_core_count: int = 4) -> dict:
    """Validate and return the corrected gem5-to-McPAT cache contract."""
    if not isinstance(contract, dict) or set(contract) != {
        "schema_version", "authority", "source", "legacy_defaults_applied",
        "records", "contract_id",
    }:
        raise ValueError("McPAT cache contract has an invalid object schema")
    if contract.get("schema_version") != CACHE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("McPAT cache contract schema_version must be 2")
    if contract.get("authority") != CACHE_CONTRACT_AUTHORITY:
        raise ValueError("McPAT cache contract has an invalid authority")
    if contract.get("source") != CACHE_CONTRACT_SOURCE:
        raise ValueError("McPAT cache contract has an invalid source")
    if isinstance(expected_core_count, bool) or not isinstance(
            expected_core_count, int) or expected_core_count <= 0:
        raise ValueError("expected cache-contract core count must be positive")
    defaults = contract.get("legacy_defaults_applied")
    if not isinstance(defaults, dict) or any(
        key not in {
            "l1_cache_banks", "l2_cache_banks",
            "l1_cache_output_width_bits", "l2_cache_output_width_bits",
        } or isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for key, value in defaults.items()
    ):
        raise ValueError("McPAT cache contract legacy defaults are invalid")
    records = contract.get("records")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("McPAT cache contract must contain three records")
    expected_record_fields = set(CONTRACT_FIELDS) | {
        "size", "contract_record_id",
    }
    levels = []
    common_settings = []
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_record_fields:
            raise ValueError("McPAT cache contract record schema is invalid")
        level = record.get("level")
        levels.append(level)
        if record.get("core_count") != expected_core_count:
            raise ValueError(f"McPAT cache contract {level}.core_count mismatch")
        for field in (
            "size_bytes", "associativity", "bank_count", "output_width_bits",
            "technology_nm", "line_size_bytes", "temperature_k", "core_count",
        ):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"McPAT cache contract {level}.{field} is invalid")
        size = record.get("size")
        try:
            valid_size = (
                isinstance(size, str)
                and parse_size_bytes(size) == record["size_bytes"]
            )
        except (TypeError, ValueError):
            valid_size = False
        if not valid_size:
            raise ValueError(f"McPAT cache contract {level}.size is invalid")
        expected_values = {
            "device_type": tuple(DEVICE_TYPES.values()),
            "interconnect_projection": tuple(INTERCONNECT_PROJECTIONS.values()),
            "access_mode": ("normal",),
            "cacti_model": ("UCA",),
            "cacti_cache_level": ("L2",),
        }
        for field, allowed in expected_values.items():
            if record.get(field) not in allowed:
                raise ValueError(
                    f"McPAT cache contract {level}.{field} is invalid"
                )
        if record.get("ecc") is not True:
            raise ValueError(f"McPAT cache contract {level}.ecc is invalid")
        temperature = record["temperature_k"]
        if temperature % 10 or not 300 <= temperature <= 400:
            raise ValueError(
                f"McPAT cache contract {level}.temperature_k is invalid"
            )
        expected_record_id = stable_identity({
            key: value for key, value in record.items()
            if key != "contract_record_id"
        })
        if record.get("contract_record_id") != expected_record_id:
            raise ValueError(
                f"McPAT cache contract {level}.contract_record_id mismatch"
            )
        common_settings.append(tuple(
            record[field] for field in _CACHE_COMMON_FIELDS
        ))
    if set(levels) != {"l1i", "l1d", "l2"} or len(levels) != len(set(levels)):
        raise ValueError("McPAT cache contract cache levels are invalid")
    if len(set(common_settings)) != 1:
        raise ValueError("McPAT cache contract common settings disagree across levels")
    expected_contract_id = stable_identity({
        key: value for key, value in contract.items() if key != "contract_id"
    })
    if contract.get("contract_id") != expected_contract_id:
        raise ValueError("McPAT cache contract contract_id mismatch")
    return dict(contract)


def validate_characterization(cacti: dict, expected_contract: dict) -> dict[str, dict]:
    """Validate a local CACTI artifact and return one record per cache level."""
    if int(cacti.get("schema_version", 0)) != 2:
        raise ValueError("CACTI characterization schema_version must be 2")
    if not str(cacti.get("rounding", "")).startswith("ceiling"):
        raise ValueError("CACTI characterization must use ceiling latency rounding")
    frequency = float(cacti.get("frequency_ghz", 0.0))
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("CACTI characterization frequency_ghz must be positive")
    records = cacti.get("records")
    if not isinstance(records, list):
        raise ValueError("CACTI characterization records must be a list")

    actual_by_level: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("CACTI characterization record must be an object")
        level = str(record.get("level", ""))
        if level in actual_by_level:
            raise ValueError(f"duplicate CACTI characterization record for {level}")
        actual_by_level[level] = record

    expected_records = expected_contract.get("records")
    if not isinstance(expected_records, list):
        raise ValueError("expected cache contract records must be a list")
    expected_by_level = {str(record["level"]): record for record in expected_records}
    if set(actual_by_level) != set(expected_by_level):
        missing = sorted(set(expected_by_level) - set(actual_by_level))
        extra = sorted(set(actual_by_level) - set(expected_by_level))
        raise ValueError(
            f"CACTI characterization level mismatch: missing={missing}, extra={extra}"
        )

    for level, expected in expected_by_level.items():
        actual = actual_by_level[level]
        for field in CONTRACT_FIELDS:
            if actual.get(field) != expected.get(field):
                raise ValueError(
                    f"CACTI {level}.{field} mismatch: "
                    f"expected {expected.get(field)!r}, observed {actual.get(field)!r}"
                )
        for field in (
            "access_time_ns", "cycle_time_ns", "area_mm2", "width_mm",
            "height_mm",
        ):
            try:
                value = float(actual[field])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"CACTI {level}.{field} must be positive") from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"CACTI {level}.{field} must be positive")
        expected_cycles = cache_access_cycles(actual["access_time_ns"], frequency)
        if actual.get("access_cycles") != expected_cycles:
            raise ValueError(
                f"CACTI {level}.access_cycles mismatch: expected "
                f"{expected_cycles}, observed {actual.get('access_cycles')!r}"
            )
        expected_cycle_cycles = cache_access_cycles(
            actual["cycle_time_ns"], frequency
        )
        if actual.get("cycle_cycles") != expected_cycle_cycles:
            raise ValueError(
                f"CACTI {level}.cycle_cycles mismatch: expected "
                f"{expected_cycle_cycles}, observed {actual.get('cycle_cycles')!r}"
            )
        record_identity = stable_identity({
            key: value for key, value in actual.items()
            if key not in ("config", "raw_output", "cacti_record_id")
        })
        if actual.get("cacti_record_id") != record_identity:
            raise ValueError(f"CACTI {level}.cacti_record_id does not match record")

    artifact_identity = characterization_identity(cacti)
    if cacti.get("characterization_id") != artifact_identity:
        raise ValueError("CACTI characterization_id does not match artifact")
    return actual_by_level
