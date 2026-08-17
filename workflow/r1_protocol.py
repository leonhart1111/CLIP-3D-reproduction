"""Canonical gem5 R1 protocols and stable protocol identities.

A protocol identity separates legacy 100M-warmup/500M-measurement runs from
short all-core convergence candidates.  It binds the instruction window, the
workload command, and the exact gem5 configuration/binary, so a successful
output directory is only reusable when every identity component matches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


PROTOCOL_FAMILY = "clip3d-r1"
PROTOCOL_SCHEMA_VERSION = 1
SUPPORTED_SCOPES = frozenset({"cpu0", "all-cores"})
LEGACY_PROFILE_NAMES = frozenset({"paper", "paper_all_cores", "smoke"})


@dataclass(frozen=True)
class R1Protocol:
    family: str
    profile: str
    warmup_insts: int
    measure_insts: int
    instruction_window_scope: str


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def canonical_protocol(value: Mapping[str, Any]) -> dict:
    """Validate and normalize one R1 protocol declaration."""
    if not isinstance(value, Mapping):
        raise ValueError("R1 protocol must be an object")
    family = value.get("family", PROTOCOL_FAMILY)
    if not isinstance(family, str) or not family:
        raise ValueError("R1 protocol family must be a non-empty string")
    profile = value.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("R1 protocol profile must be a non-empty string")
    scope = value.get("instruction_window_scope")
    if scope not in SUPPORTED_SCOPES:
        raise ValueError(
            "R1 protocol instruction_window_scope must be cpu0 or all-cores"
        )
    canonical = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "family": family,
        "profile": profile,
        "warmup_insts": _positive_int(
            value.get("warmup_insts"), "warmup_insts"
        ),
        "measure_insts": _positive_int(
            value.get("measure_insts"), "measure_insts"
        ),
        "instruction_window_scope": scope,
    }
    return canonical


def protocol_id(value: Mapping[str, Any],
                workload_options: Mapping[str, Any] | None = None,
                gem5_config_sha256: str | None = None,
                gem5_binary_sha256: str | None = None) -> str:
    """Return a stable SHA-256 protocol identity for one R1 job."""
    canonical = canonical_protocol(value)
    payload = {
        **canonical,
        "workload_options": dict(workload_options or {}),
        "gem5_config_sha256": gem5_config_sha256,
        "gem5_binary_sha256": gem5_binary_sha256,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def require_protocol(metadata: Mapping[str, Any],
                     *, family: str | None = None) -> dict:
    """Require an explicit canonical protocol in corrected R1 metadata."""
    recorded = metadata.get("r1_protocol")
    canonical = canonical_protocol(recorded) if isinstance(recorded, dict) else None
    if canonical is None:
        raise ValueError("R1 metadata lacks a canonical r1_protocol")
    if family is not None and canonical["family"] != family:
        raise ValueError(
            f"R1 protocol family mismatch: expected {family}, "
            f"observed {canonical['family']}"
        )
    return canonical


def classify_legacy_protocol(metadata: Mapping[str, Any]) -> str:
    """Classify read-only historical metadata without writing any identity."""
    if isinstance(metadata.get("r1_protocol"), dict):
        try:
            canonical = canonical_protocol(metadata["r1_protocol"])
            return (
                "legacy-paper-500m"
                if canonical["profile"] in LEGACY_PROFILE_NAMES
                and canonical["measure_insts"] >= 500_000_000
                else "corrected"
            )
        except ValueError:
            return "malformed"
    warmup = metadata.get("warmup_insts")
    measure = metadata.get("measure_insts")
    if warmup == 100_000_000 and measure == 500_000_000:
        return "legacy-paper-500m"
    return "legacy-unidentified"


def protocol_from_metadata(metadata: Mapping[str, Any]) -> dict:
    """Derive the canonical protocol object recorded by clip_r1.py."""
    if isinstance(metadata.get("r1_protocol"), dict):
        return canonical_protocol(metadata["r1_protocol"])
    return canonical_protocol({
        "profile": "legacy",
        "warmup_insts": metadata.get("warmup_insts"),
        "measure_insts": metadata.get("measure_insts"),
        "instruction_window_scope": metadata.get(
            "instruction_window_scope", "cpu0"
        ),
    })


def protocol_fields(protocol: Mapping[str, Any]) -> dict:
    """Return the flattened dataclass view used by the sweep runner."""
    canonical = canonical_protocol(protocol)
    return {
        "family": canonical["family"],
        "profile": canonical["profile"],
        "warmup_insts": canonical["warmup_insts"],
        "measure_insts": canonical["measure_insts"],
        "instruction_window_scope": canonical["instruction_window_scope"],
    }
