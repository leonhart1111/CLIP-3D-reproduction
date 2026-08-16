#!/usr/bin/env python3
"""Build equation (6)'s discrete cache/topology latency vector for gem5 R2."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from workflow.cache_contract import (
    mcpat_embedded_cache_record_identity,
    validate_cache_contract,
)
from workflow.common import parse_frequency_ghz, read_json, write_json
from workflow.floorplan.layout_metrics import (
    communication_weights_from_model,
    derive_layout_delays,
    select_rounded_wire_cycles,
)


_CACHE_AUTHORITY = "McPAT 1.3 embedded CACTI-P"
_MCPAT_PROVENANCE_AUTHORITY = "CLIP strict patched McPAT 1.3 runner"
_MCPAT_HASH_FIELDS = {
    "xml_sha256", "mapping_sha256", "output_sha256", "binary_sha256",
    "patch_sha256",
}


def access_cycles(access_time_s: float,
                  frequency_hz: float) -> tuple[float, int]:
    """Convert physical access seconds to gem5's integral cache latency."""
    access = float(access_time_s)
    frequency = float(frequency_hz)
    if not math.isfinite(access) or access <= 0:
        raise ValueError("McPAT CACTI-P access time must be finite and positive")
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("nominal cache frequency must be finite and positive")
    raw = access * frequency
    if not math.isfinite(raw):
        raise ValueError("McPAT CACTI-P raw access cycles must be finite")
    nearest_integer = round(raw)
    ceiling_input = (
        float(nearest_integer)
        if math.isclose(raw, nearest_integer, rel_tol=0.0, abs_tol=1e-12)
        else raw
    )
    return raw, max(1, math.ceil(ceiling_input))


def _validated_hashes(model: dict) -> dict[str, str]:
    provenance = model.get("mcpat_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
            "schema_version", "authority", "hashes"}:
        raise ValueError("module model lacks strict McPAT provenance")
    if provenance.get("schema_version") != 1:
        raise ValueError("McPAT provenance schema_version must be 1")
    if provenance.get("authority") != _MCPAT_PROVENANCE_AUTHORITY:
        raise ValueError("McPAT provenance authority is invalid")
    hashes = provenance.get("hashes")
    allowed = (_MCPAT_HASH_FIELDS, _MCPAT_HASH_FIELDS | {
        "build_provenance_sha256",
    })
    if not isinstance(hashes, dict) or set(hashes) not in allowed:
        raise ValueError("McPAT provenance hash schema is invalid")
    for field, value in hashes.items():
        try:
            valid = (
                isinstance(value, str) and len(value) == 64
                and int(value, 16) >= 0 and set(value) != {"0"}
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"McPAT provenance {field} must be a nonzero SHA-256 hash")
    return dict(hashes)


def _validated_native_records(model: dict) -> dict[str, list[dict]]:
    if model.get("schema_version") != 3:
        raise ValueError("R2 requires modules schema_version 3")
    metadata = model.get("architecture")
    if not isinstance(metadata, dict) or metadata.get("num_cores") != 4:
        raise ValueError("R2 requires an exact four-core module architecture")
    validate_cache_contract(model.get("cache_contract"), expected_core_count=4)
    if model.get("cache_authority") != _CACHE_AUTHORITY:
        raise ValueError("module cache authority is invalid")
    embedded = model.get("embedded_cacti_p")
    if not isinstance(embedded, dict) or set(embedded) != {
            "schema_version", "authority", "records"}:
        raise ValueError("module model lacks embedded CACTI-P schema")
    if embedded.get("schema_version") != 1:
        raise ValueError("embedded CACTI-P schema_version must be 1")
    if embedded.get("authority") != _CACHE_AUTHORITY:
        raise ValueError("embedded CACTI-P authority is invalid")
    records = embedded.get("records")
    if not isinstance(records, list):
        raise ValueError("embedded CACTI-P records must be a list")
    expected = {
        *(("l1i", core) for core in range(4)),
        *(("l1d", core) for core in range(4)),
        ("l2", None),
    }
    observed = []
    by_level = {"l1i": [], "l1d": [], "l2": []}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("embedded CACTI-P record must be an object")
        identity = (record.get("cache"), record.get("core"))
        observed.append(identity)
        rebuilt = mcpat_embedded_cache_record_identity(record)
        if record.get("record_id") != rebuilt:
            raise ValueError("embedded CACTI-P record_id does not match record")
        if identity[0] in by_level:
            by_level[identity[0]].append(record)
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("embedded CACTI-P record set is not the exact four-core set")
    for level in ("l1i", "l1d"):
        timing = {
            (float(record["access_time_s"]), float(record["cycle_time_s"]))
            for record in by_level[level]
        }
        if len(timing) != 1:
            raise ValueError(f"all four {level.upper()} records must agree on timing")
    return by_level


def build_vector(modules: Path, output: Path,
                 tsv_hops: int | None = None, wire_cycles: int | None = None,
                 layout_path: Path | None = None,
                 wire_rounding: str = "nearest", cycles_per_tsv: int = 2,
                 l1_pipeline_cycles: int = 1,
                 wire_aggregation: str = "mean") -> dict:
    model = read_json(modules)
    metadata = model["architecture"]
    records = _validated_native_records(model)
    hashes = _validated_hashes(model)
    frequency_hz = parse_frequency_ghz(metadata["cpu_clock"]) * 1e9
    converted = {}
    for level in ("l1i", "l1d", "l2"):
        access_time_s = float(records[level][0]["access_time_s"])
        raw, cycles = access_cycles(access_time_s, frequency_hz)
        converted[level] = {
            "access_time_s": access_time_s,
            "access_cycles_raw": raw,
            "rounding_policy": "ceil",
            "access_cycles": cycles,
            "record_ids": [record["record_id"] for record in records[level]],
        }
    cores = int(metadata["num_cores"])
    if wire_aggregation not in ("mean", "maximum", "traffic-weighted"):
        raise ValueError(
            "wire_aggregation must be 'mean', 'maximum', or 'traffic-weighted'"
        )
    if wire_aggregation == "traffic-weighted" and layout_path is None:
        raise ValueError(
            "traffic-weighted R2 requires a final layout to derive wire cycles"
        )
    communication_weights = communication_weights_from_model(
        model, required=wire_aggregation == "traffic-weighted"
    )
    layout_delays = None
    if layout_path is not None:
        layout_delays = derive_layout_delays(
            read_json(layout_path), frequency_hz / 1e9, wire_rounding,
            communication_weights,
        )
        if tsv_hops is None:
            tsv_hops = int(layout_delays["tsv_hops"])
        derived_wire_cycles = select_rounded_wire_cycles(
            layout_delays, wire_aggregation
        )
        if (wire_aggregation == "traffic-weighted" and wire_cycles is not None
                and int(wire_cycles) != derived_wire_cycles):
            raise ValueError(
                "traffic-weighted R2 cannot override the layout-derived wire cycles"
            )
        if wire_cycles is None:
            wire_cycles = derived_wire_cycles
    if tsv_hops is None:
        tsv_hops = 1
    if wire_cycles is None:
        wire_cycles = 0
    if tsv_hops < 0 or wire_cycles < 0:
        raise ValueError("TSV hops and wire cycles must be non-negative")
    arbitration = cores - 1
    if cycles_per_tsv < 0 or l1_pipeline_cycles < 1:
        raise ValueError("invalid TSV or L1 pipeline cycle parameter")
    tsv_cycles = cycles_per_tsv * tsv_hops
    l1_pipeline = l1_pipeline_cycles
    topology = arbitration + tsv_cycles + wire_cycles
    l1i_cycles = converted["l1i"]["access_cycles"]
    l1d_cycles = converted["l1d"]["access_cycles"]
    l2_cycles = converted["l2"]["access_cycles"]
    vector = {
        "schema_version": 1, "equation": 6,
        "components_cycles": {
            "l1i_mcpat_cacti_p": l1i_cycles,
            "l1d_mcpat_cacti_p": l1d_cycles,
            "l2_mcpat_cacti_p": l2_cycles,
            "l2_arbitration": arbitration,
            "tsv": tsv_cycles, "l1_pipeline": l1_pipeline,
            "layout_wire": wire_cycles,
        },
        "critical_l1d_to_l2_cycles": (
            l1d_cycles + l1_pipeline + l2_cycles + topology
        ),
        "gem5_overrides": {
            "l1i_tag_latency": l1i_cycles,
            "l1i_data_latency": l1i_cycles, "l1i_response_latency": l1_pipeline,
            "l1d_tag_latency": l1d_cycles,
            "l1d_data_latency": l1d_cycles, "l1d_response_latency": l1_pipeline,
            "l2_tag_latency": l2_cycles,
            "l2_data_latency": l2_cycles, "l2_response_latency": 1,
            "xbar_frontend_latency": 1, "xbar_forward_latency": max(topology, 1),
            "xbar_response_latency": 1, "xbar_snoop_response_latency": 1,
        },
        "gem5_args": [],
        "layout": str(layout_path.resolve()) if layout_path else None,
        "layout_delays": layout_delays,
        "wire_cycle_aggregation_for_r2": wire_aggregation,
        "mcpat_cacti_p_provenance": {
            "authority": _CACHE_AUTHORITY,
            "frequency_hz": frequency_hz,
            "mcpat_output_sha256": hashes["output_sha256"],
            "mcpat_binary_sha256": hashes["binary_sha256"],
            "records": converted,
        },
        "paper_parameters": [
            f"Ncores-1 arbitration = {arbitration}",
            f"{cycles_per_tsv} cycles/TSV x {tsv_hops}",
            f"L1 pipeline cycles = {l1_pipeline}",
        ],
        "reproduction_assumptions": [
            "McPAT embedded CACTI-P tag/data values are both assigned ceiling access cycles.",
            "Arbitration, TSV, and layout wire penalties are placed in xbar forward_latency only.",
            {
                "mean": (
                    "Mean Bakoglu-Meindl wire delay is used for the paper "
                    "equation-(15) mode."
                ),
                "maximum": (
                    "Maximum core-to-L2 wire delay is used as a conservative "
                    "shared-xbar timing bound."
                ),
                "traffic-weighted": (
                    "Demand-access-weighted core-to-L2 delay is represented by "
                    "one scalar shared-L2XBar latency; per-core latency is not modeled."
                ),
            }[wire_aggregation],
            "The selected wire delay is discretized using the recorded rounding policy.",
        ],
    }
    for key, value in vector["gem5_overrides"].items():
        vector["gem5_args"].extend(("--" + key.replace("_", "-"), str(value)))
    write_json(output, vector)
    return vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv-hops", type=int)
    parser.add_argument("--wire-cycles", type=int)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--wire-rounding", choices=("nearest", "ceil", "floor"), default="nearest")
    parser.add_argument("--cycles-per-tsv", type=int, default=2)
    parser.add_argument("--l1-pipeline-cycles", type=int, default=1)
    parser.add_argument(
        "--wire-aggregation", choices=("mean", "maximum", "traffic-weighted"),
        default="mean",
    )
    args = parser.parse_args()
    result = build_vector(args.modules, args.output, args.tsv_hops, args.wire_cycles,
                          args.layout.resolve() if args.layout else None,
                          args.wire_rounding, args.cycles_per_tsv,
                          args.l1_pipeline_cycles, args.wire_aggregation)
    print(f"R2 critical L1D-to-L2 path: {result['critical_l1d_to_l2_cycles']} cycles")


if __name__ == "__main__":
    main()
