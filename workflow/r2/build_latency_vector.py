#!/usr/bin/env python3
"""Build equation (6)'s discrete cache/topology latency vector for gem5 R2."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow.common import parse_size_bytes, read_json, write_json
from workflow.floorplan.layout_metrics import (
    communication_weights_from_model,
    derive_layout_delays,
)


def lookup(cacti: dict, level: str, size: str) -> dict:
    wanted = parse_size_bytes(size)
    matches = [record for record in cacti["records"]
               if record["level"] == level and record["size_bytes"] == wanted]
    if len(matches) != 1:
        raise KeyError(f"CACTI table has {len(matches)} matches for {level} {size}")
    return matches[0]


def build_vector(modules: Path, cacti_path: Path, output: Path,
                 tsv_hops: int | None = None, wire_cycles: int | None = None,
                 layout_path: Path | None = None,
                 wire_rounding: str = "nearest", cycles_per_tsv: int = 2,
                 l1_pipeline_cycles: int = 1,
                 wire_aggregation: str = "mean") -> dict:
    model = read_json(modules)
    metadata = model["architecture"]
    cacti = read_json(cacti_path)
    l1i = lookup(cacti, "l1d", metadata["l1i_size"])
    l1d = lookup(cacti, "l1d", metadata["l1d_size"])
    l2 = lookup(cacti, "l2", metadata["l2_size"])
    cores = int(metadata["num_cores"])
    if wire_aggregation not in ("mean", "maximum", "traffic-weighted"):
        raise ValueError(
            "wire_aggregation must be 'mean', 'maximum', or 'traffic-weighted'"
        )
    communication_weights = communication_weights_from_model(
        model, required=wire_aggregation == "traffic-weighted"
    )
    layout_delays = None
    if layout_path is not None:
        layout_delays = derive_layout_delays(
            read_json(layout_path), float(cacti["frequency_ghz"]), wire_rounding,
            communication_weights,
        )
        if tsv_hops is None:
            tsv_hops = int(layout_delays["tsv_hops"])
        if wire_cycles is None:
            selected_fields = {
                "mean": "wire_cycles",
                "maximum": "maximum_wire_cycles",
                "traffic-weighted": "traffic_weighted_wire_cycles",
            }
            wire_cycles = int(layout_delays[selected_fields[wire_aggregation]])
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
    vector = {
        "schema_version": 1, "equation": 6,
        "components_cycles": {
            "l1i_cacti": l1i["access_cycles"], "l1d_cacti": l1d["access_cycles"],
            "l2_cacti": l2["access_cycles"], "l2_arbitration": arbitration,
            "tsv": tsv_cycles, "l1_pipeline": l1_pipeline,
            "layout_wire": wire_cycles,
        },
        "critical_l1d_to_l2_cycles": (
            l1d["access_cycles"] + l1_pipeline + l2["access_cycles"] + topology
        ),
        "gem5_overrides": {
            "l1i_tag_latency": l1i["access_cycles"],
            "l1i_data_latency": l1i["access_cycles"], "l1i_response_latency": l1_pipeline,
            "l1d_tag_latency": l1d["access_cycles"],
            "l1d_data_latency": l1d["access_cycles"], "l1d_response_latency": l1_pipeline,
            "l2_tag_latency": l2["access_cycles"],
            "l2_data_latency": l2["access_cycles"], "l2_response_latency": 1,
            "xbar_frontend_latency": 1, "xbar_forward_latency": max(topology, 1),
            "xbar_response_latency": 1, "xbar_snoop_response_latency": 1,
        },
        "gem5_args": [],
        "layout": str(layout_path.resolve()) if layout_path else None,
        "layout_delays": layout_delays,
        "wire_cycle_aggregation_for_r2": wire_aggregation,
        "paper_parameters": [
            f"Ncores-1 arbitration = {arbitration}",
            f"{cycles_per_tsv} cycles/TSV x {tsv_hops}",
            f"L1 pipeline cycles = {l1_pipeline}",
        ],
        "reproduction_assumptions": [
            "CACTI tag/data values are both assigned the rounded array access cycles.",
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
    parser.add_argument("--cacti", type=Path, required=True)
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
    result = build_vector(args.modules, args.cacti, args.output,
                          args.tsv_hops, args.wire_cycles,
                          args.layout.resolve() if args.layout else None,
                          args.wire_rounding, args.cycles_per_tsv,
                          args.l1_pipeline_cycles, args.wire_aggregation)
    print(f"R2 critical L1D-to-L2 path: {result['critical_l1d_to_l2_cycles']} cycles")


if __name__ == "__main__":
    main()
