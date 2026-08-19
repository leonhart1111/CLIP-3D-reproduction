#!/usr/bin/env python3
"""Aggregate a homogeneous lifting sweep and compute strict ranking diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from workflow.common import format_temperature_csv_row, read_json, write_json


def kendall_tau(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("ranking vectors have different lengths")
    concordant = discordant = 0
    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            product = (first[i] - first[j]) * (second[i] - second[j])
            concordant += product > 0
            discordant += product < 0
    pairs = len(first) * (len(first) - 1) / 2
    return (concordant - discordant) / pairs if pairs else 0.0


def rank_descending(values: list[float], index: int) -> int:
    return 1 + sum(value > values[index] for value in values)


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize(root: Path, csv_path: Path, json_path: Path,
              allow_proxy: bool = False, expected_points: int | None = None) -> dict:
    rows = [read_json(path) for path in sorted(root.rglob("pipeline_summary.json"))]
    if not rows:
        raise ValueError(f"no pipeline_summary.json below {root}")
    if expected_points is not None and len(rows) != expected_points:
        raise ValueError(f"expected {expected_points} points, found {len(rows)}")

    keys = [(row["workload"], row["l1d_size"], row["l2_size"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate workload/L1D/L2 points in sweep")
    methods = {row.get("layout_method", row.get("layout_mode")) for row in rows}
    cooling_values = {float(row.get("cooling", {}).get("r_convec_k_per_w", -1))
                      for row in rows}
    if len(methods) != 1 or len(cooling_values) != 1:
        raise ValueError(f"mixed sweep: methods={methods}, R_conv={cooling_values}")
    metric_names = {
        row.get("primary_performance_metric", "bips2") for row in rows
    }
    if len(metric_names) != 1 or not metric_names <= {"bips2", "work_units_per_ns"}:
        raise ValueError(f"mixed or unsupported performance metrics: {metric_names}")
    primary_metric = next(iter(metric_names))
    semantic = primary_metric == "work_units_per_ns"
    missing_r2 = ["/".join(key) for key, row in zip(keys, rows)
                  if (row.get("ipc2") is None or row.get("bips2") is None
                      or (semantic and (
                          row.get("work_units_per_cycle") is None
                          or row.get("work_units_per_ns") is None)))]
    if semantic and missing_r2:
        raise ValueError(
            f"{len(missing_r2)} semantic points lack fixed-work R2 throughput"
        )
    if missing_r2 and not allow_proxy:
        raise ValueError(
            f"{len(missing_r2)} points lack real R2 IPC/BIPS; use --allow-proxy only for diagnostics"
        )

    fields = (
        "workload", "l1d_size", "l2_size", "layout_method", "ipc1", "tmax_c",
        "sustainable_frequency_ghz", "bips1_thermal", "ipc2", "bips2",
        "primary_performance_metric", "work_units_per_cycle", "work_units_per_ns",
        "r2_critical_path_cycles", "total_pipeline_seconds",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_temperature_csv_row({
                field: row.get(field) for field in fields
            }))

    by_workload = defaultdict(list)
    for row in rows:
        by_workload[row["workload"]].append(row)
    diagnostics = {}
    for workload, points in sorted(by_workload.items()):
        ipc = [float(point["ipc1"]) for point in points]
        bips1 = [float(point["bips1_thermal"]) for point in points]
        primary_scores = [
            float(point["work_units_per_ns"])
            if semantic else float(
                point["bips2"] if point.get("bips2") is not None
                else point["bips1_thermal"]
            )
            for point in points
        ]
        best_ipc = max(range(len(points)), key=ipc.__getitem__)
        top = min(10, len(points))
        top_ipc = set(sorted(range(len(points)), key=ipc.__getitem__, reverse=True)[:top])
        top_primary = set(sorted(
            range(len(points)), key=primary_scores.__getitem__, reverse=True
        )[:top])
        ipc_best_rank = rank_descending(primary_scores, best_ipc)
        ipc_primary_tau = kendall_tau(ipc, primary_scores)
        bips1_primary_tau = kendall_tau(bips1, primary_scores)
        diagnostics[workload] = {
            "n": len(points),
            "primary_performance_metric": primary_metric,
            "throttled_fraction": sum(
                float(point["sustainable_frequency_ghz"]) < 2.0 for point in points
            ) / len(points),
            "minimum_frequency_ghz": min(
                float(point["sustainable_frequency_ghz"]) for point in points
            ),
            "ipc_best_to_primary_rank": ipc_best_rank,
            "ipc_best_to_bips_rank": ipc_best_rank,
            "top10_overlap": len(top_ipc & top_primary),
            "kendall_tau_ipc_primary": ipc_primary_tau,
            "kendall_tau_bips1_primary": bips1_primary_tau,
            "kendall_tau_ipc_bips2": ipc_primary_tau,
            "kendall_tau_bips1_bips2": bips1_primary_tau,
            "best_gap": max(primary_scores) / primary_scores[best_ipc],
            "uses_proxy_bips2": any(point.get("bips2") is None for point in points),
        }
    values = list(diagnostics.values())
    result = {
        "schema_version": 2, "point_count": len(rows),
        "expected_points": expected_points, "complete": expected_points in (None, len(rows)),
        "csv": str(csv_path.resolve()), "layout_method": next(iter(methods)),
        "primary_performance_metric": primary_metric,
        "r_convec_k_per_w": next(iter(cooling_values)),
        "r2_complete": not missing_r2, "missing_r2_count": len(missing_r2),
        "workloads": diagnostics,
        "aggregate": {
            "point_count": len(rows),
            "mean_throttled_fraction": sum(item["throttled_fraction"] for item in values) / len(values),
            "mean_ipc_best_to_primary_rank": sum(item["ipc_best_to_primary_rank"] for item in values) / len(values),
            "mean_ipc_best_to_bips_rank": sum(item["ipc_best_to_bips_rank"] for item in values) / len(values),
            "mean_top10_overlap": sum(item["top10_overlap"] for item in values) / len(values),
            "mean_kendall_tau_ipc_primary": sum(item["kendall_tau_ipc_primary"] for item in values) / len(values),
            "mean_kendall_tau_bips1_primary": sum(item["kendall_tau_bips1_primary"] for item in values) / len(values),
            "mean_kendall_tau_ipc_bips2": sum(item["kendall_tau_ipc_bips2"] for item in values) / len(values),
            "mean_kendall_tau_bips1_bips2": sum(item["kendall_tau_bips1_bips2"] for item in values) / len(values),
            "geometric_mean_best_gap": geometric_mean([item["best_gap"] for item in values]),
        },
        "score_definition": (
            "fixed semantic work throughput*f_sus" if semantic
            else "real BIPS2=IPC2*f_sus" if not missing_r2
            else "PROXY ONLY: missing R2 rows use IPC1*f_sus"
        ),
    }
    write_json(json_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-points", type=int)
    parser.add_argument("--allow-proxy", action="store_true")
    args = parser.parse_args()
    result = summarize(args.root.resolve(), args.csv.resolve(), args.output.resolve(),
                       args.allow_proxy, args.expected_points)
    print(f"summarized {result['point_count']} points; R2 complete={result['r2_complete']}")


if __name__ == "__main__":
    main()
