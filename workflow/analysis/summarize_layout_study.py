#!/usr/bin/env python3
"""Build paper Tables V--VII diagnostics from four complete layout sweeps."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from workflow.common import read_json, write_json


METHODS = ("fixed-bin", "cool3d-standard", "sa-lambda", "clip3d")
LAYOUT_WORKLOADS = {"fft", "matmul", "stencil", "stream"}


def load_points(root: Path, expected_method: str) -> dict[tuple[str, str, str], dict]:
    points = {}
    for path in sorted(root.rglob("pipeline_summary.json")):
        row = read_json(path)
        if row.get("workload") not in LAYOUT_WORKLOADS:
            continue
        method = row.get("layout_method", row.get("layout_mode"))
        if method != expected_method:
            raise ValueError(f"{path} has method {method}, expected {expected_method}")
        if row.get("ipc2") is None or row.get("bips2") is None:
            raise ValueError(f"layout study requires real R2: {path}")
        key = (row["workload"], row["l1d_size"], row["l2_size"])
        if key in points:
            raise ValueError(f"duplicate point {key} below {root}")
        points[key] = row
    if not points:
        raise ValueError(f"no points below {root}")
    return points


def statistics(values: list[float], wins: list[bool]) -> dict:
    return {
        "mean_percent": sum(values) / len(values),
        "max_percent": max(values),
        "wins": sum(wins), "n": len(values),
    }


def summarize(roots: dict[str, Path], output: Path, csv_path: Path,
              expected_points: int = 80) -> dict:
    data = {method: load_points(roots[method], method) for method in METHODS}
    key_sets = {method: set(points) for method, points in data.items()}
    if any(keys != key_sets["fixed-bin"] for keys in key_sets.values()):
        raise ValueError("layout methods do not contain identical architecture points")
    keys = sorted(key_sets["fixed-bin"])
    if len(keys) != expected_points:
        raise ValueError(f"expected {expected_points} common points, found {len(keys)}")

    rows = []
    buckets = defaultdict(lambda: defaultdict(lambda: {"actual": [], "paper": [], "wins": []}))
    baseline_bips = defaultdict(list)
    for key in keys:
        baseline = data["fixed-bin"][key]
        state = "throttled" if baseline["sustainable_frequency_ghz"] < 2.0 else "headroom"
        row = {"workload": key[0], "l1d_size": key[1], "l2_size": key[2],
               "state": state, "fixed_bips2": baseline["bips2"],
               "fixed_bips_proxy": baseline["bips1_thermal"]}
        for bucket in ((key[0], state), ("ALL", state), ("ALL", "ALL")):
            baseline_bips[bucket].append(float(baseline["bips2"]))
        for method in METHODS[1:]:
            point = data[method][key]
            actual = 100.0 * (point["bips2"] / baseline["bips2"] - 1.0)
            paper = 100.0 * (
                point["bips1_thermal"] / baseline["bips1_thermal"] - 1.0
            )
            row[f"{method}_actual_percent"] = actual
            row[f"{method}_paper_percent"] = paper
            for bucket in ((key[0], state), ("ALL", state), ("ALL", "ALL")):
                entry = buckets[bucket][method]
                entry["actual"].append(actual)
                entry["paper"].append(paper)
                entry["wins"].append(actual > 0.0)
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table_v = {}
    for bucket, methods in sorted(buckets.items()):
        if bucket == ("ALL", "ALL"):
            continue
        table_v[f"{bucket[0]}/{bucket[1]}"] = {
            "fixed_bin_baseline": {
                "mean_bips": sum(baseline_bips[bucket]) / len(baseline_bips[bucket]),
                "max_bips": max(baseline_bips[bucket]),
                "n": len(baseline_bips[bucket]),
            },
            "methods": {
                method: statistics(values["actual"], values["wins"])
                for method, values in methods.items()
            },
        }

    table_vi = {}
    overall = buckets[("ALL", "ALL")]
    for method in METHODS[1:]:
        layout_seconds = sum(
            float(data[method][key].get("stage_seconds", {}).get("layout_and_hotspot", 0.0))
            for key in keys
        )
        values = overall[method]
        table_vi[method] = {
            "actual": statistics(values["actual"], values["wins"]),
            "paper_proxy": statistics(values["paper"],
                                      [value > 0.0 for value in values["paper"]]),
            "layout_and_hotspot_seconds": layout_seconds,
        }

    clip = overall["clip3d"]
    table_vii = {
        "paper_proxy": statistics(clip["paper"], [value > 0.0 for value in clip["paper"]]),
        "actual_r2": statistics(clip["actual"], clip["wins"]),
        "throttled_actual_wins": table_v["ALL/throttled"]["methods"]["clip3d"]["wins"],
        "headroom_actual_wins": table_v["ALL/headroom"]["methods"]["clip3d"]["wins"],
    }
    result = {
        "schema_version": 1, "point_count": len(keys),
        "roots": {method: str(path.resolve()) for method, path in roots.items()},
        "csv": str(csv_path.resolve()), "table_v": table_v,
        "table_vi": table_vi, "table_vii": table_vii,
        "reproduction_boundary": (
            "Cool3D-standard and SA+lambda algorithms are explicit local reproductions "
            "because the paper does not publish their implementation parameters."
        ),
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--cool3d-root", type=Path, required=True)
    parser.add_argument("--sa-root", type=Path, required=True)
    parser.add_argument("--clip3d-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, default=80)
    args = parser.parse_args()
    roots = {
        "fixed-bin": args.fixed_root.resolve(),
        "cool3d-standard": args.cool3d_root.resolve(),
        "sa-lambda": args.sa_root.resolve(),
        "clip3d": args.clip3d_root.resolve(),
    }
    result = summarize(roots, args.output.resolve(), args.csv.resolve(), args.expected_points)
    print(f"summarized {result['point_count']} four-method layout points")


if __name__ == "__main__":
    main()
