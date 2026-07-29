#!/usr/bin/env python3
"""Attach a separately completed gem5 R2 result to an existing lifted point."""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.thermal.sustainable_frequency import evaluate


def attach(point_dir: Path) -> dict:
    summary_path = point_dir / "pipeline_summary.json"
    summary = read_json(summary_path)
    result_path = point_dir / "gem5_r2/r2_result.json"
    status_path = point_dir / "gem5_r2/status.json"
    if not result_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("gem5_r2 result/status is missing")
    if read_json(status_path).get("state") != "success":
        raise ValueError("gem5 R2 status is not success")
    result = read_json(result_path)
    vector = read_json(point_dir / "r2_latency.json")
    if Path(result["latency_vector"]).resolve() != (point_dir / "r2_latency.json").resolve():
        raise ValueError("R2 result refers to a different latency vector")
    run_config = read_json(point_dir / "run_config.json")["config"]
    frequency = run_config["frequency"]
    performance = evaluate(
        point_dir / "modules.json", point_dir / "hotspot/thermal_result.json",
        point_dir / "performance.json", frequency["f0_ghz"],
        frequency["fmin_ghz"], frequency["tsafe_c"], frequency["ambient_c"],
        result["ipc2"],
    )
    summary["ipc2"] = result["ipc2"]
    summary["bips2"] = performance["bips2"]
    summary["r2_source"] = str(result_path.resolve())
    summary.setdefault("artifacts", {})["r2_result"] = str(result_path.resolve())
    summary.setdefault("stage_seconds", {})["gem5_r2"] = result.get("elapsed_seconds")
    summary["total_pipeline_seconds"] = sum(
        float(value) for value in summary["stage_seconds"].values() if value is not None
    )
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = attach(args.point_dir.resolve())
    print(f"attached IPC2={summary['ipc2']:.6f}, BIPS2={summary['bips2']:.6f}")


if __name__ == "__main__":
    main()
