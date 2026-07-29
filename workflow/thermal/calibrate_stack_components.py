#!/usr/bin/env python3
"""Fit separate effective silicon and TIM resistivities with HotSpot probes."""

from __future__ import annotations

import argparse
import copy
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import materialize
from workflow.thermal.run_hotspot import DEFAULT_HOTSPOT, run_hotspot


@dataclass(frozen=True)
class Anchor:
    label: str
    point_dir: Path
    target_tmax_c: float


def parse_anchor(text: str) -> Anchor:
    fields = text.split("=")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(
            "anchor must be LABEL=POINT_DIR=TARGET_TMAX_C"
        )
    label, raw_dir, raw_tmax = fields
    point_dir = Path(raw_dir).expanduser().resolve()
    required = ("pipeline_summary.json", "modules.json", "run_config.json",
                "hotspot/layout.json", "hotspot/thermal_result.json")
    missing = [name for name in required if not (point_dir / name).is_file()]
    if missing:
        raise argparse.ArgumentTypeError(f"anchor {point_dir} lacks {missing}")
    return Anchor(label, point_dir, float(raw_tmax))


def effective_resistivities(stack: dict) -> tuple[float, float]:
    common = float(stack.get("local_resistance_scale", 1.0))
    silicon = (
        float(stack["silicon_resistivity_mk_per_w"])
        * common * float(stack.get("silicon_resistance_scale", 1.0))
    )
    tim = (
        float(stack["tim_resistivity_mk_per_w"])
        * common * float(stack.get("tim_resistance_scale", 1.0))
    )
    return silicon, tim


def run_probe(anchor: Anchor, parameter: str, factor: float,
              output_dir: Path, hotspot: Path) -> dict:
    summary = read_json(anchor.point_dir / "pipeline_summary.json")
    config = read_json(anchor.point_dir / "run_config.json")["config"]
    physical = config["physical"]
    frequency = config["frequency"]
    original_stack = physical.get("thermal_stack", {})
    silicon, tim = effective_resistivities(original_stack)
    stack = copy.deepcopy(original_stack)
    stack["local_resistance_scale"] = 1.0
    stack["silicon_resistance_scale"] = 1.0
    stack["tim_resistance_scale"] = 1.0
    stack["silicon_resistivity_mk_per_w"] = silicon * (factor if parameter == "silicon" else 1.0)
    stack["tim_resistivity_mk_per_w"] = tim * (factor if parameter == "tim" else 1.0)
    case_dir = output_dir / anchor.label / f"{parameter}_x{factor:.6g}"
    if (case_dir / "thermal_result.json").is_file():
        thermal = read_json(case_dir / "thermal_result.json")
    else:
        materialize(
            anchor.point_dir / "modules.json", case_dir,
            int(physical["grid_size"]), float(physical["utilization"]),
            float(frequency["ambient_c"]), float(physical["r_convec_k_per_w"]),
            anchor.point_dir / "hotspot/layout.json", stack,
        )
        thermal = run_hotspot(case_dir, hotspot)
    return {
        "label": anchor.label,
        "parameter": parameter,
        "factor": factor,
        "base_tmax_c": float(summary["tmax_c"]),
        "probe_tmax_c": float(thermal["tmax_c"]),
        "target_tmax_c": anchor.target_tmax_c,
        "case_dir": str(case_dir.resolve()),
    }


def calibrate(anchors: list[Anchor], output_dir: Path, report_path: Path,
              probe_fraction: float = 0.10, workers: int = 2,
              hotspot: Path = DEFAULT_HOTSPOT) -> dict:
    import numpy as np

    if not anchors or not 0 < probe_fraction < 1:
        raise ValueError("anchors are required and probe_fraction must be in (0,1)")
    factor = 1.0 + probe_fraction
    jobs = [(anchor, parameter) for anchor in anchors
            for parameter in ("silicon", "tim")]
    probes = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_probe, anchor, parameter, factor,
                            output_dir, hotspot): (anchor, parameter)
            for anchor, parameter in jobs
        }
        for future in as_completed(futures):
            probes.append(future.result())
            print(f"HotSpot stack probes: {len(probes)}/{len(jobs)}", flush=True)
    probes.sort(key=lambda item: (item["label"], item["parameter"]))

    rows = []
    residuals = []
    anchor_records = []
    log_step = math.log(factor)
    for anchor in anchors:
        values = {item["parameter"]: item for item in probes
                  if item["label"] == anchor.label}
        base = values["silicon"]["base_tmax_c"]
        ds = (values["silicon"]["probe_tmax_c"] - base) / log_step
        dt = (values["tim"]["probe_tmax_c"] - base) / log_step
        rows.append([ds, dt])
        residuals.append(anchor.target_tmax_c - base)
        anchor_records.append({
            "label": anchor.label,
            "base_tmax_c": base,
            "target_tmax_c": anchor.target_tmax_c,
            "temperature_error_c": base - anchor.target_tmax_c,
            "dT_dlog_silicon": ds,
            "dT_dlog_tim": dt,
        })
    matrix = np.asarray(rows, dtype=float)
    target = np.asarray(residuals, dtype=float)
    delta, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)

    first_config = read_json(anchors[0].point_dir / "run_config.json")["config"]
    base_silicon, base_tim = effective_resistivities(
        first_config["physical"]["thermal_stack"]
    )
    fitted_silicon = base_silicon * math.exp(float(delta[0]))
    fitted_tim = base_tim * math.exp(float(delta[1]))
    predicted = matrix @ delta
    for record, correction in zip(anchor_records, predicted):
        record["linearized_predicted_tmax_c"] = record["base_tmax_c"] + float(correction)
        record["linearized_residual_c"] = (
            record["linearized_predicted_tmax_c"] - record["target_tmax_c"]
        )
    base_rmse = float(np.sqrt(np.mean(np.asarray([
        item["temperature_error_c"] for item in anchor_records
    ]) ** 2)))
    linearized_rmse = float(np.sqrt(np.mean(np.asarray([
        item["linearized_residual_c"] for item in anchor_records
    ]) ** 2)))
    accepted = bool(
        rank == 2
        and linearized_rmse < 1.0
        and max(abs(item["linearized_residual_c"]) for item in anchor_records) < 2.0
    )
    report = {
        "schema_version": 1,
        "method": "finite-difference HotSpot Jacobian in log resistivity",
        "probe_fraction": probe_fraction,
        "anchors": anchor_records,
        "probes": probes,
        "jacobian": matrix.tolist(),
        "jacobian_rank": int(rank),
        "jacobian_singular_values": [float(value) for value in singular],
        "base_effective_resistivity_mk_per_w": {
            "silicon": base_silicon, "tim": base_tim,
        },
        "log_parameter_update": {
            "silicon": float(delta[0]), "tim": float(delta[1]),
        },
        "fitted_effective_resistivity_mk_per_w": {
            "silicon": fitted_silicon, "tim": fitted_tim,
        },
        "suggested_thermal_stack": {
            **first_config["physical"]["thermal_stack"],
            "silicon_resistivity_mk_per_w": fitted_silicon,
            "tim_resistivity_mk_per_w": fitted_tim,
            "local_resistance_scale": 1.0,
            "silicon_resistance_scale": 1.0,
            "tim_resistance_scale": 1.0,
            "calibration": {
                "kind": "Table III two-component finite-difference fit",
                "report": str(report_path.resolve()),
                "requires_anchor_rerun": True,
            },
        },
        "recommendation": {
            "accepted": accepted,
            "base_rmse_c": base_rmse,
            "linearized_rmse_c": linearized_rmse,
            "acceptance": "rank 2, RMSE < 1 C, and maximum residual < 2 C",
            "action": (
                "validate the suggested stack on all anchors"
                if accepted else
                "reject the two-resistivity fit; repair module geometry/power mapping before further thermal tuning"
            ),
        },
        "limits": [
            "This is a local linearization and must be validated by rerunning every anchor.",
            "Only silicon/TIM effective resistivities are fitted; package geometry is fixed.",
            "The fit targets published temperatures, never layout improvement.",
        ],
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", action="append", type=parse_anchor, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--probe-fraction", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hotspot", type=Path, default=DEFAULT_HOTSPOT)
    args = parser.parse_args()
    report = calibrate(
        args.anchor, args.output_dir.resolve(), args.report.resolve(),
        args.probe_fraction, args.workers, args.hotspot.resolve(),
    )
    values = report["fitted_effective_resistivity_mk_per_w"]
    print(f"Fitted silicon={values['silicon']:.6g}, TIM={values['tim']:.6g} mK/W")


if __name__ == "__main__":
    main()
