#!/usr/bin/env python3
"""Build a HotSpot-calibrated linear thermal response operator.

The default ``module`` source mode needs one positive/negative HotSpot probe
per non-zero-power module, not one probe per 32x32 grid cell.  A module's
perturbation is distributed over the same area-overlap cells used by CLIP's
power rasterizer, preserving the spatial source contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import shutil
from pathlib import Path

import numpy as np

from workflow.common import read_json, sha256_file, write_json
from workflow.floorplan.generate_hotspot_inputs import overlap
from workflow.thermal.linear_response import (
    LinearThermalResponse,
    fit_central_difference,
    parse_ptrace,
)
from workflow.thermal.run_hotspot import (
    DEFAULT_HOTSPOT,
    grid_temperatures,
    run_hotspot,
)


def _temperature_vector(case_dir: Path) -> tuple[list[str], np.ndarray]:
    manifest = read_json(case_dir / "hotspot_manifest.json")
    grid_size = int(manifest["grid_size"])
    active_layers = [int(value) for value in manifest["active_power_layers"]]
    samples = dict(grid_temperatures(case_dir / "grid.steady.txt"))
    names, values = [], []
    for layer in active_layers:
        for index in range(grid_size * grid_size):
            name = f"layer_{layer}_g{index}"
            if name not in samples:
                raise ValueError(f"missing active HotSpot grid sample: {name}")
            names.append(name)
            values.append(float(samples[name]) - 273.15)
    return names, np.asarray(values, dtype=np.float64)


def _power_trace_values(names: list[str], values: np.ndarray) -> str:
    return "\t".join(names) + "\n" + "\t".join(
        f"{float(value):.17g}" for value in values
    ) + "\n"


def _required_case_files() -> tuple[str, ...]:
    return (
        "hotspot.config",
        "stack.lcf",
        "materials.txt",
        "bottom.flp",
        "top.flp",
        "hotspot_manifest.json",
    )


def _copy_case(baseline_case: Path, destination: Path,
               ptrace: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in _required_case_files():
        source = baseline_case / name
        if not source.is_file():
            raise FileNotFoundError(f"baseline case is missing {source}")
        shutil.copy2(source, destination / name)
    manifest = read_json(destination / "hotspot_manifest.json")
    manifest["files"] = {
        name: str((destination / name).resolve())
        for name in (*_required_case_files(), "power.ptrace")
    }
    manifest["power_trace"] = str((destination / "power.ptrace").resolve())
    write_json(destination / "hotspot_manifest.json", manifest)
    (destination / "power.ptrace").write_text(ptrace, encoding="utf-8")


def _grid_cells(case_dir: Path) -> tuple[list[str], dict[str, tuple[int, float]]]:
    names, values = parse_ptrace(case_dir / "power.ptrace")
    return names, {name: (index, float(values[index])) for index, name in enumerate(names)}


def _module_source_map(case_dir: Path) -> tuple[list[str], np.ndarray, list[dict[str, float]]]:
    layout = read_json(case_dir / "layout.json")
    grid = read_json(case_dir / "power_grid.json")
    names, indexed = _grid_cells(case_dir)
    source_modules = [
        module for module in layout["modules"]
        if float(module.get("total_power_w", 0.0)) > 0.0
    ]
    if not source_modules:
        raise ValueError("baseline layout contains no positive-power modules")
    manifest = read_json(case_dir / "hotspot_manifest.json")
    if manifest.get("input_granularity", "grid-cell") == "module":
        source_names, source_power, mappings = [], [], []
        for module in source_modules:
            name = str(module["name"])
            if name not in indexed:
                raise ValueError(
                    f"module-input power trace is missing module source {name}"
                )
            index, trace_power = indexed[name]
            module_power = float(module["total_power_w"])
            if not math.isclose(trace_power, module_power, rel_tol=1e-9,
                                abs_tol=1e-9):
                raise ValueError(
                    f"module power mismatch for {name}: layout={module_power} "
                    f"trace={trace_power}"
                )
            source_names.append(name)
            source_power.append(module_power)
            mappings.append({
                "module": name,
                "source_cells": [{"index": index, "weight": 1.0}],
                "source_power_w": module_power,
            })
        return source_names, np.asarray(source_power, dtype=np.float64), mappings
    cell_by_geometry = {}
    for tier in grid["tiers"]:
        for cell in tier["cells"]:
            if cell["name"] not in indexed:
                raise ValueError(f"power grid cell missing from ptrace: {cell['name']}")
            cell_by_geometry[(int(tier["tier"]), int(cell["row"]),
                             int(cell["column"]))] = cell
    grid_size = int(grid["grid_size"])
    side = float(layout["die_width_mm"])
    step = side / grid_size
    source_names, source_power, mappings = [], [], []
    for module in source_modules:
        source_names.append(str(module["name"]))
        source_power.append(float(module["total_power_w"]))
        module_area = float(module["width_mm"]) * float(module["height_mm"])
        if module_area <= 0.0:
            raise ValueError(f"module has non-positive area: {module['name']}")
        entries = []
        total_overlap = 0.0
        min_col = max(int(math.floor(float(module["x_mm"]) / step)), 0)
        max_col = min(int(math.ceil((float(module["x_mm"]) + float(module["width_mm"])) / step)), grid_size)
        min_row = max(int(math.floor(float(module["y_mm"]) / step)), 0)
        max_row = min(int(math.ceil((float(module["y_mm"]) + float(module["height_mm"])) / step)), grid_size)
        for row in range(min_row, max_row):
            for column in range(min_col, max_col):
                cell = cell_by_geometry[(int(module["tier"]), row, column)]
                shared = overlap(
                    module,
                    float(cell["x_mm"]), float(cell["y_mm"]),
                    float(cell["x_mm"]) + float(cell["width_mm"]),
                    float(cell["y_mm"]) + float(cell["height_mm"]),
                )
                if shared <= 0.0:
                    continue
                name = str(cell["name"])
                index, _ = indexed[name]
                entries.append({"index": index, "weight": shared / module_area})
                total_overlap += shared
        if not math.isclose(total_overlap, module_area, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"module-to-grid overlap is not conserved for {module['name']}: "
                f"{total_overlap} versus {module_area}"
            )
        mappings.append({
            "module": str(module["name"]),
            "source_cells": entries,
            "source_power_w": float(module["total_power_w"]),
        })
    return source_names, np.asarray(source_power, dtype=np.float64), mappings


def _perturbed_trace(baseline_values: np.ndarray, mapping: dict,
                     delta_w: float) -> np.ndarray:
    values = baseline_values.copy()
    for cell in mapping["source_cells"]:
        values[int(cell["index"])] += delta_w * float(cell["weight"])
    if np.any(values < -1.0e-12):
        raise ValueError("perturbed power trace contains negative power")
    return values


def _run_probe(baseline_case: Path, output_dir: Path, source_index: int,
               source_name: str, mapping: dict, baseline_values: np.ndarray,
               delta_w: float, hotspot: Path, force: bool) -> dict:
    probe_dir = output_dir / f"source_{source_index:04d}"
    positive_dir = probe_dir / "positive"
    negative_dir = probe_dir / "negative"
    metadata_path = probe_dir / "probe.json"
    signature = {
        "source_index": source_index,
        "source_name": source_name,
        "delta_w": delta_w,
        "baseline_power_sha256": sha256_file(baseline_case / "power.ptrace"),
    }
    if not force and metadata_path.is_file():
        existing = read_json(metadata_path)
        if existing.get("signature") == signature:
            return existing
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    plus_values = _perturbed_trace(baseline_values, mapping, delta_w)
    minus_values = _perturbed_trace(baseline_values, mapping, -delta_w)
    names, _ = parse_ptrace(baseline_case / "power.ptrace")
    _copy_case(baseline_case, positive_dir, _power_trace_values(names, plus_values))
    _copy_case(baseline_case, negative_dir, _power_trace_values(names, minus_values))
    run_hotspot(positive_dir, hotspot=hotspot)
    run_hotspot(negative_dir, hotspot=hotspot)
    _, positive_temperature = _temperature_vector(positive_dir)
    _, negative_temperature = _temperature_vector(negative_dir)
    result = {
        "schema_version": 1,
        "signature": signature,
        "source_index": source_index,
        "source_name": source_name,
        "delta_w": delta_w,
        "positive_case": str(positive_dir.resolve()),
        "negative_case": str(negative_dir.resolve()),
        "positive_temperature_c": positive_temperature.tolist(),
        "negative_temperature_c": negative_temperature.tolist(),
    }
    write_json(metadata_path, result)
    return result


def build_response(baseline_case: Path, output_dir: Path,
                   delta_w: float = 0.1, hotspot: Path = DEFAULT_HOTSPOT,
                   workers: int = 1, source_mode: str = "module",
                   force: bool = False, run_baseline: bool = False) -> LinearThermalResponse:
    baseline_case = baseline_case.resolve()
    output_dir = output_dir.resolve()
    if delta_w <= 0.0 or not math.isfinite(delta_w):
        raise ValueError("delta_w must be finite and positive")
    if source_mode != "module":
        raise ValueError(
            "phase-one linear response currently supports module sources only; "
            "grid-cell response needs one-sided probes for zero-power cells"
        )
    if not hotspot.is_file():
        raise FileNotFoundError(f"HotSpot executable does not exist: {hotspot}")
    if not (baseline_case / "grid.steady.txt").is_file():
        if not run_baseline:
            raise FileNotFoundError(
                "baseline grid.steady.txt is missing; run HotSpot first or pass "
                "--run-baseline"
            )
        run_hotspot(baseline_case, hotspot=hotspot)
    temperature_names, baseline_temperature = _temperature_vector(baseline_case)
    ptrace_names, baseline_grid_power = parse_ptrace(baseline_case / "power.ptrace")
    source_names, baseline_power, mappings = _module_source_map(baseline_case)
    too_small = [
        source_names[index] for index, value in enumerate(baseline_power)
        if value < delta_w - 1.0e-12
    ]
    if too_small:
        raise ValueError(
            "delta_w exceeds baseline power for module sources; reduce delta_w: "
            + ", ".join(too_small)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (index, name, mapping)
        for index, (name, mapping) in enumerate(zip(source_names, mappings))
    ]
    if workers < 1:
        raise ValueError("workers must be positive")
    results = []
    if workers == 1:
        for index, name, mapping in jobs:
            results.append(_run_probe(
                baseline_case, output_dir, index, name, mapping,
                baseline_grid_power, delta_w, hotspot, force,
            ))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(
                _run_probe, baseline_case, output_dir, index, name, mapping,
                baseline_grid_power, delta_w, hotspot, force,
            ) for index, name, mapping in jobs]
            for future in futures:
                results.append(future.result())
    results.sort(key=lambda item: int(item["source_index"]))
    response = fit_central_difference(
        source_names,
        temperature_names,
        baseline_power,
        baseline_temperature,
        results,
        metadata={
            "baseline_case": str(baseline_case),
            "baseline_case_contract_files": {
                name: sha256_file(baseline_case / name)
                for name in _required_case_files()
            },
            "source_mode": source_mode,
            "grid_size": int(read_json(baseline_case / "hotspot_manifest.json")["grid_size"]),
            "active_power_layers": read_json(baseline_case / "hotspot_manifest.json")["active_power_layers"],
            "hotspot_binary": str(hotspot.resolve()),
            "hotspot_binary_sha256": sha256_file(hotspot),
            "delta_w": delta_w,
            "source_mappings": mappings,
        },
    )
    response.save(output_dir / "linear_response")
    write_json(output_dir / "build_manifest.json", {
        "schema_version": 1,
        "source_mode": source_mode,
        "baseline_case": str(baseline_case),
        "source_names": source_names,
        "temperature_names": temperature_names,
        "probe_count": len(results),
        "response_manifest": str((output_dir / "linear_response.json").resolve()),
    })
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-mode", choices=("module",), default="module")
    parser.add_argument("--delta-w", type=float, default=0.1)
    parser.add_argument("--hotspot", type=Path, default=DEFAULT_HOTSPOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    response = build_response(
        args.baseline_case,
        args.output_dir,
        args.delta_w,
        args.hotspot,
        args.workers,
        args.source_mode,
        args.force,
        args.run_baseline,
    )
    print(
        f"linear response built: {response.source_count} sources, "
        f"{response.temperature_count} temperatures; "
        f"output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
