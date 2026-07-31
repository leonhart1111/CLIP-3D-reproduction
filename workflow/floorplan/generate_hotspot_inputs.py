#!/usr/bin/env python3
"""Place CLIP-3D modules, grid power by overlap, and write HotSpot 3-D inputs.

Equation (5) is implemented literally.  Every module rectangle is intersected
with every covered cell, and the generated manifest contains per-tier and
global conservation residuals for total, dynamic, and leakage power.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from workflow.common import read_json, write_json


DEFAULT_THERMAL_STACK = {
    "silicon_resistivity_mk_per_w": 0.01,
    "tim_resistivity_mk_per_w": 0.25,
    "interposer_thickness_m": 1.0e-4,
    "active_silicon_thickness_m": 5.0e-5,
    "tim_thickness_m": 2.0e-5,
    "local_resistance_scale": 1.0,
    "silicon_resistance_scale": 1.0,
    "tim_resistance_scale": 1.0,
}


def baseline_layout(model: dict, utilization: float = 0.70) -> dict:
    modules = [dict(module) for module in model["modules"]]
    for module in modules:
        module["tier"] = 1 if module["kind"] in ("l2", "interconnect") else 0
    tier_areas = [sum(m["area_mm2"] for m in modules if m["tier"] == z) for z in (0, 1)]
    die_side = math.sqrt(max(tier_areas) / utilization)
    quadrant = die_side / 2.0

    placed = []
    for core in range(4):
        group = [m for m in modules if m.get("core") == core]
        area = sum(m["area_mm2"] for m in group)
        width = math.sqrt(area)
        height = area / width
        qx, qy = core % 2, core // 2
        left = qx * quadrant + (quadrant - width) / 2.0
        bottom = qy * quadrant + (quadrant - height) / 2.0
        y = bottom
        for module in sorted(group, key=lambda item: item["kind"]):
            module["x_mm"] = left
            module["y_mm"] = y
            module["width_mm"] = width
            module["height_mm"] = module["area_mm2"] / width
            y += module["height_mm"]
            placed.append(module)

    top = sorted((m for m in modules if m["tier"] == 1),
                 key=lambda module: (module["kind"] != "l2", module["name"]))
    dimensions = []
    for module in top:
        width = float(module.get("preferred_width_mm", math.sqrt(module["area_mm2"])))
        if width <= 0:
            raise ValueError(f"module has non-positive preferred width: {module['name']}")
        # Derive height from the authoritative area so small CACTI text-rounding
        # errors cannot break power-density or overlap conservation.
        dimensions.append((width, float(module["area_mm2"]) / width))
    gap = die_side * 0.02 if len(top) > 1 else 0.0
    row_width = sum(width for width, _ in dimensions) + gap * max(len(top) - 1, 0)
    if row_width > die_side + 1e-9:
        raise ValueError("top-tier shelf does not fit inside the die")
    if dimensions and max(height for _, height in dimensions) > die_side + 1e-9:
        raise ValueError("top-tier module height does not fit inside the die")
    # The paper's layout-study baseline anchors L2 at the lower-left corner.
    # Keep all memory-side modules in one deterministic shelf from that point.
    x = 0.0
    for module, (width, height) in zip(top, dimensions):
        module["x_mm"] = x
        module["y_mm"] = 0.0
        module["width_mm"] = width
        module["height_mm"] = height
        x += width + gap
        placed.append(module)

    unplaced = {m["name"] for m in modules} - {m["name"] for m in placed}
    if unplaced:
        raise ValueError(f"no placement rule for modules: {sorted(unplaced)}")
    check_geometry(placed, die_side)
    return {
        "schema_version": 1, "policy": "paper fixed-bin P1 baseline",
        "die_width_mm": die_side, "die_height_mm": die_side,
        "utilization_target": utilization, "tier_module_area_mm2": tier_areas,
        "modules": placed,
        "paper_parameters": ["four quadrant core placement", "P1: cores bottom, L2 top",
                             "lower-left memory-side shelf", "70% utilization"],
    }


def overlap(a: dict, x0: float, y0: float, x1: float, y1: float) -> float:
    return max(0.0, min(a["x_mm"] + a["width_mm"], x1) - max(a["x_mm"], x0)) * max(
        0.0, min(a["y_mm"] + a["height_mm"], y1) - max(a["y_mm"], y0)
    )


def check_geometry(modules: list[dict], die_side: float, tolerance: float = 1e-9) -> None:
    for module in modules:
        if min(module["x_mm"], module["y_mm"]) < -tolerance:
            raise ValueError(f"module outside die: {module['name']}")
        if module["x_mm"] + module["width_mm"] > die_side + tolerance:
            raise ValueError(f"module outside die: {module['name']}")
        if module["y_mm"] + module["height_mm"] > die_side + tolerance:
            raise ValueError(f"module outside die: {module['name']}")
    for i, first in enumerate(modules):
        for second in modules[i + 1:]:
            if first["tier"] != second["tier"]:
                continue
            if overlap(first, second["x_mm"], second["y_mm"],
                       second["x_mm"] + second["width_mm"],
                       second["y_mm"] + second["height_mm"]) > tolerance:
                raise ValueError(f"overlap: {first['name']} and {second['name']}")


def grid_power(layout: dict, grid_size: int) -> dict:
    side = layout["die_width_mm"]
    step = side / grid_size
    fields = ("dynamic_power_w", "leakage_power_w", "total_power_w")
    tiers = []
    checks = []
    for tier in (0, 1):
        cells = []
        for row in range(grid_size):
            for column in range(grid_size):
                cell = {"name": f"t{tier}_r{row:02d}_c{column:02d}",
                        "row": row, "column": column,
                        "x_mm": column * step, "y_mm": row * step,
                        "width_mm": step, "height_mm": step}
                for field in fields:
                    cell[field] = 0.0
                cells.append(cell)
        for module in (m for m in layout["modules"] if m["tier"] == tier):
            area = module["width_mm"] * module["height_mm"]
            if area <= 0:
                raise ValueError(f"module has non-positive area: {module['name']}")
            min_col = max(int(math.floor(module["x_mm"] / step)), 0)
            max_col = min(int(math.ceil((module["x_mm"] + module["width_mm"]) / step)), grid_size)
            min_row = max(int(math.floor(module["y_mm"] / step)), 0)
            max_row = min(int(math.ceil((module["y_mm"] + module["height_mm"]) / step)), grid_size)
            for row in range(min_row, max_row):
                for column in range(min_col, max_col):
                    shared = overlap(module, column * step, row * step,
                                     (column + 1) * step, (row + 1) * step)
                    cell = cells[row * grid_size + column]
                    for field in fields:
                        cell[field] += module[field] * shared / area
        tier_checks = {"tier": tier}
        for field in fields:
            source = sum(m[field] for m in layout["modules"] if m["tier"] == tier)
            gridded = sum(cell[field] for cell in cells)
            residual = gridded - source
            tier_checks[field] = {"source": source, "grid": gridded, "residual": residual}
            if not math.isclose(source, gridded, rel_tol=1e-10, abs_tol=1e-10):
                raise ArithmeticError(f"tier {tier} {field} is not conserved: {residual}")
        checks.append(tier_checks)
        tiers.append({"tier": tier, "cells": cells})
    return {"grid_size": grid_size, "cell_side_mm": step, "tiers": tiers,
            "power_conservation": checks}


def write_floorplan(path: Path, cells: list[dict]) -> None:
    lines = ["# name\twidth(m)\theight(m)\tleft-x(m)\tbottom-y(m)"]
    for cell in cells:
        lines.append(
            f"{cell['name']}\t{cell['width_mm'] / 1000:.12g}\t"
            f"{cell['height_mm'] / 1000:.12g}\t{cell['x_mm'] / 1000:.12g}\t"
            f"{cell['y_mm'] / 1000:.12g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ptrace(path: Path, tiers: list[dict], field: str) -> None:
    cells = [cell for tier in tiers for cell in tier["cells"]]
    path.write_text(
        "\t".join(cell["name"] for cell in cells) + "\n" +
        "\t".join(f"{cell[field]:.17g}" for cell in cells) + "\n",
        encoding="utf-8",
    )


def materialize(model_path: Path, output_dir: Path, grid_size: int = 32,
                utilization: float = 0.70, ambient_c: float = 25.0,
                r_convec: float = 3.5, layout_path: Path | None = None,
                stack_config: dict | None = None) -> dict:
    model = read_json(model_path)
    stack = {**DEFAULT_THERMAL_STACK, **(stack_config or {})}
    local_scale = float(stack["local_resistance_scale"])
    silicon_scale = float(stack["silicon_resistance_scale"])
    tim_scale = float(stack["tim_resistance_scale"])
    if min(local_scale, silicon_scale, tim_scale) <= 0:
        raise ValueError("thermal stack resistance scales must be positive")
    silicon_resistivity = (
        float(stack["silicon_resistivity_mk_per_w"]) * local_scale * silicon_scale
    )
    tim_resistivity = (
        float(stack["tim_resistivity_mk_per_w"]) * local_scale * tim_scale
    )
    layout = read_json(layout_path) if layout_path else baseline_layout(model, utilization)
    check_geometry(layout["modules"], layout["die_width_mm"])
    grids = grid_power(layout, grid_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "layout.json", layout)
    write_json(output_dir / "power_grid.json", grids)
    bottom = grids["tiers"][0]["cells"]
    top = grids["tiers"][1]["cells"]
    write_floorplan(output_dir / "bottom.flp", bottom)
    write_floorplan(output_dir / "top.flp", top)
    write_ptrace(output_dir / "power.ptrace", grids["tiers"], "total_power_w")
    write_ptrace(output_dir / "power_dynamic.ptrace", grids["tiers"], "dynamic_power_w")
    write_ptrace(output_dir / "power_leakage.ptrace", grids["tiers"], "leakage_power_w")

    lcf = f"""# layer, lateral, power, heat capacity, resistivity, thickness, floorplan
0
Y
N
1.75e6
{silicon_resistivity:.12g}
{float(stack['interposer_thickness_m']):.12g}
bottom.flp
1
Y
Y
1.75e6
{silicon_resistivity:.12g}
{float(stack['active_silicon_thickness_m']):.12g}
bottom.flp
2
Y
N
4.0e6
{tim_resistivity:.12g}
{float(stack['tim_thickness_m']):.12g}
bottom.flp
3
Y
Y
1.75e6
{silicon_resistivity:.12g}
{float(stack['active_silicon_thickness_m']):.12g}
top.flp
4
Y
N
4.0e6
{tim_resistivity:.12g}
{float(stack['tim_thickness_m']):.12g}
top.flp
"""
    (output_dir / "stack.lcf").write_text(lcf, encoding="utf-8")
    materials = """silicon
solid
130.0
1630300

copper
solid
400.0
3.55e6
"""
    (output_dir / "materials.txt").write_text(materials, encoding="utf-8")
    ambient_k = ambient_c + 273.15
    config = f"""# CLIP-3D generated HotSpot configuration
-material_chip silicon
-c_convec 140.4
-r_convec {r_convec}
-s_sink 0.06
-t_sink 0.0069
-material_sink copper
-s_spreader 0.03
-t_spreader 0.001
-material_spreader copper
-t_interface 2.0e-05
-k_interface 4.0
-p_interface 4.0e6
-ambient {ambient_k}
-init_temp {ambient_k}
-sampling_intvl 0.01
-base_proc_freq 2.0e9
-dtm_used 0
-model_type grid
-grid_rows {grid_size}
-grid_cols {grid_size}
-grid_map_mode avg
-model_secondary 0
-package_model_used 0
-leakage_used 0
"""
    (output_dir / "hotspot.config").write_text(config, encoding="utf-8")
    manifest = {
        "schema_version": 1, "model": str(model_path.resolve()),
        "layout": str((output_dir / "layout.json").resolve()),
        "grid_size": grid_size, "ambient_c": ambient_c, "r_convec_k_per_w": r_convec,
        "thermal_stack": {
            **stack,
            "effective_silicon_resistivity_mk_per_w": silicon_resistivity,
            "effective_tim_resistivity_mk_per_w": tim_resistivity,
        },
        "files": {name: str((output_dir / name).resolve()) for name in
                  ("bottom.flp", "top.flp", "stack.lcf", "materials.txt",
                   "hotspot.config", "power.ptrace", "power_dynamic.ptrace",
                   "power_leakage.ptrace", "power_grid.json")},
        "power_conservation": grids["power_conservation"],
        "paper_parameters": ["32x32 per tier", "five layers", "50um active silicon"],
        "reproduction_assumptions": [
            "Experiment T_amb=25 C is used; methodology prose separately mentions a 45 C natural-convection setup.",
            "The passive interposer thickness is 100 um because the translated paper does not publish it.",
            "local_resistance_scale is explicit because the paper names Cool-3D defaults but does not publish all layer material values.",
        ],
    }
    write_json(output_dir / "hotspot_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--utilization", type=float, default=0.70)
    parser.add_argument("--ambient-c", type=float, default=25.0)
    parser.add_argument("--r-convec", type=float, default=3.5)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--local-resistance-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if not 0 < args.utilization <= 1:
        parser.error("--utilization must be in (0, 1]")
    result = materialize(args.modules.resolve(), args.output_dir.resolve(),
                         args.grid_size, args.utilization, args.ambient_c,
                         args.r_convec, args.layout.resolve() if args.layout else None,
                         {"local_resistance_scale": args.local_resistance_scale})
    residual = max(abs(item[field]["residual"])
                   for item in result["power_conservation"]
                   for field in ("dynamic_power_w", "leakage_power_w", "total_power_w"))
    print(f"HotSpot inputs written; maximum power residual = {residual:.3e} W")


if __name__ == "__main__":
    main()
