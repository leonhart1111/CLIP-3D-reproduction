#!/usr/bin/env python3
"""Draw HotSpot's grid temperature maps with a yellow-to-red gradient.

The input is a pipeline ``hotspot`` directory (or a ``grid.steady.txt`` file).
HotSpot's detailed-3D output contains one ``Layer N:`` section per physical
stack layer; by default only the active power layers from
``hotspot_manifest.json`` are drawn (for example layers 1 and 3 for a two-tier
stack).  Temperatures are rendered with the ``autumn`` colormap (yellow = cool,
red = hot) on the actual 32x32 (or configured) grid, with axes in millimetres.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__import__("tempfile").gettempdir()) / "matplotlib-clip")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from workflow.thermal.run_hotspot import parse_grid_temperatures


def resolve_grid_source(source: Path) -> tuple[Path, Path | None]:
    """Return (grid.steady.txt, hotspot_manifest.json) from a dir or file."""
    source = Path(source)
    if source.is_dir():
        grid = source / "grid.steady.txt"
        manifest = source / "hotspot_manifest.json"
        if not grid.is_file():
            raise FileNotFoundError(f"no grid.steady.txt in {source}")
        return grid, manifest if manifest.is_file() else None
    if not source.is_file():
        raise FileNotFoundError(source)
    return source, None


def grid_geometry(manifest: dict | None) -> tuple[int, float, float]:
    """Return (grid_size, die_width_mm, die_height_mm)."""
    if manifest is not None:
        grid_size = int(manifest.get("grid_size", 32))
        cell_side = float(manifest.get("cell_side_mm", 0.0))
        if cell_side > 0:
            return grid_size, grid_size * cell_side, grid_size * cell_side
        layout_ref = manifest.get("layout")
        if isinstance(layout_ref, dict):
            layout = layout_ref
        elif isinstance(layout_ref, str) and Path(layout_ref).is_file():
            layout = json.loads(Path(layout_ref).read_text(encoding="utf-8"))
        else:
            layout = None
        if isinstance(layout, dict):
            return (
                grid_size,
                float(layout.get("die_width_mm", grid_size)),
                float(layout.get("die_height_mm", grid_size)),
            )
    return 32, 32.0, 32.0


def active_layers(manifest: dict | None, requested: list[int] | None) -> list[int]:
    if requested:
        return requested
    if manifest is not None:
        recorded = manifest.get("active_power_layers")
        if isinstance(recorded, list) and recorded:
            return [int(value) for value in recorded]
    return [0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument(
        "--layer", type=int, action="append", dest="layers",
        help="draw only this physical layer (repeatable); default: active layers",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--tmin-c", type=float, default=None)
    parser.add_argument("--tmax-c", type=float, default=None)
    args = parser.parse_args()

    grid_path, manifest_path = resolve_grid_source(args.source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path is not None else None
    grid_size, die_width, die_height = grid_geometry(manifest)
    layers = active_layers(manifest, args.layers)

    values = parse_grid_temperatures(grid_path.read_text(encoding="utf-8"))
    by_layer: dict[int, np.ndarray] = {}
    for layer in layers:
        cells = np.full(grid_size * grid_size, np.nan)
        for name, value in values:
            prefix = f"layer_{layer}_g"
            if name.startswith(prefix):
                cells[int(name[len(prefix):])] = float(value)
        if not np.isfinite(cells).any():
            raise ValueError(
                f"grid.steady.txt has no cells for layer {layer}; "
                f"available layers: {sorted({int(n.split('_')[1]) for n, _ in values})}"
            )
        by_layer[layer] = cells.reshape(grid_size, grid_size)

    all_temps = np.concatenate([by_layer[layer].ravel() for layer in layers])
    vmin = args.tmin_c if args.tmin_c is not None else float(np.nanmin(all_temps))
    vmax = args.tmax_c if args.tmax_c is not None else float(np.nanmax(all_temps))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin >= vmax:
        raise ValueError("invalid temperature range for the color scale")

    figure, axes = plt.subplots(
        1, len(layers), figsize=(5.4 * len(layers), 5.0), squeeze=False,
    )
    image = None
    for column, layer in enumerate(layers):
        axis = axes[0][column]
        image = axis.imshow(
            by_layer[layer], cmap="autumn", vmin=vmin, vmax=vmax,
            origin="lower", aspect="equal",
            extent=(0, die_width, 0, die_height),
        )
        axis.set_title(f"Layer {layer}", fontsize=11)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.grid(False)
    if args.title:
        figure.suptitle(args.title, fontsize=12)
    colorbar = figure.colorbar(image, ax=axes[0][-1], fraction=0.046, pad=0.04)
    colorbar.set_label("temperature (°C)")

    output = args.output or (
        grid_path.with_suffix("").with_name(
            f"temperature_grid_{grid_size}x{grid_size}.png"
        )
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"wrote {output} (grid {grid_size}x{grid_size}, layers {layers})")


if __name__ == "__main__":
    main()
