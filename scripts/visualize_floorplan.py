#!/usr/bin/env python3
"""Visualize a placed CLIP-3D floorplan with module power as color.

The input is either a layout JSON (for example ``optimized_layout.json``,
``hotspot/layout.json``, or ``baseline_layout.json``) or a pipeline output
directory containing one of those files.  Each module rectangle is drawn at
its physical (x, y, width, height) position in its tier; the fill color
encodes the selected power metric on a log10 scale so that low-power blocks
remain distinguishable from multi-watt cores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__import__("tempfile").gettempdir()) / "matplotlib-clip")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import ticker
from matplotlib.patches import Rectangle
import numpy as np


METRICS = ("total_power_w", "dynamic_power_w", "leakage_power_w")
LAYOUT_CANDIDATES = (
    "optimized_layout.json", "hotspot/layout.json", "layout.json",
    "baseline_layout.json",
)


def resolve_layout(source: Path) -> dict:
    """Return a placed layout dict from a JSON file or pipeline directory."""
    source = Path(source)
    if source.is_dir():
        for name in LAYOUT_CANDIDATES:
            candidate = source / name
            if candidate.is_file():
                source = candidate
                break
        else:
            raise FileNotFoundError(
                f"no layout file in {source}; looked for {LAYOUT_CANDIDATES}"
            )
    if not source.is_file():
        raise FileNotFoundError(source)
    layout = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(layout, dict) or not isinstance(layout.get("modules"), list):
        raise ValueError(f"{source} is not a placed layout JSON")
    return layout


def validate_modules(layout: dict) -> list[dict]:
    modules = layout["modules"]
    for module in modules:
        if not isinstance(module, dict):
            raise ValueError("layout modules must be objects")
        for field in ("name", "x_mm", "y_mm", "width_mm", "height_mm", "tier"):
            if field not in module:
                raise ValueError(
                    f"module {module.get('name')!r} lacks placed field {field!r}"
                )
        if not all(
            isinstance(module[field], (int, float))
            and math.isfinite(float(module[field]))
            for field in ("x_mm", "y_mm", "width_mm", "height_mm")
        ):
            raise ValueError(
                f"module {module.get('name')!r} has non-finite geometry"
            )
    return modules


def power_values(modules: list[dict], metric: str) -> list[float]:
    values = []
    for module in modules:
        value = module.get(metric)
        if value is None:
            raise ValueError(
                f"module {module.get('name')!r} lacks {metric!r}; "
                "pass --metric total|dynamic|leakage"
            )
        values.append(float(value))
    return values


def draw_tier(axis, modules: list[dict], values: list[float], metric: str,
              tier: int, die_width: float, die_height: float,
              norm, label_power: bool, font_size: float) -> None:
    axis.set_xlim(0, die_width)
    axis.set_ylim(0, die_height)
    axis.set_aspect("equal")
    axis.set_title(f"tier {tier} — power: {metric}", fontsize=font_size + 1)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.grid(True, linewidth=0.3, alpha=0.4)

    for module, value in zip(modules, values):
        x, y = float(module["x_mm"]), float(module["y_mm"])
        width, height = float(module["width_mm"]), float(module["height_mm"])
        color = matplotlib.colormaps["viridis"](norm(value))
        axis.add_patch(Rectangle(
            (x, y), width, height, facecolor=color, edgecolor="black",
            linewidth=0.6,
        ))
        area_fraction = width * height / (die_width * die_height)
        if label_power and area_fraction >= 0.01:
            axis.text(
                x + width / 2, y + height / 2,
                f"{module['name']}\n{value:.3g} W",
                ha="center", va="center", fontsize=font_size - 2,
                color="white" if norm(value) < 0.55 else "black",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="layout JSON or a pipeline output directory containing one",
    )
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument(
        "--metric", choices=METRICS, default="total_power_w",
        help="power field used for the fill color",
    )
    parser.add_argument(
        "--tier", type=int, choices=(0, 1),
        help="draw only one tier; default draws both",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="skip module name/power labels",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    layout = resolve_layout(args.source)
    modules = validate_modules(layout)
    values = power_values(modules, args.metric)
    die_width = float(layout.get("die_width_mm", max(
        float(m["x_mm"]) + float(m["width_mm"]) for m in modules
    )))
    die_height = float(layout.get("die_height_mm", die_width))

    tiers = [args.tier] if args.tier is not None else sorted(
        {int(m["tier"]) for m in modules}
    )
    positive = [v for v in values if v > 0]
    if not positive:
        raise ValueError("no positive power values; cannot build a log color scale")
    vmin = math.log10(min(positive))
    vmax = math.log10(max(positive))
    if vmax - vmin < 0.5:
        vmin = math.floor(vmin)
        vmax = math.ceil(vmax)
    norm = mcolors.LogNorm(vmin=10 ** vmin, vmax=10 ** vmax)

    figure, axes = plt.subplots(
        1, len(tiers), figsize=(6.4 * len(tiers), 6.0),
        squeeze=False,
    )
    for column, tier in enumerate(tiers):
        selected = [
            (module, value) for module, value in zip(modules, values)
            if int(module["tier"]) == tier
        ]
        if not selected:
            raise ValueError(f"tier {tier} has no modules")
        draw_tier(
            axes[0][column],
            [module for module, _ in selected],
            [value for _, value in selected],
            args.metric, tier, die_width, die_height, norm,
            not args.no_labels, 9,
        )

    if args.title:
        figure.suptitle(args.title, fontsize=12)
    colorbar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axes[0][-1], fraction=0.046, pad=0.04,
    )
    colorbar.set_label(args.metric.replace("_", " ") + " (W, log scale)")
    colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda value, _position: f"{10 ** value:.3g}"
    ))

    output = args.output or args.source.with_suffix(
        f".floorplan_{args.metric.replace('_power_w', '')}.png"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
