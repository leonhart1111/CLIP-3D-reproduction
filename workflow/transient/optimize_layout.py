#!/usr/bin/env python3
"""Backend-selectable transient L2 layout optimization."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

from workflow.common import read_json, write_json
from workflow.floorplan.discrete_partition import place_l2, search_discrete_partitions
from workflow.floorplan.generate_hotspot_inputs import baseline_layout
from workflow.floorplan.layout_metrics import (
    aggregate_wire_cycles,
    communication_weights_from_model,
    derive_layout_delays,
    mean_wire_cycles,
    round_wire_cycles,
)
from workflow.transient.five_state import (
    find_five_state_sustainable_frequency,
    parse_five_state_settings,
)


def _finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or nonnegative and result < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return result


def _frequency_grid(config: dict) -> list[float]:
    from workflow.transient.rom.optimize_layout import canonical_frequency_grid

    return canonical_frequency_grid(config)


def _peak_evidence(search: dict, sustainable: float) -> dict:
    matches = [
        value for value in search.get("evaluations", [])
        if isinstance(value, dict) and value.get("frequency_ghz") == sustainable
    ]
    if len(matches) != 1:
        raise ValueError("five-state search lacks unique sustainable evidence")
    return matches[0]


def _optimize_five_state(modules_path: Path, output_dir: Path,
                         config_path: Path, power_windows_path: Path) -> dict:
    for path in (modules_path, config_path, power_windows_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"transient optimizer output directory is not empty: {output_dir}"
        )
    model = read_json(modules_path)
    config = read_json(config_path)
    power_windows = read_json(power_windows_path)
    if not isinstance(model, dict) or not isinstance(model.get("modules"), list):
        raise ValueError("modules must contain a module list")
    if not isinstance(config, dict) or not isinstance(power_windows, dict):
        raise ValueError("config and power windows must be dictionaries")
    settings = parse_five_state_settings(config)
    if settings.backend != "five-state":
        raise ValueError("five-state optimizer requires backend five-state")
    ipc1 = _finite(model.get("ipc1"), "modules.ipc1", nonnegative=True)
    if ipc1 <= 0.0:
        raise ValueError("modules.ipc1 must be positive")
    optimizer = config.get("layout_optimizer")
    delay = config.get("delay")
    physical = config.get("physical", {})
    if not isinstance(optimizer, dict) or not isinstance(delay, dict):
        raise ValueError("config requires layout_optimizer and delay dictionaries")
    if not isinstance(physical, dict):
        raise ValueError("config physical must be a dictionary")
    utilization = _finite(
        physical.get("utilization", 0.7), "physical.utilization"
    )
    if not 0.0 < utilization <= 1.0:
        raise ValueError("physical.utilization must lie in (0, 1]")
    base = baseline_layout(model, utilization)
    l2s = [module for module in base["modules"] if module.get("kind") == "l2"]
    if len(l2s) != 1:
        raise ValueError("five-state search requires exactly one L2")
    l2_name = l2s[0]["name"]
    allowed_tiers = optimizer.get("allowed_l2_tiers", [0, 1])
    if not isinstance(allowed_tiers, list) or not allowed_tiers:
        raise ValueError("layout_optimizer.allowed_l2_tiers must be a list")
    wire_objective = optimizer.get("wire_objective", "continuous")
    if wire_objective not in ("continuous", "r2-quantized", "discrete-partition"):
        raise ValueError("unsupported transient wire objective")
    grid_steps = optimizer.get("partition_grid_steps", 41)
    include_fixed = optimizer.get("include_fixed_baseline", True)
    lambda_wire = _finite(
        optimizer.get("lambda_wire"), "layout_optimizer.lambda_wire",
        nonnegative=True,
    )
    wire_rounding = delay.get("wire_rounding", "nearest")
    wire_aggregation = delay.get("wire_aggregation", "mean")
    weights = communication_weights_from_model(
        model, required=wire_aggregation == "traffic-weighted"
    )
    frequencies = _frequency_grid(config)

    def evaluate(layout: dict, origin: str, start: str) -> dict | None:
        l2 = next(module for module in layout["modules"] if module["kind"] == "l2")
        search = find_five_state_sustainable_frequency(
            layout, power_windows, frequencies, config
        )
        sustainable = search.get("sustainable_frequency_ghz")
        _, per_core = mean_wire_cycles(
            layout["modules"], float(config["frequency"]["f0_ghz"])
        )
        continuous = aggregate_wire_cycles(per_core, wire_aggregation, weights)
        rounded = round_wire_cycles(continuous, wire_rounding)
        common = {
            "origin": origin,
            "start": start,
            "stage": origin,
            "tier": int(l2["tier"]),
            "x_mm": float(l2["x_mm"]),
            "y_mm": float(l2["y_mm"]),
            "thermal_backend": "five-state",
            "f_sus_trans_pred_ghz": sustainable,
            "f_sus_trans_rom_ghz": sustainable,
            "wire_objective_cycles": float(rounded),
            "wire_aggregation": wire_aggregation,
            "wire_objective": wire_objective,
            "continuous_selected_wire_cycles": continuous,
            "r2_wire_cycles": rounded,
            "layout_delays": derive_layout_delays(
                layout, float(config["frequency"]["f0_ghz"]),
                wire_rounding, weights,
            ),
            "frequency_evidence": search,
        }
        if sustainable is None:
            return None
        sustainable = _finite(sustainable, "sustainable frequency", nonnegative=True)
        objective = -ipc1 * sustainable + lambda_wire * ipc1 * rounded
        return {
            **common,
            "score": objective,
            "objective_loss": objective,
            "bips1_trans_pred": ipc1 * sustainable,
            "bips1_trans_rom_pred": ipc1 * sustainable,
            "temperature_evidence": _peak_evidence(search, sustainable),
        }

    if wire_objective == "discrete-partition":
        search = search_discrete_partitions(
            base_layout=base,
            l2_name=l2_name,
            allowed_tiers=allowed_tiers,
            grid_steps=grid_steps,
            include_fixed_baseline=include_fixed,
            evaluate=evaluate,
        )
        selected = search["selected"]
        search = {**search, "shared_partition_engine": True}
    else:
        # The ordinary transient config has no integer partition requirement.
        # Evaluate a finite, deterministic lattice and use the same score.
        original = l2s[0]
        upper_x = float(base["die_width_mm"]) - float(original["width_mm"])
        upper_y = float(base["die_height_mm"]) - float(original["height_mm"])
        points = []
        for tier in allowed_tiers:
            for yi in range(25):
                for xi in range(25):
                    try:
                        layout = place_l2(
                            base, l2_name, int(tier),
                            upper_x * xi / 24.0, upper_y * yi / 24.0,
                        )
                        candidate = evaluate(layout, "lattice", f"GRID-{yi}-{xi}")
                    except ValueError:
                        candidate = None
                    if candidate is not None:
                        points.append(candidate)
        if not points:
            raise RuntimeError("five-state lattice found no legal selectable placement")
        selected = min(
            points,
            key=lambda item: (
                item["score"], item["tier"], item["y_mm"], item["x_mm"]
            ),
        )
        search = {
            "search_kind": "lattice",
            "lattice_points_per_axis": 25,
            "candidate_count": len(points),
            "candidates": points,
            "fixed_baseline_included": False,
        }
    selected_layout = place_l2(
        base, l2_name, selected["tier"], selected["x_mm"], selected["y_mm"]
    )
    selected_layout["policy"] = "five-state transient discrete-partition proposal"
    output_dir.mkdir(parents=True, exist_ok=True)
    proposed = (output_dir / "proposed_layout.json").resolve()
    write_json(proposed, selected_layout)
    report = {
        "schema_version": 1,
        "mode": "five-state transient layout optimization",
        "thermal_mode": "transient-rom",
        "thermal_backend": "five-state",
        "parameter_status": settings.parameter_status,
        "non_formal": True,
        "paper_equivalent": False,
        "hotspot_calls_inside_optimizer": 0,
        "modules": str(modules_path),
        "power_windows": str(power_windows_path),
        "config": str(config_path),
        "package_dir": None,
        "package_acceptance": None,
        "parameters": {
            "ipc1": ipc1,
            "lambda_wire": lambda_wire,
            "allowed_l2_tiers": allowed_tiers,
            "frequency_grid_ghz": frequencies,
            "wire_objective": wire_objective,
            "wire_aggregation": wire_aggregation,
            "wire_rounding": wire_rounding,
            "tau_core_s": settings.tau_core_s,
            "tau_l2_s": settings.tau_l2_s,
            "score_equation": (
                "-IPC1*f_sus_trans_pred + lambda_wire*IPC1*wire_objective_cycles"
            ),
        },
        "search": {**search, "shared_partition_engine": True},
        "selected": selected,
        "selected_layout": selected_layout,
        "proposed_layout": str(proposed),
    }
    write_json(output_dir / "optimization_report.json", report)
    return report


def optimize_transient_layout(
    modules_path: Path,
    output_dir: Path,
    config_path: Path,
    power_windows_path: Path,
    *,
    backend: str | None = None,
    package_dir: Path | None = None,
    hotspot: Path | None = None,
) -> dict:
    """Dispatch to the configured closed-form or preserved POD-ROM backend."""
    modules_path = Path(modules_path).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    power_windows_path = Path(power_windows_path).resolve()
    config = read_json(config_path)
    configured = parse_five_state_settings(config).backend
    selected = backend or configured
    if selected not in ("five-state", "pod-rom"):
        raise ValueError("backend must be five-state or pod-rom")
    if backend is not None and backend != configured:
        # A CLI override is permitted, but is explicit in the returned report.
        selected = backend
    if selected == "five-state":
        report = _optimize_five_state(
            modules_path, output_dir, config_path, power_windows_path
        )
        report["backend_source"] = "override" if backend else "configuration"
        write_json(output_dir / "optimization_report.json", report)
        return report
    if package_dir is None:
        raise ValueError("pod-rom backend requires package_dir")
    if hotspot is None:
        raise ValueError("pod-rom backend requires hotspot")
    from workflow.transient.rom.optimize_layout import (
        optimize_transient_layout as optimize_pod_layout,
    )

    report = optimize_pod_layout(
        modules_path, Path(package_dir).resolve(), output_dir, config_path,
        power_windows_path, hotspot=Path(hotspot).resolve(),
    )
    return {**report, "thermal_backend": "pod-rom"}
