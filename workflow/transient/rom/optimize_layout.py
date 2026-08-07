"""Deterministic L2 layout search using an accepted transient ROM package."""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any

from workflow.common import read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import check_geometry
from workflow.floorplan.layout_metrics import (
    aggregate_wire_cycles,
    communication_weights_from_model,
    derive_layout_delays,
    mean_wire_cycles,
    round_wire_cycles,
)
from workflow.transient.rom.calibration_design import build_design
from workflow.transient.rom.contracts import parse_settings, require_accepted_package
from workflow.transient.rom.layout_rom import (
    find_rom_sustainable_frequency,
    interpolate_l2_input,
)
from workflow.transient.rom.pod_state_space import load_model


_LATTICE_POINTS_PER_AXIS = 25
_REFINEMENT_STARTS = 5
_NEIGHBORS = (
    (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
    (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be a finite number not less than {minimum}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _accepted_package(package_dir: Path, modules_path: Path) -> tuple[dict, dict]:
    try:
        raw = read_json(package_dir / "rom_acceptance.json")
    except (OSError, ValueError):
        raw = {}
    identity = raw.get("identity") if isinstance(raw, dict) else None
    requested = dict(identity) if isinstance(identity, dict) else {}
    requested["modules_geometry_hash"] = _sha256(modules_path)
    acceptance = require_accepted_package(package_dir, requested)
    return acceptance, requested


def _package_power_windows(package_dir: Path) -> tuple[Path, dict]:
    cases = read_json(package_dir / "calibration_cases.json")
    source = cases.get("source_power_windows") if isinstance(cases, dict) else None
    if not isinstance(source, str) or not source:
        raise ValueError("ROM package lacks source_power_windows provenance")
    path = Path(source)
    if not path.is_absolute():
        path = package_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, read_json(path)


def _frequency_grid(config: dict) -> list[float]:
    frequency = config.get("frequency")
    if not isinstance(frequency, dict):
        raise ValueError("config lacks frequency settings")
    rom = config.get("transient_rom", {})
    if rom is None:
        rom = {}
    if not isinstance(rom, dict):
        raise ValueError("transient_rom must be a dictionary")
    values = rom.get("frequencies_ghz", frequency.get("grid_ghz"))
    if values is None:
        values = [frequency.get("fmin_ghz"), frequency.get("f0_ghz")]
    if not isinstance(values, list) or not values:
        raise ValueError("transient ROM frequencies_ghz must be a non-empty list")
    result = sorted({_finite(value, "frequencies_ghz", minimum=0.0) for value in values})
    if not result or result[0] <= 0.0:
        raise ValueError("frequencies_ghz must contain finite positive numbers")
    return result


def _layout(base: dict, l2_name: str, tier: int, x_mm: float, y_mm: float) -> dict:
    modules = []
    found = 0
    for module in base["modules"]:
        if module.get("name") == l2_name and module.get("kind") == "l2":
            modules.append(dict(module, tier=tier, x_mm=x_mm, y_mm=y_mm))
            found += 1
        else:
            modules.append(dict(module))
    if found != 1:
        raise ValueError("ROM design base layout lacks exactly one designed L2")
    return {**base, "modules": modules}


def _temperature_evidence(search: dict, sustainable: float) -> dict:
    evaluations = search.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("ROM frequency search lacks evaluation evidence")
    matches = [value for value in evaluations if isinstance(value, dict)
               and value.get("frequency_ghz") == sustainable]
    if len(matches) != 1:
        raise ValueError("ROM frequency search lacks unique sustainable-temperature evidence")
    return matches[0]


def optimize_transient_layout(modules_path: Path, package_dir: Path,
                              output_dir: Path, config: dict) -> dict:
    """Search a fixed 25x25 lattice and five local refinements with ROM calls only."""
    modules_path = Path(modules_path).resolve()
    package_dir = Path(package_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not modules_path.is_file():
        raise FileNotFoundError(modules_path)
    if not package_dir.is_dir():
        raise FileNotFoundError(package_dir)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a dictionary")

    acceptance, identity = _accepted_package(package_dir, modules_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"ROM optimizer output directory is not empty: {output_dir}")
    source_power_path, power_windows = _package_power_windows(package_dir)
    model, model_metadata = load_model(package_dir / "pod_model.npz")
    modules = read_json(modules_path)
    if not isinstance(modules, dict) or not isinstance(modules.get("modules"), list):
        raise ValueError("modules must contain a module list")
    ipc1 = _finite(modules.get("ipc1"), "modules IPC1", minimum=0.0)
    if ipc1 <= 0.0:
        raise ValueError("modules IPC1 must be finite and positive")

    settings = parse_settings(config)
    if settings.search_grid_points_per_axis != _LATTICE_POINTS_PER_AXIS:
        raise ValueError("transient ROM search_grid_points_per_axis must equal 25")
    if settings.refinement_starts != _REFINEMENT_STARTS:
        raise ValueError("transient ROM refinement_starts must equal 5")
    tiers = identity["allowed_l2_tiers"]
    design = build_design(modules, tiers, settings)
    base = design["base_layout"]
    l2_name = design["l2_name"]
    original = next(
        module for module in base["modules"]
        if module.get("kind") == "l2" and module.get("name") == l2_name
    )
    upper_x = float(base["die_width_mm"]) - float(original["width_mm"])
    upper_y = float(base["die_height_mm"]) - float(original["height_mm"])
    if min(upper_x, upper_y) < 0.0:
        raise ValueError("L2 geometry does not fit inside the die")

    optimizer = config.get("layout_optimizer", {})
    delay = config.get("delay", {})
    if not isinstance(optimizer, dict) or not isinstance(delay, dict):
        raise ValueError("config layout_optimizer and delay must be dictionaries")
    lambda_wire = _finite(
        optimizer.get("lambda_wire"), "layout_optimizer.lambda_wire", minimum=0.0
    )
    wire_objective = optimizer.get("wire_objective", "continuous")
    if wire_objective not in ("continuous", "r2-quantized"):
        raise ValueError("layout_optimizer.wire_objective is invalid")
    wire_aggregation = delay.get("wire_aggregation", "mean")
    if wire_aggregation not in ("mean", "maximum", "traffic-weighted"):
        raise ValueError("delay.wire_aggregation is invalid")
    wire_rounding = delay.get("wire_rounding", "nearest")
    communication_weights = communication_weights_from_model(
        modules, required=wire_aggregation == "traffic-weighted"
    )
    frequencies = _frequency_grid(config)
    rejections: list[dict] = []

    def evaluate(tier: int, x_mm: float, y_mm: float, stage: str) -> dict | None:
        layout = _layout(base, l2_name, tier, float(x_mm), float(y_mm))
        point = {
            "stage": stage, "tier": tier,
            "x_mm": float(x_mm), "y_mm": float(y_mm),
        }
        try:
            check_geometry(layout["modules"], float(base["die_width_mm"]))
        except ValueError as error:
            rejections.append({**point, "category": "geometry", "reason": str(error)})
            return None
        try:
            interpolate_l2_input(model, design, tier, float(x_mm), float(y_mm))
        except ValueError as error:
            rejections.append({
                **point, "category": "interpolation_domain", "reason": str(error)
            })
            return None

        search = find_rom_sustainable_frequency(
            model, design, power_windows, layout, frequencies, settings, config
        )
        sustainable = search.get("sustainable_frequency_ghz")
        layout_delays = derive_layout_delays(
            layout, float(config["frequency"]["f0_ghz"]),
            wire_rounding, communication_weights,
        )
        _, per_core = mean_wire_cycles(
            layout["modules"], float(config["frequency"]["f0_ghz"])
        )
        continuous_wire = aggregate_wire_cycles(
            per_core, wire_aggregation, communication_weights
        )
        objective_wire = (
            continuous_wire if wire_objective == "continuous"
            else float(round_wire_cycles(continuous_wire, wire_rounding))
        )
        candidate = {
            **point,
            "f_sus_trans_rom_ghz": sustainable,
            "wire_objective_cycles": objective_wire,
            "wire_aggregation": wire_aggregation,
            "wire_objective": wire_objective,
            "layout_delays": layout_delays,
            "frequency_evidence": search,
        }
        if sustainable is None:
            candidate.update({
                "score": None, "bips1_trans_rom_pred": None,
                "temperature_evidence": None,
            })
            rejections.append({
                **point, "category": "frequency",
                "reason": "ROM found no sustainable frequency on the configured grid",
            })
            return candidate
        sustainable = _finite(sustainable, "ROM sustainable frequency", minimum=0.0)
        temperature = _temperature_evidence(search, sustainable)
        candidate.update({
            "score": -ipc1 * sustainable + lambda_wire * ipc1 * objective_wire,
            "bips1_trans_rom_pred": ipc1 * sustainable,
            "temperature_evidence": temperature,
        })
        return candidate

    legal_seeds = []
    for tier in tiers:
        for y_index in range(_LATTICE_POINTS_PER_AXIS):
            y_mm = upper_y * y_index / (_LATTICE_POINTS_PER_AXIS - 1)
            for x_index in range(_LATTICE_POINTS_PER_AXIS):
                x_mm = upper_x * x_index / (_LATTICE_POINTS_PER_AXIS - 1)
                candidate = evaluate(tier, x_mm, y_mm, "lattice")
                if candidate is not None:
                    legal_seeds.append(candidate)

    selectable = [candidate for candidate in legal_seeds
                  if candidate["score"] is not None]
    if len(selectable) < _REFINEMENT_STARTS:
        raise RuntimeError(
            "ROM optimizer found fewer than five selectable legal lattice seeds"
        )
    ranked = sorted(
        selectable,
        key=lambda item: (item["score"], item["tier"], item["x_mm"], item["y_mm"]),
    )
    seeds = ranked[:_REFINEMENT_STARTS]
    lattice_step = max(
        upper_x / (_LATTICE_POINTS_PER_AXIS - 1),
        upper_y / (_LATTICE_POINTS_PER_AXIS - 1),
    )
    coordinate_tolerance = max(max(upper_x, upper_y) * 1e-5, 1e-6)
    refinements = []
    refined_candidates = []
    for index, seed in enumerate(seeds):
        current = seed
        step = max(lattice_step / 2.0, coordinate_tolerance * 2.0)
        iterations = 0
        legal_trials = []
        while step > coordinate_tolerance:
            trials = []
            seen = set()
            for dx, dy in _NEIGHBORS:
                x_mm = min(max(current["x_mm"] + dx * step, 0.0), upper_x)
                y_mm = min(max(current["y_mm"] + dy * step, 0.0), upper_y)
                key = (x_mm.hex(), y_mm.hex())
                if key in seen or (x_mm == current["x_mm"] and y_mm == current["y_mm"]):
                    continue
                seen.add(key)
                candidate = evaluate(seed["tier"], x_mm, y_mm, f"refinement-{index}")
                if candidate is not None:
                    legal_trials.append(candidate)
                    if candidate["score"] is not None:
                        trials.append(candidate)
            best = min(
                trials,
                key=lambda item: (item["score"], item["x_mm"], item["y_mm"]),
                default=None,
            )
            if best is not None and best["score"] < current["score"]:
                current = best
            else:
                step /= 2.0
            iterations += 1
            if iterations >= 1000:
                raise RuntimeError("ROM pattern refinement exceeded 1000 iterations")
        refinements.append({
            "index": index,
            "seed": seed,
            "selected": current,
            "iterations": iterations,
            "legal_candidates": legal_trials,
        })
        refined_candidates.append(current)

    selected = min(
        [*selectable, *refined_candidates],
        key=lambda item: (item["score"], item["tier"], item["x_mm"], item["y_mm"]),
    )
    selected_layout = _layout(
        base, l2_name, selected["tier"], selected["x_mm"], selected["y_mm"]
    )
    selected_layout["policy"] = "accepted transient ROM grid-plus-refinement proposal"
    output_dir.mkdir(parents=True, exist_ok=True)
    proposed_path = output_dir / "proposed_layout.json"
    write_json(proposed_path, selected_layout)
    report = {
        "schema_version": 1,
        "mode": "transient ROM layout optimization",
        "thermal_mode": "transient-rom",
        "non_formal": True,
        "paper_equivalent": False,
        "hotspot_calls_inside_optimizer": 0,
        "modules": str(modules_path),
        "package_dir": str(package_dir),
        "source_power_windows": str(source_power_path),
        "package_acceptance": acceptance,
        "model_metadata": model_metadata,
        "parameters": {
            "ipc1": ipc1,
            "lambda_wire": lambda_wire,
            "allowed_l2_tiers": tiers,
            "frequency_grid_ghz": frequencies,
            "wire_objective": wire_objective,
            "wire_aggregation": wire_aggregation,
            "wire_rounding": wire_rounding,
            "score_equation": (
                "-IPC1*f_sus_trans_rom + lambda_wire*IPC1*wire_objective_cycles"
            ),
        },
        "search": {
            "lattice_points_per_axis": _LATTICE_POINTS_PER_AXIS,
            "lattice_points_attempted": len(tiers) * _LATTICE_POINTS_PER_AXIS ** 2,
            "legal_seeds": legal_seeds,
            "rejections": rejections,
            "refinement_starts": _REFINEMENT_STARTS,
            "coordinate_tolerance_mm": coordinate_tolerance,
            "refinements": refinements,
        },
        "selected": selected,
        "selected_layout": selected_layout,
        "proposed_layout": str(proposed_path),
    }
    write_json(output_dir / "optimization_report.json", report)
    return report
