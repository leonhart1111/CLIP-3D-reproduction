"""Deterministic L2 layout search using an accepted transient ROM package."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from workflow.common import read_json, write_json
from workflow.floorplan.discrete_partition import search_discrete_partitions
from workflow.floorplan.generate_hotspot_inputs import check_geometry
from workflow.floorplan.layout_metrics import (
    aggregate_wire_cycles,
    communication_weights_from_model,
    derive_layout_delays,
    mean_wire_cycles,
    round_wire_cycles,
)
from workflow.transient.rom.calibration_design import (
    calibration_design_hash,
    load_calibration_design,
    require_design_matches_modules,
)
from workflow.transient.rom.contracts import (
    parse_settings,
    require_package_calibration_evidence,
    r1_input_hash_identity,
    rom_input_identity,
)
from workflow.transient.rom.layout_rom import (
    find_rom_sustainable_frequency,
    interpolate_l2_input,
)
from workflow.transient.validation import power_trace_identity


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


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _requested_identity(modules_path: Path, config_path: Path, hotspot: Path,
                        power_windows: dict, config: dict, design: dict) -> dict:
    canonical_source = power_windows.get("canonical_source_r1")
    if not isinstance(canonical_source, str) or not canonical_source:
        raise ValueError("power windows lack canonical_source_r1 provenance")
    canonical_metadata = Path(canonical_source) / "r1_metadata.json"
    if not canonical_metadata.is_file():
        raise FileNotFoundError(canonical_metadata)
    periodic_source = power_windows.get("transient_r1")
    if not isinstance(periodic_source, str) or not periodic_source:
        raise ValueError("power windows lack transient_r1 provenance")
    physical = config.get("physical")
    frequency = config.get("frequency")
    optimizer = config.get("layout_optimizer")
    if not isinstance(physical, dict) or not isinstance(frequency, dict):
        raise ValueError("config lacks physical/frequency identity settings")
    if not isinstance(optimizer, dict):
        raise ValueError("config lacks layout_optimizer identity settings")
    grid_size = physical.get("grid_size")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("physical.grid_size must be a positive integer")
    tiers = optimizer.get("allowed_l2_tiers")
    if tiers != design.get("allowed_l2_tiers"):
        raise ValueError("layout_optimizer.allowed_l2_tiers differs from ROM design")
    return rom_input_identity(
        canonical_r1_metadata_hash=_sha256(canonical_metadata),
        power_trace=power_trace_identity(power_windows),
        modules_geometry_hash=_sha256(modules_path),
        layout_geometry_hash=_json_sha256(design.get("base_layout")),
        configuration_hash=_sha256(config_path),
        hotspot_hash=_sha256(hotspot),
        grid={"rows": grid_size, "columns": grid_size},
        stack=physical.get("thermal_stack"),
        cooling={
            "ambient_c": frequency.get("ambient_c"),
            "r_convec_k_per_w": physical.get("r_convec_k_per_w"),
        },
        allowed_l2_tiers=tiers,
        calibration_design_hash=calibration_design_hash(design),
        r1_input_hashes=r1_input_hash_identity(
            Path(canonical_source), Path(periodic_source)
        ),
    )


def canonical_frequency_grid(config: dict) -> list[float]:
    """Return the configured finite, endpoint-bounded ROM frequency grid."""
    frequency = config.get("frequency")
    if not isinstance(frequency, dict):
        raise ValueError("frequency grid config lacks frequency settings")

    def endpoint(name: str) -> float:
        value = frequency.get(name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"frequency grid {name} must be finite and positive")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"frequency grid {name} must be finite and positive")
        return result

    fmin = endpoint("fmin_ghz")
    f0 = endpoint("f0_ghz")
    if fmin > f0:
        raise ValueError("frequency grid fmin_ghz must not exceed f0_ghz")
    rom = config.get("transient_rom", {})
    if rom is None:
        rom = {}
    if not isinstance(rom, dict):
        raise ValueError("frequency grid transient_rom must be a dictionary")
    values = rom.get("frequencies_ghz", frequency.get("grid_ghz"))
    if values is None:
        values = [fmin, f0]
    if not isinstance(values, list) or not values:
        raise ValueError("frequency grid must be a non-empty list")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("frequency grid values must be finite and positive")
        normalized = float(value)
        if (not math.isfinite(normalized) or normalized <= 0.0
                or normalized < fmin or normalized > f0):
            raise ValueError("frequency grid values must lie within configured bounds")
        result.append(normalized)
    result = sorted(set(result))
    if fmin not in result or f0 not in result:
        raise ValueError("frequency grid must include fmin_ghz and f0_ghz")
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
                              output_dir: Path, config_path: Path,
                              power_windows_path: Path, *, hotspot: Path) -> dict:
    """Search a fixed 25x25 lattice and five local refinements with ROM calls only."""
    modules_path = Path(modules_path).resolve()
    package_dir = Path(package_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    power_windows_path = Path(power_windows_path).resolve()
    hotspot = Path(hotspot).resolve()
    for path in (modules_path, config_path, power_windows_path, hotspot):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not package_dir.is_dir():
        raise FileNotFoundError(package_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"ROM optimizer output directory is not empty: {output_dir}")

    modules = read_json(modules_path)
    if not isinstance(modules, dict) or not isinstance(modules.get("modules"), list):
        raise ValueError("modules must contain a module list")
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a dictionary")
    power_windows = read_json(power_windows_path)
    if not isinstance(power_windows, dict):
        raise ValueError("power windows must be a dictionary")
    ipc1 = _finite(modules.get("ipc1"), "modules IPC1", minimum=0.0)
    if ipc1 <= 0.0:
        raise ValueError("modules IPC1 must be finite and positive")

    settings = parse_settings(config)
    optimizer = config.get("layout_optimizer", {})
    tiers = optimizer.get("allowed_l2_tiers") if isinstance(optimizer, dict) else None
    design = load_calibration_design(package_dir)
    require_design_matches_modules(design, modules)
    identity = _requested_identity(
        modules_path, config_path, hotspot, power_windows, config, design,
    )
    package_evidence = require_package_calibration_evidence(
        package_dir, identity, settings,
    )
    acceptance = package_evidence["acceptance"]
    model = package_evidence["model"]
    model_metadata = package_evidence["model_metadata"]
    design_identity = calibration_design_hash(design)
    training_ids = sorted(point["id"] for point in design["training"])
    if sorted(model.b_l2_anchors) != training_ids:
        raise ValueError("saved calibration design anchors differ from POD B_L2 columns")
    if model_metadata.get("calibration_design_hash") != design_identity:
        raise ValueError("POD model calibration design hash differs")
    if model_metadata.get("case_order") != training_ids:
        raise ValueError("POD model case order differs from saved calibration design")
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

    delay = config.get("delay", {})
    if not isinstance(optimizer, dict) or not isinstance(delay, dict):
        raise ValueError("config layout_optimizer and delay must be dictionaries")
    lambda_wire = _finite(
        optimizer.get("lambda_wire"), "layout_optimizer.lambda_wire", minimum=0.0
    )
    wire_objective = optimizer.get("wire_objective", "continuous")
    if wire_objective not in (
        "continuous", "r2-quantized", "discrete-partition"
    ):
        raise ValueError("layout_optimizer.wire_objective is invalid")
    if wire_objective != "discrete-partition":
        if settings.search_grid_points_per_axis != _LATTICE_POINTS_PER_AXIS:
            raise ValueError("transient ROM search_grid_points_per_axis must equal 25")
        if settings.refinement_starts != _REFINEMENT_STARTS:
            raise ValueError("transient ROM refinement_starts must equal 5")
    wire_aggregation = delay.get("wire_aggregation", "mean")
    if wire_aggregation not in ("mean", "maximum", "traffic-weighted"):
        raise ValueError("delay.wire_aggregation is invalid")
    wire_rounding = delay.get("wire_rounding", "nearest")
    communication_weights = communication_weights_from_model(
        modules, required=wire_aggregation == "traffic-weighted"
    )
    frequencies = canonical_frequency_grid(config)
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
        rounded_wire = round_wire_cycles(continuous_wire, wire_rounding)
        candidate = {
            **point,
            "f_sus_trans_rom_ghz": sustainable,
            "wire_objective_cycles": objective_wire,
            "wire_aggregation": wire_aggregation,
            "wire_objective": wire_objective,
            "continuous_selected_wire_cycles": continuous_wire,
            "r2_wire_cycles": rounded_wire,
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
        candidate["objective_loss"] = candidate["score"]
        if wire_aggregation == "traffic-weighted":
            candidate["traffic_weighted_wire_cycles"] = continuous_wire
        return candidate

    partition_search = None
    if wire_objective == "discrete-partition":
        partition_grid_steps = optimizer.get("partition_grid_steps", 41)
        include_fixed_baseline = optimizer.get("include_fixed_baseline", True)

        def evaluate_partition(layout: dict, origin: str, start: str) -> dict | None:
            l2 = next(module for module in layout["modules"] if module["kind"] == "l2")
            candidate = evaluate(
                int(l2["tier"]), float(l2["x_mm"]), float(l2["y_mm"]),
                origin,
            )
            if candidate is None or candidate["score"] is None:
                return None
            candidate["origin"] = origin
            candidate["start"] = start
            return candidate

        partition_search = search_discrete_partitions(
            base_layout=base,
            l2_name=l2_name,
            allowed_tiers=tiers,
            grid_steps=partition_grid_steps,
            include_fixed_baseline=include_fixed_baseline,
            evaluate=evaluate_partition,
        )
        selected = partition_search["selected"]
        search_report = {
            **partition_search,
            "shared_partition_engine": True,
            "rejections": rejections,
        }
    else:
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
            key=lambda item: (
                item["score"], item["tier"], item["x_mm"], item["y_mm"]
            ),
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
                    if key in seen or (
                        x_mm == current["x_mm"] and y_mm == current["y_mm"]
                    ):
                        continue
                    seen.add(key)
                    candidate = evaluate(
                        seed["tier"], x_mm, y_mm, f"refinement-{index}"
                    )
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
            key=lambda item: (
                item["score"], item["tier"], item["x_mm"], item["y_mm"]
            ),
        )
        search_report = {
            "lattice_points_per_axis": _LATTICE_POINTS_PER_AXIS,
            "lattice_points_attempted": len(tiers) * _LATTICE_POINTS_PER_AXIS ** 2,
            "legal_seeds": legal_seeds,
            "rejections": rejections,
            "refinement_starts": _REFINEMENT_STARTS,
            "coordinate_tolerance_mm": coordinate_tolerance,
            "refinements": refinements,
        }
    selected_layout = _layout(
        base, l2_name, selected["tier"], selected["x_mm"], selected["y_mm"]
    )
    selected_layout["policy"] = (
        "accepted transient ROM integer-cycle partition proposal"
        if wire_objective == "discrete-partition"
        else "accepted transient ROM grid-plus-refinement proposal"
    )
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
        "power_windows": str(power_windows_path),
        "config": str(config_path),
        "hotspot": str(hotspot),
        "package_acceptance": acceptance,
        "package_calibration_cases": package_evidence["cases"],
        "package_holdout_validation": package_evidence["validation"],
        "requested_identity": identity,
        "calibration_design_hash": design_identity,
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
        "search": search_report,
        "selected": selected,
        "selected_layout": selected_layout,
        "proposed_layout": str(proposed_path),
    }
    if partition_search is not None:
        write_json(output_dir / "partition_search.json", search_report)
    write_json(output_dir / "optimization_report.json", report)
    return report
