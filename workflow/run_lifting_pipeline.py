#!/usr/bin/env python3
"""Run one complete CLIP-3D lifting/thermal/layout/R2 point."""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import subprocess
import time
from pathlib import Path

from workflow.cacti.characterize_cache import characterize
from workflow.common import PROJECT_ROOT, format_temperature_c, read_json, write_json
from workflow.floorplan.build_module_model import build_model
from workflow.floorplan.comparison_layouts import METHODS as COMPARISON_METHODS
from workflow.floorplan.comparison_layouts import generate as generate_comparison_layouts
from workflow.floorplan.generate_hotspot_inputs import materialize
from workflow.floorplan.layout_metrics import (
    communication_weights_from_model,
    derive_layout_delays,
    select_rounded_wire_cycles,
)
from workflow.floorplan.optimize_layout import optimize
from workflow.mcpat.gem5_to_mcpat import convert
from workflow.mcpat.parse_mcpat import parse_mcpat_text
from workflow.r2.build_latency_vector import build_vector
from workflow.r2.run_r2 import run as run_r2
from workflow.thermal.run_hotspot import run_hotspot
from workflow.thermal.sustainable_frequency import evaluate


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/clip3d_pipeline.json"
LAYOUT_METHODS = ("fixed-bin", "clip3d", *COMPARISON_METHODS)


def boolean_text(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def validate_r1(r1_dir: Path) -> None:
    missing = [name for name in ("r1_metadata.json", "stats.txt")
               if not (r1_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"R1 directory lacks: {', '.join(missing)}")
    status = r1_dir / "status.json"
    if status.is_file() and read_json(status).get("state") not in (None, "success"):
        raise ValueError(f"R1 status is not success: {status}")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one formal-validation artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def validate_accepted_strict_p1(config: dict) -> None:
    """Verify that an accepted strict-P1 config still derives from its reports."""
    formal = config.get("formal_validation", {})
    if formal.get("promotion") != "workflow.analysis.promote_validated_config only":
        raise ValueError("accepted strict-P1 config has invalid promotion provenance")
    artifacts = formal.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("accepted strict-P1 config requires formal_validation.artifacts")

    reports = {}
    for name in ("proxy_report", "wire_summary", "frequency_report"):
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict):
            raise ValueError(f"accepted strict-P1 config requires {name} artifact")
        path_text = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(path_text, str) or not Path(path_text).is_absolute():
            raise ValueError(f"accepted strict-P1 {name} artifact path must be absolute")
        if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"accepted strict-P1 {name} artifact sha256 must be 64 lowercase hex characters")
        path = Path(path_text)
        if not path.is_file():
            raise ValueError(f"accepted strict-P1 {name} artifact is missing: {path}")
        if sha256(path) != digest:
            raise ValueError(f"accepted strict-P1 {name} artifact sha256 does not match")
        report = read_json(path)
        if not isinstance(report, dict) or report.get("recommendation", {}).get("accepted") is not True:
            raise ValueError(f"accepted strict-P1 {name} report is not accepted")
        reports[name] = report

    proxy = reports["proxy_report"]
    if proxy.get("strict_p1", {}).get("beta_status") != "fixed_unidentifiable_under_p1":
        raise ValueError("accepted strict-P1 proxy beta status is invalid")
    optimizer = config["layout_optimizer"]
    proxy_parameters = proxy.get("fit", {}).get("parameters", {})
    expected = {
        "alpha": {
            "artifact": "proxy_report",
            "field": "fit.parameters.alpha",
            "value": finite_number(proxy_parameters.get("alpha"), "proxy alpha"),
        },
        "cross_tier_weight": {
            "artifact": "proxy_report",
            "field": "fit.parameters.cross_tier_weight",
            "value": finite_number(
                proxy_parameters.get("cross_tier_weight"), "proxy cross_tier_weight"
            ),
        },
        "lambda_wire": {
            "artifact": "wire_summary",
            "field": "selected_lambda_wire",
            "value": finite_number(
                reports["wire_summary"].get("selected_lambda_wire"), "wire selected_lambda_wire"
            ),
        },
        "beta": {"source": "fixed_unidentifiable_under_p1", "value": 0.0},
    }
    provenance = optimizer.get("parameter_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("accepted strict-P1 config requires parameter_provenance")
    for parameter, record in expected.items():
        if provenance.get(parameter) != record:
            raise ValueError(
                f"accepted strict-P1 {parameter} parameter_provenance does not match reports"
            )
        if finite_number(optimizer.get(parameter), parameter) != record["value"]:
            raise ValueError(f"accepted strict-P1 {parameter} does not match report provenance")


def validate_config(config: dict, layout_method: str) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported CLIP-3D pipeline config schema")
    if layout_method not in LAYOUT_METHODS:
        raise ValueError(f"unknown layout method: {layout_method}")
    cacti_config = config.get("cacti", {})
    if "use_paper_table_ii" in cacti_config:
        raise ValueError(
            "cacti.use_paper_table_ii has been removed; cache area and latency "
            "must come from the local CACTI run"
        )
    physical_r = float(config["physical"]["r_convec_k_per_w"])
    optimizer_r = float(config["layout_optimizer"]["r_convec_k_per_w"])
    if layout_method != "fixed-bin" and not abs(physical_r - optimizer_r) < 1e-12:
        raise ValueError(
            "layout search and final HotSpot must use the same R_conv: "
            f"optimizer={optimizer_r}, physical={physical_r}"
        )
    allowed_mcpat = {
        "temperature_k", "device_type", "longer_channel_device",
        "interconnect_projection_type", "opt_for_clk",
    }
    unknown_mcpat = set(config.get("mcpat", {})) - allowed_mcpat
    if unknown_mcpat:
        raise ValueError(f"unsupported mcpat settings: {sorted(unknown_mcpat)}")
    tolerance = float(config.get("layout_optimizer", {}).get(
        "baseline_guard_bips_tolerance", 1e-9
    ))
    if tolerance < 0:
        raise ValueError("layout_optimizer.baseline_guard_bips_tolerance must be non-negative")
    validation_policy = config.get("layout_optimizer", {}).get(
        "validation_policy", "guarded"
    )
    if validation_policy not in ("guarded", "paper-single"):
        raise ValueError("layout_optimizer.validation_policy must be guarded or paper-single")
    allowed_tiers = config.get("layout_optimizer", {}).get("allowed_l2_tiers", [0, 1])
    if not allowed_tiers or any(int(tier) not in (0, 1) for tier in allowed_tiers):
        raise ValueError("layout_optimizer.allowed_l2_tiers must contain tier 0 and/or tier 1")
    aggregation = config.get("delay", {}).get("wire_aggregation", "mean")
    if aggregation not in ("mean", "maximum", "traffic-weighted"):
        raise ValueError(
            "delay.wire_aggregation must be mean, maximum, or traffic-weighted"
        )
    if (aggregation == "traffic-weighted"
            and config.get("formal_validation", {}).get("accepted") is True):
        raise ValueError(
            "traffic-weighted wire aggregation is a non-formal research extension"
        )
    if layout_method == "clip3d" and aggregation == "maximum":
        raise ValueError(
            "maximum wire aggregation is only a conservative R2 sensitivity mode, "
            "not a CLIP-3D optimizer objective"
        )
    if aggregation == "traffic-weighted" and layout_method in COMPARISON_METHODS:
        raise ValueError(
            "traffic-weighted aggregation is not supported by comparison layout "
            "methods because their candidate selection uses a different objective"
        )
    if config.get("formal_validation", {}).get("strict_p1") is True:
        if list(allowed_tiers) != [1]:
            raise ValueError("strict P1 requires layout_optimizer.allowed_l2_tiers == [1]")
        if validation_policy != "paper-single":
            raise ValueError("strict P1 requires layout_optimizer.validation_policy == paper-single")
        if float(config.get("layout_optimizer", {}).get("beta", 0.0)) != 0.0:
            raise ValueError("strict P1 requires layout_optimizer.beta == 0.0")
        if config.get("formal_validation", {}).get("accepted") is True:
            validate_accepted_strict_p1(config)
    proxy_model = config.get("layout_optimizer", {}).get(
        "proxy_spatial_model", "center"
    )
    if proxy_model not in ("center", "area-quadrature"):
        raise ValueError("layout_optimizer.proxy_spatial_model is invalid")
    quadrature_order = int(config.get("layout_optimizer", {}).get(
        "proxy_quadrature_order", 2
    ))
    if quadrature_order not in (1, 2, 3):
        raise ValueError("layout_optimizer.proxy_quadrature_order must be 1, 2, or 3")
    wire_objective = config.get("layout_optimizer", {}).get(
        "wire_objective", "continuous"
    )
    if wire_objective not in ("continuous", "r2-quantized"):
        raise ValueError("layout_optimizer.wire_objective is invalid")


def refresh_copied_hotspot_paths(hotspot_dir: Path) -> None:
    manifest_path = hotspot_dir / "hotspot_manifest.json"
    manifest = read_json(manifest_path)
    manifest["layout"] = str((hotspot_dir / "layout.json").resolve())
    for name in manifest["files"]:
        manifest["files"][name] = str((hotspot_dir / name).resolve())
    write_json(manifest_path, manifest)
    thermal_path = hotspot_dir / "thermal_result.json"
    thermal = read_json(thermal_path)
    thermal["power_trace"] = str((hotspot_dir / "power.ptrace").resolve())
    thermal["steady_file"] = str((hotspot_dir / "steady.txt").resolve())
    thermal["grid_steady_file"] = str((hotspot_dir / "grid.steady.txt").resolve())
    write_json(thermal_path, thermal)


def evaluate_comparison_candidates(modules_path: Path, output_dir: Path,
                                   config: dict, method: str) -> tuple[Path, dict, dict]:
    frequency = config["frequency"]
    physical = config["physical"]
    optimizer = config["layout_optimizer"]
    comparison = config["comparison_layouts"]
    search_dir = output_dir / "layout_search"
    report = generate_comparison_layouts(
        modules_path, search_dir, method, physical["utilization"],
        comparison["candidate_grid"], comparison["top_k_hotspot"],
        frequency["ambient_c"], physical["r_convec_k_per_w"],
        optimizer["alpha"], optimizer["beta"], optimizer["cross_tier_weight"],
        frequency["f0_ghz"], comparison["sa_iterations"], comparison["sa_seed"],
    )
    evaluations = []
    for emitted in report["emitted"]:
        candidate_root = output_dir / "layout_candidates" / f"candidate_{emitted['rank']:02d}"
        hotspot_dir = candidate_root / "hotspot"
        layout_path = Path(emitted["layout"])
        materialize(
            modules_path, hotspot_dir, physical["grid_size"], physical["utilization"],
            frequency["ambient_c"], physical["r_convec_k_per_w"], layout_path,
            physical.get("thermal_stack"),
        )
        thermal = run_hotspot(hotspot_dir)
        performance = evaluate(
            modules_path, hotspot_dir / "thermal_result.json",
            candidate_root / "performance.json", frequency["f0_ghz"],
            frequency["fmin_ghz"], frequency["tsafe_c"], frequency["ambient_c"],
        )
        layout_delays = derive_layout_delays(
            read_json(hotspot_dir / "layout.json"), frequency["f0_ghz"],
            config["delay"].get("wire_rounding", "nearest"),
        )
        evaluations.append({
            "rank": emitted["rank"], "layout": emitted["layout"],
            "hotspot_dir": str(hotspot_dir.resolve()), "tmax_c": thermal["tmax_c"],
            "sustainable_frequency_ghz": performance["sustainable_frequency_ghz"],
            "wire_cycles_unrounded": layout_delays["wire_cycles_unrounded"],
            "wire_cycles": layout_delays["wire_cycles"],
            "tsv_hops": layout_delays["tsv_hops"],
        })

    if method == "cool3d-standard":
        selected = min(evaluations, key=lambda item: (
            item["tmax_c"], item["wire_cycles_unrounded"], item["rank"]))
        selection_rule = "minimum real HotSpot Tmax; wire delay breaks ties"
    else:
        temp_values = [item["tmax_c"] for item in evaluations]
        wire_values = [item["wire_cycles_unrounded"] for item in evaluations]
        tlo, thi = min(temp_values), max(temp_values)
        wlo, whi = min(wire_values), max(wire_values)
        weight = float(comparison["sa_selection_lambda"])
        for item in evaluations:
            nt = (item["tmax_c"] - tlo) / (thi - tlo) if thi > tlo else 0.0
            nw = ((item["wire_cycles_unrounded"] - wlo) / (whi - wlo)
                  if whi > wlo else 0.0)
            item["selection_score"] = weight * nt + (1.0 - weight) * nw
        selected = min(evaluations, key=lambda item: (item["selection_score"], item["rank"]))
        selection_rule = f"minimum normalized lambda surrogate, lambda_thermal={weight}"

    source = Path(selected["hotspot_dir"])
    destination = output_dir / "hotspot"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    refresh_copied_hotspot_paths(destination)
    selection = {
        "schema_version": 1, "method": method, "selection_rule": selection_rule,
        "selected": selected, "evaluations": evaluations,
        "search_report": str((search_dir / "search_report.json").resolve()),
        "hotspot_solves": len(evaluations),
    }
    write_json(output_dir / "layout_selection.json", selection)
    return destination / "layout.json", read_json(destination / "thermal_result.json"), selection


def select_clip3d_candidate(evaluations: list[dict],
                            bips_tolerance: float = 1e-9) -> tuple[dict, str]:
    """Select a HotSpot-validated CLIP proposal without regressing fixed-bin.

    Thermal BIPS1 is the primary metric because real R2 IPC is not available at
    layout-selection time.  Only an effective tie is broken by the discrete
    wire latency that R2 will actually consume, and then by real Tmax.
    """
    if bips_tolerance < 0:
        raise ValueError("CLIP-3D baseline-guard tolerance must be non-negative")
    by_policy = {item["policy"]: item for item in evaluations}
    if set(by_policy) != {"fixed-bin", "optimized"}:
        raise ValueError("CLIP-3D validation requires fixed-bin and optimized evaluations")
    fixed = by_policy["fixed-bin"]
    optimized = by_policy["optimized"]
    aggregation = fixed.get("r2_wire_aggregation", "mean")
    if optimized.get("r2_wire_aggregation", aggregation) != aggregation:
        raise ValueError("CLIP-3D candidates use different R2 wire aggregations")
    aggregation_label = {
        "mean": "mean",
        "maximum": "maximum",
        "traffic-weighted": "traffic-weighted",
    }.get(aggregation, str(aggregation))
    delta = optimized["bips1_thermal"] - fixed["bips1_thermal"]
    if delta > bips_tolerance:
        return optimized, "higher real-HotSpot thermal BIPS1"
    if delta < -bips_tolerance:
        return fixed, "baseline guard: optimized real-HotSpot thermal BIPS1 is lower"

    fixed_wire = int(fixed.get("r2_wire_cycles", fixed["wire_cycles"]))
    optimized_wire = int(optimized.get("r2_wire_cycles", optimized["wire_cycles"]))
    if optimized_wire < fixed_wire:
        return optimized, f"thermal BIPS1 tie; lower discrete {aggregation_label} wire latency"
    if optimized_wire > fixed_wire:
        return fixed, (
            f"thermal BIPS1 tie; fixed-bin has lower discrete "
            f"{aggregation_label} wire latency"
        )
    if optimized["tmax_c"] < fixed["tmax_c"]:
        return optimized, "thermal BIPS1 and wire-latency tie; lower real HotSpot Tmax"
    return fixed, "thermal BIPS1 and wire-latency tie; fixed-bin has no higher real Tmax"


def evaluate_clip3d_candidate(modules_path: Path, proposed_layout: Path,
                              output_dir: Path, config: dict,
                              hotspot_tool: Path) -> tuple[Path, dict, dict]:
    """Validate both the optimizer proposal and fixed-bin with real HotSpot."""
    frequency = config["frequency"]
    physical = config["physical"]
    optimizer = config["layout_optimizer"]
    delay = config["delay"]
    model = read_json(modules_path)
    aggregation = delay.get("wire_aggregation", "mean")
    communication_weights = communication_weights_from_model(
        model, required=aggregation == "traffic-weighted"
    )
    evaluations = []
    validation_root = output_dir / "layout_validation"
    for policy, layout in (("fixed-bin", None), ("optimized", proposed_layout)):
        candidate_root = validation_root / policy
        hotspot_dir = candidate_root / "hotspot"
        materialize(
            modules_path, hotspot_dir, physical["grid_size"], physical["utilization"],
            frequency["ambient_c"], physical["r_convec_k_per_w"], layout,
            physical.get("thermal_stack"),
        )
        thermal = run_hotspot(hotspot_dir, hotspot_tool)
        performance = evaluate(
            modules_path, hotspot_dir / "thermal_result.json",
            candidate_root / "performance.json", frequency["f0_ghz"],
            frequency["fmin_ghz"], frequency["tsafe_c"], frequency["ambient_c"],
        )
        layout_delays = derive_layout_delays(
            read_json(hotspot_dir / "layout.json"), frequency["f0_ghz"],
            delay.get("wire_rounding", "nearest"),
            communication_weights,
        )
        r2_wire_cycles = select_rounded_wire_cycles(
            layout_delays, aggregation
        )
        evaluation = {
            "policy": policy,
            "layout": str((hotspot_dir / "layout.json").resolve()),
            "hotspot_dir": str(hotspot_dir.resolve()),
            "tmax_c": thermal["tmax_c"],
            "sustainable_frequency_ghz": performance["sustainable_frequency_ghz"],
            "bips1_thermal": performance["bips1_thermal"],
            "wire_cycles_unrounded": layout_delays["wire_cycles_unrounded"],
            "wire_cycles": layout_delays["wire_cycles"],
            "maximum_wire_cycles_unrounded": layout_delays["maximum_wire_cycles_unrounded"],
            "maximum_wire_cycles": layout_delays["maximum_wire_cycles"],
            "r2_wire_aggregation": aggregation,
            "r2_wire_cycles": r2_wire_cycles,
        }
        if aggregation == "traffic-weighted":
            evaluation.update({
                "traffic_weighted_wire_cycles_unrounded": layout_delays[
                    "traffic_weighted_wire_cycles_unrounded"
                ],
                "traffic_weighted_wire_cycles": layout_delays[
                    "traffic_weighted_wire_cycles"
                ],
            })
        evaluations.append(evaluation)

    tolerance = float(optimizer.get("baseline_guard_bips_tolerance", 1e-9))
    selected, reason = select_clip3d_candidate(evaluations, tolerance)
    source = Path(selected["hotspot_dir"])
    destination = output_dir / "hotspot"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    refresh_copied_hotspot_paths(destination)
    selection = {
        "schema_version": 1,
        "method": "clip3d",
        "selection_rule": (
            "maximum real-HotSpot thermal BIPS1; discrete "
            f"{aggregation} wire cycles and real Tmax break ties"
        ),
        "selected_policy": selected["policy"],
        "fallback_used": selected["policy"] == "fixed-bin",
        "selection_reason": reason,
        "bips_tolerance": tolerance,
        "evaluations": evaluations,
        "hotspot_solves": len(evaluations),
        "note": (
            "This is a reproduction safety guard, not a claimed paper algorithm. "
            "It prevents an inaccurate analytic proxy from silently regressing fixed-bin."
        ),
    }
    write_json(output_dir / "layout_selection.json", selection)
    return destination / "layout.json", read_json(destination / "thermal_result.json"), selection


def evaluate_clip3d_paper_single(modules_path: Path, proposed_layout: Path,
                                 output_dir: Path, config: dict,
                                 hotspot_tool: Path) -> tuple[Path, dict, dict]:
    """Materialize exactly the optimizer proposal and run one final HotSpot.

    This is Algorithm 1 lines 11--13.  Unlike the guarded reproduction mode it
    does not query fixed-bin during selection, so fixed and CLIP-3D must be run
    as separate points for an unbiased comparison.
    """
    frequency = config["frequency"]
    physical = config["physical"]
    hotspot_dir = output_dir / "hotspot"
    materialize(
        modules_path, hotspot_dir, physical["grid_size"], physical["utilization"],
        frequency["ambient_c"], physical["r_convec_k_per_w"], proposed_layout,
        physical.get("thermal_stack"),
    )
    thermal = run_hotspot(hotspot_dir, hotspot_tool)
    selection = {
        "schema_version": 1,
        "method": "clip3d",
        "validation_policy": "paper-single",
        "selected_policy": "optimized",
        "fallback_used": False,
        "selection_reason": "Algorithm 1 single final HotSpot validation",
        "evaluations": [{
            "policy": "optimized",
            "layout": str((hotspot_dir / "layout.json").resolve()),
            "hotspot_dir": str(hotspot_dir.resolve()),
            "tmax_c": thermal["tmax_c"],
        }],
        "hotspot_solves": 1,
        "note": "Paper-strict selection; no fixed-bin guard was queried.",
    }
    write_json(output_dir / "layout_selection.json", selection)
    return hotspot_dir / "layout.json", thermal, selection


def run_pipeline(r1_dir: Path, output_dir: Path, config_path: Path,
                 layout_method: str = "fixed-bin", execute_r2: bool = False,
                 rerun_r2: bool = False,
                 reuse_r2_dir: Path | None = None,
                 wire_objective_override: str | None = None,
                 proxy_spatial_model_override: str | None = None) -> dict:
    validate_r1(r1_dir)
    config = read_json(config_path)
    if wire_objective_override is not None:
        config["layout_optimizer"]["wire_objective"] = wire_objective_override
    if proxy_spatial_model_override is not None:
        config["layout_optimizer"]["proxy_spatial_model"] = proxy_spatial_model_override
    validate_config(config, layout_method)
    if execute_r2 and reuse_r2_dir is not None:
        raise ValueError("choose either execute_r2 or reuse_r2_dir, not both")
    frequency = config["frequency"]
    physical = config["physical"]
    mcpat_config = config.get("mcpat", {})
    metadata = read_json(r1_dir / "r1_metadata.json")
    tools = {
        "mcpat": PROJECT_ROOT / "tools/src/mcpat/mcpat",
        "cacti": PROJECT_ROOT / "tools/src/cacti/cacti",
        "cacti_config": PROJECT_ROOT / "tools/src/cacti/cache.cfg",
        "hotspot": PROJECT_ROOT / "tools/src/hotspot/hotspot",
    }
    for name, path in tools.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", {
        "schema_version": 1, "source": str(config_path.resolve()),
        "layout_method": layout_method, "config": config,
    })
    stage_seconds = {}

    started = time.perf_counter()
    mcpat_dir = output_dir / "mcpat"
    mcpat_xml = mcpat_dir / "input.xml"
    convert(r1_dir, mcpat_xml, settings={
        key: mcpat_config[key] for key in (
            "temperature_k", "device_type", "longer_channel_device",
            "interconnect_projection_type",
        ) if key in mcpat_config
    })
    opt_for_clk = int(mcpat_config.get("opt_for_clk", 0))
    command = [str(tools["mcpat"]), "-infile", str(mcpat_xml),
               "-print_level", "5", "-opt_for_clk", str(opt_for_clk)]
    process = subprocess.run(command, cwd=tools["mcpat"].parent, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    mcpat_text = process.stdout
    (mcpat_dir / "mcpat.out").write_text(mcpat_text, encoding="utf-8")
    if process.returncode != 0 or "McPAT (version 1.3" not in mcpat_text or "results" not in mcpat_text:
        raise RuntimeError(f"McPAT failed; see {mcpat_dir / 'mcpat.out'}")
    parsed_mcpat = parse_mcpat_text(mcpat_text)
    parsed_mcpat["command"] = command
    write_json(mcpat_dir / "mcpat.json", parsed_mcpat)
    stage_seconds["mcpat"] = time.perf_counter() - started

    started = time.perf_counter()
    l1_sizes = list(dict.fromkeys((metadata["l1i_size"], metadata["l1d_size"])))
    characterize(
        tools["cacti"], tools["cacti_config"], output_dir / "cacti",
        l1_sizes, [metadata["l2_size"]], frequency["f0_ghz"],
    )
    stage_seconds["cacti"] = time.perf_counter() - started

    started = time.perf_counter()
    modules_path = output_dir / "modules.json"
    reference_raw = float(physical.get("area_reference_raw_mm2", 45.7538495872))
    area_scale = float(physical["area_reference_mm2"]) / reference_raw
    cacti_json = output_dir / "cacti/cacti_characterization.json"
    model = build_model(
        r1_dir, mcpat_dir / "mcpat.json", cacti_json, modules_path, area_scale,
        require_communication_profile=(
            config["delay"].get("wire_aggregation", "mean") == "traffic-weighted"
        ),
    )
    stage_seconds["module_model"] = time.perf_counter() - started

    started = time.perf_counter()
    selection = None
    if layout_method == "clip3d":
        optimizer = config["layout_optimizer"]
        proposed_layout = output_dir / "optimized_layout.json"
        optimize(
            modules_path, proposed_layout, output_dir / "optimizer_report.json",
            physical["utilization"], frequency["ambient_c"],
            physical["r_convec_k_per_w"], optimizer["alpha"], optimizer["beta"],
            optimizer["cross_tier_weight"], frequency["f0_ghz"],
            frequency["fmin_ghz"], frequency["tsafe_c"], optimizer["lambda_wire"],
            optimizer.get("require_scipy", False),
            optimizer.get("allowed_l2_tiers"),
            optimizer.get("proxy_spatial_model", "center"),
            int(optimizer.get("proxy_quadrature_order", 2)),
            optimizer.get("wire_objective", "continuous"),
            config["delay"].get("wire_rounding", "nearest"),
            config["delay"].get("wire_aggregation", "mean"),
        )
        if optimizer.get("validation_policy", "guarded") == "paper-single":
            layout_path, thermal, selection = evaluate_clip3d_paper_single(
                modules_path, proposed_layout, output_dir, config, tools["hotspot"]
            )
        else:
            layout_path, thermal, selection = evaluate_clip3d_candidate(
                modules_path, proposed_layout, output_dir, config, tools["hotspot"]
            )
        hotspot_dir = output_dir / "hotspot"
    elif layout_method in COMPARISON_METHODS:
        layout_path, thermal, selection = evaluate_comparison_candidates(
            modules_path, output_dir, config, layout_method
        )
        hotspot_dir = output_dir / "hotspot"
    else:
        hotspot_dir = output_dir / "hotspot"
        materialize(modules_path, hotspot_dir, physical["grid_size"],
                    physical["utilization"], frequency["ambient_c"],
                    physical["r_convec_k_per_w"], None,
                    physical.get("thermal_stack"))
        layout_path = hotspot_dir / "layout.json"
        thermal = run_hotspot(hotspot_dir, tools["hotspot"])
    stage_seconds["layout_and_hotspot"] = time.perf_counter() - started

    started = time.perf_counter()
    performance_path = output_dir / "performance.json"
    performance = evaluate(
        modules_path, hotspot_dir / "thermal_result.json", performance_path,
        frequency["f0_ghz"], frequency["fmin_ghz"], frequency["tsafe_c"],
        frequency["ambient_c"],
    )
    delay = config["delay"]
    latency_path = output_dir / "r2_latency.json"
    vector = build_vector(
        modules_path, output_dir / "cacti/cacti_characterization.json", latency_path,
        None, None, hotspot_dir / "layout.json", delay.get("wire_rounding", "nearest"),
        int(delay.get("cycles_per_tsv", 2)), int(delay.get("l1_pipeline_cycles", 1)),
        delay.get("wire_aggregation", "mean"),
    )
    stage_seconds["frequency_and_latency"] = time.perf_counter() - started

    r2_result = None
    r2_source = None
    if reuse_r2_dir is not None:
        source_vector = read_json(reuse_r2_dir / "r2_latency.json")
        if source_vector.get("gem5_overrides") != vector.get("gem5_overrides"):
            raise ValueError(
                f"cannot reuse R2 with different latency vector: {reuse_r2_dir}"
            )
        source_result = reuse_r2_dir / "gem5_r2/r2_result.json"
        source_status = reuse_r2_dir / "gem5_r2/status.json"
        if not source_result.is_file() or not source_status.is_file():
            raise FileNotFoundError(f"reusable R2 result/status missing below {reuse_r2_dir}")
        if read_json(source_status).get("state") != "success":
            raise ValueError(f"reusable R2 is not successful: {source_status}")
        r2_result = read_json(source_result)
        r2_source = str(source_result.resolve())
        performance = evaluate(
            modules_path, hotspot_dir / "thermal_result.json", performance_path,
            frequency["f0_ghz"], frequency["fmin_ghz"], frequency["tsafe_c"],
            frequency["ambient_c"], r2_result["ipc2"],
        )
    if execute_r2:
        started = time.perf_counter()
        r2_result = run_r2(
            r1_dir, latency_path, output_dir / "gem5_r2", rerun=rerun_r2
        )
        r2_source = str((output_dir / "gem5_r2/r2_result.json").resolve())
        performance = evaluate(
            modules_path, hotspot_dir / "thermal_result.json", performance_path,
            frequency["f0_ghz"], frequency["fmin_ghz"], frequency["tsafe_c"],
            frequency["ambient_c"], r2_result["ipc2"],
        )
        stage_seconds["gem5_r2"] = time.perf_counter() - started

    final_layout = read_json(hotspot_dir / "layout.json")
    tier_power = {
        str(tier): sum(
            module["total_power_w"] for module in final_layout["modules"]
            if int(module["tier"]) == tier
        )
        for tier in range(int(physical["tiers"]))
    }
    layout_diagnostics = {
        "movable_kinds": model["power_distribution"]["movable_kinds"],
        "movable_power_w": model["power_distribution"]["movable_power_w"],
        "movable_power_fraction": model["power_distribution"]["movable_power_fraction"],
        "power_by_tier_w": tier_power,
        "wire_cycle_aggregation_for_r2": vector["wire_cycle_aggregation_for_r2"],
        "mean_wire_cycles_unrounded": vector["layout_delays"]["wire_cycles_unrounded"],
        "maximum_wire_cycles_unrounded": vector["layout_delays"]["maximum_wire_cycles_unrounded"],
        "mean_wire_cycles_rounded": vector["layout_delays"]["wire_cycles"],
        "maximum_wire_cycles_rounded": vector["layout_delays"]["maximum_wire_cycles"],
    }
    if "traffic_weighted_wire_cycles" in vector["layout_delays"]:
        layout_diagnostics.update({
            "traffic_weighted_wire_cycles_unrounded": vector["layout_delays"][
                "traffic_weighted_wire_cycles_unrounded"
            ],
            "traffic_weighted_wire_cycles_rounded": vector["layout_delays"][
                "traffic_weighted_wire_cycles"
            ],
        })
    if layout_method == "clip3d":
        optimizer_report = read_json(output_dir / "optimizer_report.json")
        selected_proxy = optimizer_report["selected"]["proxy_tmax_c"]
        validated = {item["policy"]: item for item in selection["evaluations"]}
        diagnostics = {
            "optimizer_proxy_tmax_c": selected_proxy,
            "optimized_validated_hotspot_tmax_c": validated["optimized"]["tmax_c"],
            "proxy_minus_optimized_hotspot_c": (
                selected_proxy - validated["optimized"]["tmax_c"]
            ),
            "selected_policy": selection["selected_policy"],
            "baseline_guard_fallback_used": selection["fallback_used"],
            "baseline_guard_selection_reason": selection["selection_reason"],
            "legal_candidate_tiers": sorted({
                candidate["tier"] for candidate in optimizer_report["candidates"]
                if candidate["collision_mm2"] <= 1e-8
            }),
        }
        if "fixed-bin" in validated:
            diagnostics["fixed_bin_validated_hotspot_tmax_c"] = validated["fixed-bin"]["tmax_c"]
        layout_diagnostics.update(diagnostics)

    summary = {
        "schema_version": 2, "r1": str(r1_dir), "output": str(output_dir),
        "experiment": config.get("name", config_path.stem),
        "workload": metadata["workload"], "l1d_size": metadata["l1d_size"],
        "l2_size": metadata["l2_size"], "layout_method": layout_method,
        "layout_mode": layout_method,
        "cooling": {"r_convec_k_per_w": physical["r_convec_k_per_w"],
                    "ambient_c": frequency["ambient_c"]},
        "module_count": len(model["modules"]), "total_power_w": model["totals"]["total_power_w"],
        "power_provenance": model["power_provenance"],
        "area_calibration": model.get("area_calibration"),
        "power_distribution": model.get("power_distribution"),
        "communication_profile": model.get("communication_profile"),
        "gamma": model["gamma"], "tmax_c": thermal["tmax_c"],
        "sustainable_frequency_ghz": performance["sustainable_frequency_ghz"],
        "ipc1": performance["ipc1"], "bips1_thermal": performance["bips1_thermal"],
        "r2_critical_path_cycles": vector["critical_l1d_to_l2_cycles"],
        "layout_delays": vector["layout_delays"],
        "layout_diagnostics": layout_diagnostics,
        "layout_selection": selection,
        "ipc2": r2_result["ipc2"] if r2_result else None,
        "bips2": performance.get("bips2"), "r2_source": r2_source,
        "stage_seconds": stage_seconds,
        "total_pipeline_seconds": sum(stage_seconds.values()),
        "comparison_selection": selection,
        "artifacts": {
            "config": str((output_dir / "run_config.json").resolve()),
            "mcpat_xml": str(mcpat_xml), "mcpat_json": str(mcpat_dir / "mcpat.json"),
            "cacti": str(output_dir / "cacti/cacti_characterization.json"),
            "modules": str(modules_path), "layout": str((hotspot_dir / "layout.json").resolve()),
            "hotspot_manifest": str(hotspot_dir / "hotspot_manifest.json"),
            "thermal": str(hotspot_dir / "thermal_result.json"),
            "performance": str(performance_path), "r2_latency": str(latency_path),
            "r2_result": r2_source,
            "layout_selection": (
                str((output_dir / "layout_selection.json").resolve())
                if selection is not None else None
            ),
        },
    }
    write_json(output_dir / "pipeline_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--layout-method", choices=LAYOUT_METHODS, default="fixed-bin")
    parser.add_argument("--optimized-layout", action="store_true",
                        help="deprecated alias for --layout-method clip3d")
    parser.add_argument("--run-r2", action="store_true",
                        help="launch the long second gem5 run after generating its vector")
    parser.add_argument("--rerun-r2", action="store_true")
    parser.add_argument("--reuse-r2-dir", type=Path,
                        help="reuse a successful R2 when the generated latency vector is identical")
    parser.add_argument(
        "--wire-objective", choices=("continuous", "r2-quantized"),
        help="override the layout objective without editing the source config",
    )
    parser.add_argument(
        "--proxy-spatial-model", choices=("center", "area-quadrature"),
        help="override the thermal proxy geometry without editing the source config",
    )
    parser.add_argument(
        "--transient", type=boolean_text, default=False, metavar="{true,false}",
        help=("run the separate time-windowed McPAT/HotSpot branch after the "
              "unchanged steady pipeline; default: false"),
    )
    parser.add_argument("--transient-sample-ms", type=positive_float, default=10.0)
    parser.add_argument(
        "--transient-r1-dir", type=Path,
        help=("reuse an existing periodic-statistics R1; when omitted, the transient "
              "branch creates output-dir/transient/r1 without touching the source R1"),
    )
    parser.add_argument(
        "--transient-initial-temperature", choices=("steady", "ambient"),
        default="steady",
    )
    parser.add_argument("--rerun-transient-r1", action="store_true")
    args = parser.parse_args()
    if args.optimized_layout:
        if args.layout_method != "fixed-bin":
            parser.error("do not combine --optimized-layout and --layout-method")
        args.layout_method = "clip3d"
    summary = run_pipeline(
        args.r1_dir.resolve(), args.output_dir.resolve(), args.config.resolve(),
        args.layout_method, args.run_r2, args.rerun_r2,
        args.reuse_r2_dir.resolve() if args.reuse_r2_dir else None,
        args.wire_objective, args.proxy_spatial_model,
    )
    if args.transient:
        from workflow.transient.run_transient_pipeline import run_transient_pipeline

        transient = run_transient_pipeline(
            args.r1_dir.resolve(), args.output_dir.resolve(),
            (args.output_dir / "transient").resolve(), args.config.resolve(),
            args.transient_sample_ms,
            args.transient_r1_dir.resolve() if args.transient_r1_dir else None,
            args.transient_initial_temperature,
            args.rerun_transient_r1,
        )
        summary["transient"] = {
            "enabled": True,
            "summary": str(
                (args.output_dir / "transient/transient_pipeline_summary.json").resolve()
            ),
            "sample_interval_ms": transient["sample_interval_ms"],
            "window_count": transient["window_count"],
            "tmax_c": transient["transient_tmax_c"],
        }
        write_json(args.output_dir / "pipeline_summary.json", summary)
    print(f"Pipeline complete: method={summary['layout_method']}, "
          f"Tmax={format_temperature_c(summary['tmax_c'])} C, "
          f"f_sus={summary['sustainable_frequency_ghz']:.6f} GHz")
    if args.transient:
        print(
            f"Transient thermal complete: windows={transient['window_count']}, "
            f"Tmax={format_temperature_c(transient['transient_tmax_c'])} C"
        )


if __name__ == "__main__":
    main()
