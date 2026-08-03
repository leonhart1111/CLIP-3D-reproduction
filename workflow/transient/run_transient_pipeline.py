#!/usr/bin/env python3
"""Run the optional gem5-window/McPAT/HotSpot transient branch for one point."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

from workflow.common import PROJECT_ROOT, read_json, write_json
from workflow.floorplan.generate_hotspot_inputs import DEFAULT_THERMAL_STACK
from workflow.transient.generate_hotspot_trace import materialize_trace
from workflow.transient.run_hotspot_transient import run_hotspot_transient
from workflow.transient.run_transient_r1 import run as run_transient_r1
from workflow.transient.run_windowed_mcpat import run_windows
from workflow.transient.stats_windows import split_windows
from workflow.transient.validation import (
    power_trace_identity,
    validate_power_windows,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/clip3d_pipeline.json"
RAW_POWER_PROVENANCE = {
    "dynamic": "McPAT Runtime Dynamic",
    "leakage": "McPAT Subthreshold Leakage + Gate Leakage",
    "postprocessing": "none",
}
WINDOW_RAW_POWER_PROVENANCE = {
    "dynamic": "McPAT Runtime Dynamic",
    "subthreshold_leakage": "McPAT Subthreshold Leakage",
    "gate_leakage": "McPAT Gate Leakage",
    "postprocessing": "none",
}


def validate_matching_r1(source: dict, transient: dict) -> None:
    mismatches = [
        key for key, value in source.items() if transient.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "transient R1 does not match the source steady R1 metadata: "
            + ", ".join(mismatches)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def reject_overlapping_output(output_dir: Path, read_only_inputs: list[Path]) -> None:
    """Reject equal, ancestor, or descendant output paths before any write."""
    output_dir = output_dir.resolve()
    for source in read_only_inputs:
        source = source.resolve()
        if (
            output_dir == source
            or output_dir in source.parents
            or source in output_dir.parents
        ):
            raise ValueError(
                f"output path {output_dir} overlaps read-only input {source}"
            )


def _require_path(value: object, expected: Path, label: str) -> None:
    if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
        raise ValueError(f"{label} does not identify expected path {expected.resolve()}")


def _module_identity(module: dict) -> dict:
    return {
        field: module.get(field)
        for field in (
            "name", "kind", "core", "area_mm2", "dynamic_power_w",
            "leakage_power_w", "total_power_w",
        )
    }


def _validate_raw_power_calibration(value: object, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{context}: raw-power calibration must be null or an object")
    for field in ("dynamic_scale", "leakage_scale"):
        if field not in value or float(value[field]) != 1.0:
            raise ValueError(f"{context}: raw-power scale {field} must equal 1.0")


def validate_source_r1(source_r1_dir: Path) -> dict:
    source_r1_dir = source_r1_dir.resolve()
    for name in ("r1_metadata.json", "status.json", "stats.txt"):
        path = source_r1_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"canonical source R1 artifact is missing: {path}")
    status = read_json(source_r1_dir / "status.json")
    if status.get("state") != "success":
        raise ValueError("canonical source R1 is not successful")
    metadata_path = source_r1_dir / "r1_metadata.json"
    return {
        "path": str(source_r1_dir),
        "metadata": read_json(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
    }


def validate_steady_output(steady_output_dir: Path,
                           expected_layout: str | None = None,
                           source_r1_dir: Path | None = None,
                           config: dict | None = None,
                           config_path: Path | None = None) -> dict:
    """Preflight one completed steady pipeline before expensive transient work."""
    steady_output_dir = steady_output_dir.resolve()
    required = (
        steady_output_dir / "modules.json",
        steady_output_dir / "hotspot/layout.json",
        steady_output_dir / "hotspot/steady.txt",
        steady_output_dir / "hotspot/hotspot_manifest.json",
        steady_output_dir / "pipeline_summary.json",
        steady_output_dir / "run_config.json",
        steady_output_dir / "mcpat/mcpat.json",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"steady pipeline artifact required by transient branch: {path}"
            )
    summary = read_json(steady_output_dir / "pipeline_summary.json")
    if expected_layout is not None and summary.get("layout_method") != expected_layout:
        raise ValueError(
            f"steady pipeline layout must be {expected_layout}, "
            f"got {summary.get('layout_method')}"
        )
    if source_r1_dir is None or config is None:
        return summary

    mcpat_config = config.get("mcpat", {})
    if isinstance(mcpat_config, dict) and "power_calibration" in mcpat_config:
        raise ValueError(
            "selected config must not declare mcpat.power_calibration for transient validation"
        )

    source = validate_source_r1(source_r1_dir)
    source_r1_dir = source_r1_dir.resolve()
    metadata = source["metadata"]
    _require_path(summary.get("r1"), source_r1_dir, "steady pilot source R1")
    _require_path(summary.get("output"), steady_output_dir, "steady pilot output")
    for field in ("workload", "l1d_size", "l2_size"):
        if summary.get(field) != metadata.get(field):
            raise ValueError(f"steady pilot {field} does not match canonical source R1")
    if summary.get("experiment") != config.get("name"):
        raise ValueError("steady pilot experiment does not match selected config")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("steady pilot artifact provenance is missing")
    expected_artifacts = {
        "config": steady_output_dir / "run_config.json",
        "modules": steady_output_dir / "modules.json",
        "layout": steady_output_dir / "hotspot/layout.json",
        "hotspot_manifest": steady_output_dir / "hotspot/hotspot_manifest.json",
        "mcpat_json": steady_output_dir / "mcpat/mcpat.json",
    }
    for field, expected in expected_artifacts.items():
        _require_path(
            artifacts.get(field), expected, f"steady pilot artifact {field}"
        )

    run_config = read_json(steady_output_dir / "run_config.json")
    if run_config.get("config") != config:
        raise ValueError("steady pilot embedded config does not match selected config")
    if run_config.get("layout_method") != summary.get("layout_method"):
        raise ValueError("steady pilot run_config layout does not match summary")
    if config_path is not None:
        recorded_config_source = run_config.get("source")
        if not isinstance(recorded_config_source, str):
            raise ValueError("steady pilot config source path is missing")
        recorded_config_path = Path(recorded_config_source).resolve()
        if not recorded_config_path.is_file() or read_json(recorded_config_path) != config:
            raise ValueError(
                "steady pilot config source is not scientifically identical "
                "to the selected config"
            )

    expected_cooling = {
        "r_convec_k_per_w": float(config["physical"]["r_convec_k_per_w"]),
        "ambient_c": float(config["frequency"]["ambient_c"]),
    }
    if summary.get("cooling") != expected_cooling:
        raise ValueError("steady pilot cooling does not match selected config")

    modules = read_json(steady_output_dir / "modules.json")
    _require_path(modules.get("source_r1"), source_r1_dir, "module source R1")
    if modules.get("architecture") != metadata:
        raise ValueError("module architecture does not match canonical source metadata")
    if modules.get("power_provenance") != RAW_POWER_PROVENANCE:
        raise ValueError("module raw-power provenance is incompatible")
    _validate_raw_power_calibration(
        modules.get("power_calibration"), "module model"
    )
    module_records = modules.get("modules")
    if not isinstance(module_records, list) or not module_records:
        raise ValueError("module model must contain modules")
    module_by_name = {}
    for module in module_records:
        name = module.get("name")
        if not isinstance(name, str) or name in module_by_name:
            raise ValueError("module identity contains missing or duplicate names")
        for field in ("dynamic_power_w", "leakage_power_w", "total_power_w"):
            value = float(module[field])
            if not math.isfinite(value) or value < -1e-12:
                raise ValueError(f"steady module {name} {field} is invalid")
        module_by_name[name] = module

    layout = read_json(steady_output_dir / "hotspot/layout.json")
    layout_records = layout.get("modules")
    if not isinstance(layout_records, list):
        raise ValueError("steady layout modules are missing")
    layout_by_name = {module.get("name"): module for module in layout_records}
    if None in layout_by_name or len(layout_by_name) != len(layout_records):
        raise ValueError("steady layout contains missing or duplicate module names")
    if set(layout_by_name) != set(module_by_name):
        raise ValueError("steady layout module identity does not match module model")
    for name, module in module_by_name.items():
        if _module_identity(layout_by_name[name]) != _module_identity(module):
            raise ValueError(f"steady layout module identity differs for {name}")
    utilization = float(layout.get("utilization_target", -1))
    if not math.isclose(
        utilization, float(config["physical"]["utilization"]),
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ValueError("steady layout utilization does not match selected config")
    tiers = {int(module["tier"]) for module in layout_records}
    if tiers != set(range(int(config["physical"]["tiers"]))):
        raise ValueError("steady layout tier identity does not match selected config")

    hotspot_manifest = read_json(steady_output_dir / "hotspot/hotspot_manifest.json")
    if int(hotspot_manifest.get("grid_size", -1)) != int(
        config["physical"]["grid_size"]
    ):
        raise ValueError("steady pilot grid is incompatible with selected config")
    for field, expected in expected_cooling.items():
        manifest_field = "r_convec_k_per_w" if field.startswith("r_convec") else field
        if not math.isclose(
            float(hotspot_manifest.get(manifest_field, float("nan"))), expected,
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(f"steady pilot {field} is incompatible")
    expected_stack = {
        **DEFAULT_THERMAL_STACK,
        **config["physical"].get("thermal_stack", {}),
    }
    recorded_stack = hotspot_manifest.get("thermal_stack")
    if not isinstance(recorded_stack, dict):
        raise ValueError("steady pilot thermal stack is missing")
    for field, expected in expected_stack.items():
        if field not in recorded_stack or recorded_stack[field] != expected:
            raise ValueError(f"steady pilot thermal-stack field {field} is incompatible")

    mcpat = read_json(steady_output_dir / "mcpat/mcpat.json")
    if mcpat.get("power_provenance") != RAW_POWER_PROVENANCE:
        raise ValueError("McPAT raw-power provenance is incompatible")
    _validate_raw_power_calibration(mcpat.get("power_calibration"), "steady McPAT")
    if summary.get("power_provenance") != RAW_POWER_PROVENANCE:
        raise ValueError("steady summary raw-power provenance is incompatible")
    _validate_raw_power_calibration(
        summary.get("power_calibration"), "steady summary"
    )
    module_total = sum(float(module["total_power_w"]) for module in module_records)
    if not math.isclose(
        float(summary.get("total_power_w", float("nan"))), module_total,
        rel_tol=1e-9, abs_tol=1e-9,
    ):
        raise ValueError("steady summary power does not match module aggregate")
    if int(summary.get("module_count", -1)) != len(module_records):
        raise ValueError("steady summary module count does not match module model")

    return {
        "summary": summary,
        "canonical_source_r1": str(source_r1_dir),
        "canonical_metadata_sha256": source["metadata_sha256"],
        "experiment": summary["experiment"],
        "workload": summary["workload"],
        "l1d_size": summary["l1d_size"],
        "l2_size": summary["l2_size"],
        "layout_method": summary["layout_method"],
        "cooling": expected_cooling,
        "grid_size": int(hotspot_manifest["grid_size"]),
        "thermal_stack": expected_stack,
        "module_identity": [
            _module_identity(module) for module in module_records
        ],
        "raw_power": {
            "provenance": RAW_POWER_PROVENANCE,
        },
    }


def validate_dual_steady_inputs(source_r1_dir: Path, fixed_steady_dir: Path,
                                clip3d_steady_dir: Path, config: dict,
                                config_path: Path) -> dict:
    """Validate both steady pilots against canonical inputs and each other."""
    fixed = validate_steady_output(
        fixed_steady_dir, "fixed-bin", source_r1_dir, config, config_path
    )
    clip3d = validate_steady_output(
        clip3d_steady_dir, "clip3d", source_r1_dir, config, config_path
    )
    shared_fields = (
        "canonical_source_r1", "canonical_metadata_sha256", "experiment",
        "workload", "l1d_size", "l2_size", "cooling", "grid_size",
        "thermal_stack", "module_identity", "raw_power",
    )
    mismatches = [field for field in shared_fields if fixed[field] != clip3d[field]]
    if mismatches:
        raise ValueError(
            "steady pilots are mutually incompatible: " + ", ".join(mismatches)
        )
    return {
        "canonical_source_r1": fixed["canonical_source_r1"],
        "canonical_metadata_sha256": fixed["canonical_metadata_sha256"],
        "experiment": fixed["experiment"],
        "workload": fixed["workload"],
        "l1d_size": fixed["l1d_size"],
        "l2_size": fixed["l2_size"],
        "cooling": fixed["cooling"],
        "grid_size": fixed["grid_size"],
        "thermal_stack": fixed["thermal_stack"],
        "module_identity": fixed["module_identity"],
        "raw_power": fixed["raw_power"],
        "layouts": {
            "fixed-bin": fixed["summary"],
            "clip3d": clip3d["summary"],
        },
    }


def prepare_power_windows(source_r1_dir: Path, transient_r1_dir: Path,
                          output_dir: Path, config: dict,
                          sample_ms: float) -> dict:
    """Split one periodic R1 and run McPAT once for all thermal layouts."""
    source_r1_dir = source_r1_dir.resolve()
    transient_r1_dir = transient_r1_dir.resolve()
    output_dir = output_dir.resolve()
    if not math.isfinite(sample_ms) or sample_ms <= 0:
        raise ValueError("sample_ms must be positive")

    source_metadata = read_json(source_r1_dir / "r1_metadata.json")
    transient_metadata = read_json(transient_r1_dir / "r1_metadata.json")
    validate_matching_r1(source_metadata, transient_metadata)
    recorded_ms = float(transient_metadata.get("sample_interval_ms", -1))
    if abs(recorded_ms - sample_ms) > 1e-12:
        raise ValueError(
            f"transient R1 sampling mismatch: requested {sample_ms} ms, "
            f"recorded {recorded_ms} ms"
        )

    stage_seconds = {}
    gem5_windows_dir = output_dir / "gem5"
    started = time.perf_counter()
    print("Transient branch: splitting cumulative gem5 statistics", flush=True)
    windows = split_windows(transient_r1_dir, gem5_windows_dir)
    stage_seconds["split_stats"] = time.perf_counter() - started

    mcpat_dir = output_dir / "mcpat"
    started = time.perf_counter()
    print(
        f"Transient branch: running McPAT for {windows['window_count']} windows",
        flush=True,
    )
    power_windows = run_windows(
        gem5_windows_dir / "windows_manifest.json", mcpat_dir, config
    )
    stage_seconds["windowed_mcpat"] = time.perf_counter() - started
    if int(power_windows["window_count"]) != int(windows["window_count"]):
        raise ValueError("McPAT power window count does not match gem5 windows")
    timeline_audit = validate_power_windows(power_windows)
    _require_path(
        power_windows.get("canonical_source_r1"), source_r1_dir,
        "power-window canonical source R1",
    )
    _require_path(
        power_windows.get("transient_r1"), transient_r1_dir,
        "power-window transient R1",
    )
    run_settings = power_windows.get("run_settings")
    if not isinstance(run_settings, dict):
        raise ValueError("power windows lack McPAT run settings")
    if power_windows.get("power_provenance") != WINDOW_RAW_POWER_PROVENANCE:
        raise ValueError("power windows raw-power provenance is incompatible")
    actual_duration_s = float(timeline_audit["total_duration_s"])
    power_windows_path = (mcpat_dir / "power_windows.json").resolve()
    return {
        "schema_version": 1,
        "mode": "operational transient validation",
        "non_formal": True,
        "paper_equivalent": False,
        "source_r1": str(Path(power_windows["canonical_source_r1"]).resolve()),
        "transient_r1": str(Path(power_windows["transient_r1"]).resolve()),
        "sample_interval_ms": sample_ms,
        "window_count": int(windows["window_count"]),
        "actual_gem5_duration_s": actual_duration_s,
        "hotspot_trace_duration_s": timeline_audit["hotspot_trace_duration_s"],
        "padded_final_duration_s": timeline_audit["padded_final_duration_s"],
        "windows_manifest": str(
            (gem5_windows_dir / "windows_manifest.json").resolve()
        ),
        "power_windows": str(power_windows_path),
        "power_trace_identity": power_trace_identity(power_windows),
        "timeline_audit": timeline_audit,
        "raw_power_evidence": {
            "power_provenance": WINDOW_RAW_POWER_PROVENANCE,
        },
        "acceptance_checks": {
            "checks": {
                "matching_canonical_and_transient_r1": True,
                "at_least_two_windows": timeline_audit["window_count"] >= 2,
                "fixed_step_timeline": True,
                "actual_duration_within_hotspot_duration": (
                    timeline_audit["total_duration_s"]
                    <= timeline_audit["hotspot_trace_duration_s"]
                ),
                "raw_power_unscaled": True,
                "module_power_conservation": True,
            },
            "all_passed": True,
            "failure_reasons": [],
        },
        "stage_seconds": stage_seconds,
    }


def run_layout_thermal(source_r1_dir: Path, steady_output_dir: Path,
                       output_dir: Path, config: dict,
                       power_windows_path: Path,
                       initial_temperature: str = "steady") -> dict:
    """Map shared McPAT windows onto one layout and run transient HotSpot."""
    source_r1_dir = source_r1_dir.resolve()
    steady_output_dir = steady_output_dir.resolve()
    output_dir = output_dir.resolve()
    power_windows_path = power_windows_path.resolve()
    reject_overlapping_output(output_dir, [source_r1_dir, steady_output_dir])
    steady_audit = validate_steady_output(
        steady_output_dir, source_r1_dir=source_r1_dir, config=config
    )
    steady_summary = steady_audit["summary"]
    power_windows = read_json(power_windows_path)
    _require_path(
        power_windows.get("canonical_source_r1"),
        Path(steady_audit["canonical_source_r1"]),
        "power-window canonical source R1",
    )
    transient_r1_value = power_windows.get("transient_r1")
    if not isinstance(transient_r1_value, str):
        raise ValueError("power-window transient R1 provenance is missing")
    if initial_temperature not in ("steady", "ambient"):
        raise ValueError("initial_temperature must be steady or ambient")

    stage_seconds = {}
    hotspot_dir = output_dir / "hotspot"
    started = time.perf_counter()
    print("Transient branch: mapping module powers to the selected 3-D layout", flush=True)
    trace = materialize_trace(
        steady_output_dir / "modules.json",
        steady_output_dir / "hotspot/layout.json",
        power_windows_path,
        hotspot_dir,
        config,
    )
    stage_seconds["power_trace_mapping"] = time.perf_counter() - started

    started = time.perf_counter()
    print("Transient branch: running one multi-row detailed-3D HotSpot solve", flush=True)
    thermal = run_hotspot_transient(
        hotspot_dir,
        initial_temperature=initial_temperature,
        steady_source=steady_output_dir / "hotspot/steady.txt",
    )
    stage_seconds["hotspot_transient"] = time.perf_counter() - started

    return {
        "schema_version": 1,
        "mode": "operational transient validation",
        "non_formal": True,
        "paper_equivalent": False,
        "source_r1": steady_audit["canonical_source_r1"],
        "transient_r1": str(Path(transient_r1_value).resolve()),
        "steady_output": str(steady_output_dir),
        "output": str(output_dir),
        "workload": steady_summary["workload"],
        "layout_method": steady_summary["layout_method"],
        "sample_interval_ms": float(trace["sample_interval_s"]) * 1000.0,
        "window_count": int(trace["window_count"]),
        "actual_gem5_duration_s": float(trace["actual_gem5_duration_s"]),
        "hotspot_trace_duration_s": float(trace["hotspot_trace_duration_s"]),
        "padded_final_duration_s": float(trace["padded_final_duration_s"]),
        "initial_temperature": initial_temperature,
        "steady_tmax_c": float(steady_summary["tmax_c"]),
        "transient_tmax_c": float(thermal["tmax_c"]),
        "transient_minus_steady_c": (
            float(thermal["tmax_c"]) - float(steady_summary["tmax_c"])
        ),
        "trace_peak_minus_steady_c": (
            float(thermal["trace_peak"]["tmax_c"])
            - float(steady_summary["tmax_c"])
        ),
        "initial_peak": thermal["initial_peak"],
        "trace_min_peak": thermal["trace_min_peak"],
        "trace_peak": thermal["trace_peak"],
        "final_peak": thermal["final_peak"],
        "overall_peak": thermal["overall_peak"],
        "temperature": thermal,
        "power_summary": trace["power_summary"],
        "power_trace_identity": power_trace_identity(read_json(power_windows_path)),
        "provenance_audit": steady_audit,
        "raw_power_evidence": trace["raw_power_evidence"],
        "conservation_evidence": trace["conservation_evidence"],
        "acceptance_checks": {
            "checks": {
                "canonical_source_provenance": True,
                "steady_pilot_provenance": True,
                "semantic_power_identity": True,
                "fixed_step_timeline": trace["acceptance_checks"]["checks"][
                    "fixed_step_timeline"
                ],
                "raw_power_unscaled": trace["acceptance_checks"]["checks"][
                    "raw_unscaled_power"
                ],
                "module_power_conservation": trace["acceptance_checks"]["checks"][
                    "module_power_conservation"
                ],
                "grid_power_conservation": trace["acceptance_checks"]["checks"][
                    "grid_power_conservation"
                ],
                "hotspot_return_code_zero": thermal["acceptance_checks"]["checks"][
                    "hotspot_return_code_zero"
                ],
                "temperature_sample_count_matches_windows": thermal[
                    "acceptance_checks"
                ]["checks"]["temperature_sample_count_matches_windows"],
            },
            "all_passed": True,
            "failure_reasons": [],
        },
        "limitations": [
            "McPAT leakage uses a fixed configured temperature.",
            "There is no temperature-leakage-DVFS feedback loop.",
            "10 ms averaging cannot observe sub-window microsecond power peaks.",
            "The final partial gem5 window is padded to one HotSpot interval.",
            "Steady initialization omits the program's incomplete startup history.",
        ],
        "stage_seconds": stage_seconds,
        "artifacts": {
            "power_windows": str(power_windows_path),
            "power_trace": str((hotspot_dir / "power_transient.ptrace").resolve()),
            "temperature_trace": str((hotspot_dir / "transient.ttrace").resolve()),
            "thermal_result": str((hotspot_dir / "transient_result.json").resolve()),
            "temperature_summary_csv": str(
                (hotspot_dir / "transient_summary.csv").resolve()
            ),
        },
    }


def run_transient_pipeline(source_r1_dir: Path, steady_output_dir: Path,
                           output_dir: Path, config_path: Path,
                           sample_ms: float = 10.0,
                           transient_r1_dir: Path | None = None,
                           initial_temperature: str = "steady",
                           rerun_transient_r1: bool = False) -> dict:
    source_r1_dir = source_r1_dir.resolve()
    steady_output_dir = steady_output_dir.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    if not math.isfinite(sample_ms) or sample_ms <= 0:
        raise ValueError("sample_ms must be positive")
    config = read_json(config_path)
    reject_overlapping_output(output_dir, [source_r1_dir, steady_output_dir])
    validate_steady_output(
        steady_output_dir, source_r1_dir=source_r1_dir, config=config,
        config_path=config_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_seconds = {}

    if transient_r1_dir is None:
        transient_r1_dir = output_dir / "r1"
        started = time.perf_counter()
        print(
            f"Transient branch: launching/reusing periodic-statistics R1 in "
            f"{transient_r1_dir}",
            flush=True,
        )
        r1_status = run_transient_r1(
            source_r1_dir, transient_r1_dir, sample_ms,
            rerun=rerun_transient_r1,
        )
        stage_seconds["transient_r1"] = time.perf_counter() - started
        transient_r1_source = "generated"
    else:
        transient_r1_dir = transient_r1_dir.resolve()
        r1_status = (
            read_json(transient_r1_dir / "status.json")
            if (transient_r1_dir / "status.json").is_file()
            else None
        )
        transient_r1_source = "provided"
    prepared = prepare_power_windows(
        source_r1_dir, transient_r1_dir, output_dir / "windows", config, sample_ms
    )
    stage_seconds.update(prepared["stage_seconds"])
    result = run_layout_thermal(
        source_r1_dir,
        steady_output_dir,
        output_dir,
        config,
        Path(prepared["power_windows"]),
        initial_temperature,
    )
    stage_seconds.update(result["stage_seconds"])
    result.update({
        "transient_r1_source": transient_r1_source,
        "transient_r1_status": r1_status,
        "config": str(config_path),
        "stage_seconds": stage_seconds,
        "total_pipeline_seconds": sum(stage_seconds.values()),
        "limitations": [
            "McPAT leakage is evaluated at its configured fixed operating temperature.",
            "There is no temperature-leakage-DVFS feedback loop in this validation branch.",
            "A final partial gem5 window is padded to one full HotSpot interval.",
            "10 ms averaging cannot observe sub-window microsecond power peaks.",
            "Steady initialization omits the program's incomplete startup history.",
        ],
    })
    result["artifacts"]["windows_manifest"] = prepared["windows_manifest"]
    write_json(output_dir / "transient_pipeline_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-r1-dir", type=Path, required=True)
    parser.add_argument("--steady-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-ms", type=float, default=10.0)
    parser.add_argument("--transient-r1-dir", type=Path)
    parser.add_argument("--initial-temperature", choices=("steady", "ambient"),
                        default="steady")
    parser.add_argument("--rerun-transient-r1", action="store_true")
    args = parser.parse_args()
    result = run_transient_pipeline(
        args.source_r1_dir, args.steady_output_dir, args.output_dir,
        args.config, args.sample_ms, args.transient_r1_dir,
        args.initial_temperature, args.rerun_transient_r1,
    )
    print(
        f"Transient pipeline complete: windows={result['window_count']}, "
        f"Tmax={result['transient_tmax_c']:.3f} C"
    )


if __name__ == "__main__":
    main()
