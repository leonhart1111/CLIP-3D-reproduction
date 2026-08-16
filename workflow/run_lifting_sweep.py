#!/usr/bin/env python3
"""Resume and parallelize one CLIP-3D lifting experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
from copy import deepcopy
from pathlib import Path

from workflow.cache_contract import (
    mcpat_embedded_cache_record_identity,
    validate_cache_contract,
)
from workflow.common import (
    PROJECT_ROOT,
    parse_frequency_ghz,
    parse_size_bytes,
    read_json,
    sha256_file,
    write_json,
)
from workflow.floorplan.build_module_model import (
    construct_model,
    validate_mobility_contract,
)
from workflow.mcpat.parse_mcpat import parse_mcpat_text
from workflow.mcpat.run_mcpat import (
    BUILD_PROVENANCE,
    MCPAT_PROVENANCE_AUTHORITY,
    MCPAT_PROVENANCE_SCHEMA_VERSION,
    PATCH_FILE,
)
from workflow.r1_catalog import build_catalogue, canonical_directories
from workflow.run_lifting_pipeline import LAYOUT_METHODS, run_pipeline


DEFAULT_R1_ROOT = PROJECT_ROOT / "runs/architecture_sweep/r1/paper"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs/architecture_sweep/lifting"
DEFAULT_R1_EXPERIMENT = PROJECT_ROOT / "configs/experiments/r1_cache_sweep.json"
DEFAULT_SWEEP_CONFIG = PROJECT_ROOT / (
    "configs/experiments/"
    "clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json"
)


required_artifacts = (
    "run_config.json",
    "mcpat/mcpat.json",
    "modules.json",
    "hotspot/layout.json",
    "hotspot/thermal_result.json",
    "performance.json",
    "r2_latency.json",
    "pipeline_summary.json",
)
CLIP3D_REQUIRED_ARTIFACTS = ("optimizer_report.json", "layout_selection.json")
_CACHE_AUTHORITY = "McPAT 1.3 embedded CACTI-P"
_MCPAT_HASH_FIELDS = {
    "xml_sha256", "mapping_sha256", "output_sha256", "binary_sha256",
    "patch_sha256",
}
_AREA_PROVENANCE = {
    "core_logic_and_interconnect": "unmodified McPAT area",
    "l1i_l1d_l2": "McPAT aggregate area with embedded CACTI-P aspect ratio",
    "global_scaling": "none",
}
_PARSED_MCPAT_FIELDS = {
    "schema_version", "technology_nm", "clock_mhz", "power_provenance",
    "processor", "modules", "module_totals", "core_parent_metrics", "checks",
    "embedded_cacti_p",
}
_RUNNER_MCPAT_FIELDS = _PARSED_MCPAT_FIELDS | {
    "command", "cache_contract", "optimization_constraints", "provenance",
}


def _nonzero_sha256(value: object) -> bool:
    try:
        return (
            isinstance(value, str) and len(value) == 64
            and int(value, 16) >= 0 and set(value) != {"0"}
        )
    except ValueError:
        return False


def _validate_cache_architecture(contract: dict, architecture: dict,
                                 mcpat: dict) -> None:
    """Bind the input mapping to the live R1 architecture it describes."""
    records = {record["level"]: record for record in contract["records"]}
    for level, size_key, associativity_key in (
            ("l1i", "l1i_size", "l1_associativity"),
            ("l1d", "l1d_size", "l1_associativity"),
            ("l2", "l2_size", "l2_associativity")):
        record = records[level]
        try:
            agrees = (
                record["size_bytes"] == parse_size_bytes(architecture[size_key])
                and record["associativity"] == int(architecture[associativity_key])
                and record["line_size_bytes"] == int(
                    architecture.get("cache_line_bytes", 64)
                )
                and record["core_count"] == int(architecture["num_cores"])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("live R1 cache architecture is incomplete") from error
        if not agrees:
            raise ValueError(f"McPAT cache contract differs from live R1 {level}")
    clock = architecture.get("cpu_clock", architecture.get("clock"))
    try:
        clock_mhz = parse_frequency_ghz(clock) * 1000.0
    except (TypeError, ValueError) as error:
        raise ValueError("live R1 clock identity is invalid") from error
    if clock_mhz != float(mcpat.get("clock_mhz")) \
            or any(float(record["technology_nm"]) != float(mcpat.get("technology_nm"))
                   for record in records.values()):
        raise ValueError("McPAT clock/technology differs from live R1 mapping")


def _validate_native_mcpat(mcpat: dict, output: Path) -> dict:
    if set(mcpat) != _RUNNER_MCPAT_FIELDS or mcpat.get("schema_version") != 1:
        raise ValueError("McPAT artifact top-level runner schema is invalid")
    embedded = mcpat.get("embedded_cacti_p")
    if not isinstance(embedded, dict) or set(embedded) != {
            "schema_version", "authority", "records"}:
        raise ValueError("McPAT artifact lacks the embedded_cacti_p schema")
    if embedded.get("schema_version") != 1 \
            or embedded.get("authority") != _CACHE_AUTHORITY:
        raise ValueError("McPAT embedded_cacti_p authority is invalid")
    records = embedded.get("records")
    if not isinstance(records, list):
        raise ValueError("McPAT embedded_cacti_p records must be an array")
    expected = {
        *(("l1i", core) for core in range(4)),
        *(("l1d", core) for core in range(4)),
        ("l2", None),
    }
    observed = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("McPAT embedded_cacti_p record must be an object")
        identity = (record.get("cache"), record.get("core"))
        observed.append(identity)
        if record.get("record_id") != mcpat_embedded_cache_record_identity(record):
            raise ValueError("McPAT embedded_cacti_p record identity is invalid")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("McPAT embedded_cacti_p record set is incomplete")

    checks = mcpat.get("checks")
    if not isinstance(checks, dict) or checks.get("core_count") != 4 \
            or checks.get("core_logic_granularity") != (
                "McPAT top-level functional blocks"):
        raise ValueError("McPAT artifact lacks strict granular parser checks")
    validate_cache_contract(mcpat.get("cache_contract"), expected_core_count=4)

    provenance = mcpat.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
            "schema_version", "authority", "hashes"}:
        raise ValueError("McPAT artifact lacks strict provenance")
    if provenance.get("schema_version") != MCPAT_PROVENANCE_SCHEMA_VERSION \
            or provenance.get("authority") != MCPAT_PROVENANCE_AUTHORITY:
        raise ValueError("McPAT provenance authority is invalid")
    hashes = provenance.get("hashes")
    allowed_hashes = (_MCPAT_HASH_FIELDS, _MCPAT_HASH_FIELDS | {
        "build_provenance_sha256",
    })
    if not isinstance(hashes, dict) or set(hashes) not in allowed_hashes \
            or not all(_nonzero_sha256(value) for value in hashes.values()):
        raise ValueError("McPAT provenance hash schema is invalid")

    mcpat_output = output / "mcpat/mcpat.out"
    command = mcpat.get("command")
    if not isinstance(command, list) or len(command) != 7 \
            or not all(isinstance(value, str) for value in command) \
            or command[1] != "-infile" or command[3:6] != [
                "-print_level", "5", "-opt_for_clk",
            ] or command[6] not in {"0", "1"}:
        raise ValueError("McPAT artifact lacks the strict executed command")
    binary = Path(command[0])
    if not binary.is_absolute():
        raise ValueError("McPAT executed binary path must be absolute")
    xml_path = output / "mcpat/input.xml"
    if not Path(command[2]).is_absolute() \
            or Path(command[2]).resolve() != xml_path.resolve():
        raise ValueError("McPAT command input does not match live XML")
    if sha256_file(mcpat_output) != hashes["output_sha256"]:
        raise ValueError("McPAT output hash does not match live output")
    live_mcpat = parse_mcpat_text(
        mcpat_output.read_text(encoding="utf-8"),
        expected_core_count=4,
        require_granular_cores=True,
        require_embedded_cacti=True,
    )
    if any(mcpat[field] != live_mcpat[field]
           for field in _PARSED_MCPAT_FIELDS):
        raise ValueError("McPAT parsed scientific payload does not match live output")
    if sha256_file(binary) != hashes["binary_sha256"]:
        raise ValueError("McPAT binary hash does not match live executable")
    if sha256_file(xml_path) != hashes["xml_sha256"]:
        raise ValueError("McPAT XML hash does not match live input")
    mapping_path = output / "mcpat/mapping_report.json"
    if sha256_file(mapping_path) != hashes["mapping_sha256"]:
        raise ValueError("McPAT mapping hash does not match live report")
    mapping = read_json(mapping_path)
    if not isinstance(mapping, dict) \
            or mapping.get("cache_contract") != mcpat["cache_contract"] \
            or mapping.get("optimization_constraints") != mcpat[
                "optimization_constraints"]:
        raise ValueError("McPAT artifact differs from live input mapping")
    if sha256_file(PATCH_FILE) != hashes["patch_sha256"]:
        raise ValueError("McPAT patch hash does not match audited patch")
    if "build_provenance_sha256" in hashes and sha256_file(
            BUILD_PROVENANCE) != hashes["build_provenance_sha256"]:
        raise ValueError("McPAT build provenance hash does not match live record")
    return {"embedded_cacti_p": embedded, "provenance": provenance,
            "binary": binary, "output": mcpat_output}


def validate_corrected_physical_artifacts(output: Path) -> dict:
    """Validate corrected McPAT-native evidence before resume or pairing."""
    output = Path(output).resolve()
    mcpat_path = output / "mcpat/mcpat.json"
    modules_path = output / "modules.json"
    summary_path = output / "pipeline_summary.json"
    mcpat = read_json(mcpat_path)
    modules = read_json(modules_path)
    summary = read_json(summary_path)
    if not all(isinstance(value, dict) for value in (mcpat, modules, summary)):
        raise ValueError("corrected physical artifacts must contain objects")
    native = _validate_native_mcpat(mcpat, output)

    if modules.get("schema_version") != 3:
        raise ValueError("modules.json is not strict schema version 3")
    # Negative compatibility gate only: these legacy identities are rejected,
    # never consumed as corrected evidence.
    legacy_identity_fields = {"source_cacti", "cacti_characterization_id"}
    if legacy_identity_fields.intersection(mcpat) \
            or legacy_identity_fields.intersection(modules) \
            or any(legacy_identity_fields.intersection(module)
                   for module in modules.get("modules", [])
                   if isinstance(module, dict)):
        raise ValueError("corrected physical artifacts retain standalone cache identity")
    if modules.get("cache_authority") != _CACHE_AUTHORITY \
            or modules.get("embedded_cacti_p") != native["embedded_cacti_p"] \
            or modules.get("mcpat_provenance") != native["provenance"]:
        raise ValueError("module model does not preserve McPAT-native authority")
    module_cache_contract = validate_cache_contract(
        modules.get("cache_contract"), expected_core_count=4,
    )
    if module_cache_contract != mcpat.get("cache_contract"):
        raise ValueError("module cache contract differs from McPAT input mapping")
    module_schema = modules.get("module_schema")
    module_records = modules.get("modules")
    if not isinstance(module_schema, dict) or not isinstance(module_records, list) \
            or module_schema.get("schema_version") != 1 \
            or module_schema.get("module_count") != len(module_records) \
            or module_schema.get("core_count") != 4 \
            or module_schema.get("core_logic_granularity") != (
                "McPAT top-level functional blocks") \
            or module_schema.get("requires_granular_cores") is not True:
        raise ValueError("module model lacks the strict granular schema")
    if any(not isinstance(module, dict) for module in module_records):
        raise ValueError("module records must contain objects")
    if any(module.get("movable") is not (module.get("name") == "shared_l2")
           for module in module_records):
        raise ValueError("serialized module mobility flags are invalid")
    observed_mobility = validate_mobility_contract(deepcopy(module_records), 4)
    if modules.get("mobility_contract") != observed_mobility:
        raise ValueError("module mobility contract does not match modules")
    if modules.get("area_provenance") != _AREA_PROVENANCE:
        raise ValueError("module area provenance is not McPAT-native")

    source_r1 = modules.get("source_r1")
    source_mcpat = modules.get("source_mcpat")
    if not isinstance(source_r1, str) or not Path(source_r1).is_absolute() \
            or source_mcpat != str(mcpat_path.resolve()) \
            or summary.get("r1") != str(Path(source_r1).resolve()):
        raise ValueError("module source identity does not match live R1/McPAT inputs")
    reconstructed = construct_model(Path(source_r1), mcpat_path)
    if modules != reconstructed:
        raise ValueError("module scientific payload differs from live R1/McPAT reconstruction")
    _validate_cache_architecture(module_cache_contract, modules["architecture"], mcpat)

    if summary.get("cache_authority") != _CACHE_AUTHORITY \
            or summary.get("mcpat_provenance") != native["provenance"]:
        raise ValueError("pipeline summary lacks McPAT-native provenance")
    artifacts = summary.get("artifacts")
    artifact_hashes = summary.get("artifact_sha256")
    stage_seconds = summary.get("stage_seconds")
    if not isinstance(stage_seconds, dict) or "cacti" in stage_seconds \
            or not isinstance(artifacts, dict) or "cacti" in artifacts \
            or not isinstance(artifact_hashes, dict) or "cacti" in artifact_hashes:
        raise ValueError("pipeline summary retains standalone cache evidence")
    expected_artifacts = {
        "mcpat_json": str(mcpat_path.resolve()),
        "mcpat_output": str(native["output"].resolve()),
        "mcpat_binary": str(native["binary"].resolve()),
    }
    expected_hashes = {
        "mcpat_json": sha256_file(mcpat_path),
        "mcpat_output": native["provenance"]["hashes"]["output_sha256"],
        "mcpat_binary": native["provenance"]["hashes"]["binary_sha256"],
    }
    if any(artifacts.get(name) != value
           for name, value in expected_artifacts.items()):
        raise ValueError("pipeline summary McPAT artifact paths are invalid")
    if any(artifact_hashes.get(name) != value
           for name, value in expected_hashes.items()):
        raise ValueError("pipeline summary McPAT artifact hashes are invalid")
    return {
        "cache_authority": _CACHE_AUTHORITY,
        "mcpat_provenance": native["provenance"],
        "module_count": len(module_records),
        "source_r1": str(Path(source_r1).resolve()),
    }


def require_nonformal_classification(config: dict) -> dict:
    """Return the explicit exploratory classification required for every sweep."""
    classification = config.get("experiment_classification")
    if not isinstance(classification, dict):
        raise ValueError("experiment_classification must be an object")
    if not isinstance(classification.get("mode"), str) or not classification["mode"]:
        raise ValueError("experiment_classification must include a nonempty mode")
    if (classification.get("non_formal") is not True
            or classification.get("paper_equivalent") is not False
            or classification.get("shared_parameter_accepted") is not False):
        raise ValueError("experiment_classification must be non-formal exploratory")
    return classification


def discover(root: Path, experiment_path: Path, profile: str = "paper",
             workloads: set[str] | None = None,
             require_complete: bool = True) -> tuple[list[Path], dict]:
    """Return valid configured canonical points after validating the full grid."""
    catalogue = build_catalogue(root, experiment_path, profile=profile)
    if require_complete and not catalogue["complete"]:
        raise ValueError("canonical R1 catalogue is incomplete or invalid")
    points = canonical_directories(catalogue)
    if workloads is not None:
        points = [point for point in points
                  if point.relative_to(root.resolve()).parts[0] in workloads]
    return points, catalogue


def completed(output: Path, config: dict, layout_method: str,
              require_r2: bool) -> bool:
    artifacts = required_artifacts
    if layout_method == "clip3d":
        artifacts += CLIP3D_REQUIRED_ARTIFACTS
    if any(not (output / artifact).is_file() for artifact in artifacts):
        return False
    path = output / "pipeline_summary.json"
    run_config_path = output / "run_config.json"
    try:
        summary = read_json(path)
        recorded_config = read_json(run_config_path).get("config")
        validate_corrected_physical_artifacts(output)
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    # Cooling alone is not a sufficient cache key: McPAT activity mapping,
    # embedded cache records, module geometry, or layer materials may change
    # while R_conv stays identical.
    if recorded_config != config:
        return False
    if summary.get("layout_method", summary.get("layout_mode")) != layout_method:
        return False
    cooling = summary.get("cooling", {})
    if float(cooling.get("r_convec_k_per_w", -1)) != float(
            config["physical"]["r_convec_k_per_w"]):
        return False
    if require_r2 and (summary.get("ipc2") is None or summary.get("bips2") is None):
        return False
    if not require_r2 and (summary.get("ipc2") is not None
                           or summary.get("bips2") is not None):
        return False
    return True


def one_job(args_tuple):
    r1, output, config_path, layout_method, execute_r2, rerun_r2, reuse_r2 = args_tuple
    try:
        summary = run_pipeline(
            r1, output, config_path, layout_method, execute_r2, rerun_r2, reuse_r2
        )
        return {"r1": str(r1), "output": str(output), "state": "success",
                "summary": summary}
    except Exception as error:  # retain all independent completed points
        return {"r1": str(r1), "output": str(output), "state": "failed",
                "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-root", type=Path, default=DEFAULT_R1_ROOT)
    parser.add_argument("--r1-experiment", type=Path, default=DEFAULT_R1_EXPERIMENT)
    parser.add_argument("--r1-profile", default="paper")
    parser.add_argument("--allow-incomplete-canonical", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_SWEEP_CONFIG)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--layout-method", choices=LAYOUT_METHODS, default="fixed-bin")
    parser.add_argument("--optimized-layout", action="store_true",
                        help="deprecated alias for --layout-method clip3d")
    parser.add_argument("--run-r2", action="store_true")
    parser.add_argument("--rerun-r2", action="store_true")
    parser.add_argument("--reuse-r2-root", type=Path,
                        help="reuse matching per-point R2 results from another sweep root")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.optimized_layout:
        if args.layout_method != "fixed-bin":
            parser.error("do not combine --optimized-layout and --layout-method")
        args.layout_method = "clip3d"
    if args.run_r2 and args.reuse_r2_root:
        parser.error("choose either --run-r2 or --reuse-r2-root")

    r1_root = args.r1_root.resolve()
    r1_experiment = args.r1_experiment.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    classification = require_nonformal_classification(config)
    reuse_root = args.reuse_r2_root.resolve() if args.reuse_r2_root else None
    points, catalogue = discover(
        r1_root, r1_experiment, args.r1_profile,
        set(args.workloads) if args.workloads else None,
        not args.allow_incomplete_canonical,
    )
    jobs = []
    skipped = []
    for point in points:
        output = output_root / point.relative_to(r1_root)
        if not args.rerun and completed(
                output, config, args.layout_method, args.run_r2 or reuse_root is not None):
            skipped.append(str(point))
        else:
            reuse_r2 = reuse_root / point.relative_to(r1_root) if reuse_root else None
            jobs.append((point, output, config_path, args.layout_method,
                         args.run_r2, args.rerun_r2, reuse_r2))

    if args.jobs == 1:
        results = [one_job(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(executor.map(one_job, jobs))
    skipped_summaries = [read_json(
        output_root / point.relative_to(r1_root) / "pipeline_summary.json"
    ) for point in points if str(point) in skipped]
    result_summaries = [result["summary"] for result in results
                        if result["state"] == "success"]
    all_summaries = skipped_summaries + result_summaries
    contains_r2 = any(summary.get("ipc2") is not None or summary.get("bips2") is not None
                      for summary in all_summaries)
    layout_only = not args.run_r2 and reuse_root is None
    validation_error = (
        "layout-only lifting sweep contains R2 performance results"
        if layout_only and contains_r2 else None
    )
    failed = sum(result["state"] == "failed" for result in results)
    report = {
        "schema_version": 3, "r1_root": str(r1_root),
        "r1_experiment": str(r1_experiment), "r1_profile": args.r1_profile,
        "output_root": str(output_root), "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "experiment": config.get("name", config_path.stem),
        "classification": classification,
        "layout_method": args.layout_method, "run_r2": args.run_r2,
        "reuse_r2_root": str(reuse_root) if reuse_root else None,
        "contains_r2": contains_r2,
        "validation_error": validation_error,
        "workloads": args.workloads,
        "selected_workload_count": len({
            point.relative_to(r1_root).parts[0] for point in points
        }),
        "canonical_expected": catalogue["expected_count"],
        "canonical_valid": catalogue["valid_count"],
        "canonical_complete": catalogue["complete"],
        "canonical_selected": len(points),
        "excluded_noncanonical": catalogue["excluded_noncanonical"],
        "discovered": len(points), "executed": len(jobs),
        "skipped_count": len(skipped), "failed": failed,
        "skipped": skipped, "results": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "sweep_status.json", report)
    success = sum(result["state"] == "success" for result in results)
    print(f"lifting sweep: discovered={len(points)} success={success} "
          f"failed={failed} skipped={len(skipped)}")
    if failed or validation_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
