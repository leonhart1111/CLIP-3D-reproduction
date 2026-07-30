#!/usr/bin/env python3
"""Run a provenance-locked matched R2 layout-wire sensitivity series."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

from workflow.common import read_json, write_json
from workflow.r2.build_latency_vector import build_vector
from workflow.r2.calibrate_lambda_wire import calibrate_series, summarize_workloads
from workflow.r2.run_r2 import run as run_r2


def _gem5_args(overrides: dict) -> list[str]:
    return [item for key, value in overrides.items()
            for item in ("--" + key.replace("_", "-"), str(value))]


def _validated_cycles(cycles: list[int]) -> list[int]:
    """Validate the distinct wire levels required for an R2 calibration."""
    if not cycles:
        raise ValueError("wire sensitivity requires at least three distinct cycle levels")
    requested = [int(cycle) for cycle in cycles]
    if any(cycle < 0 for cycle in requested):
        raise ValueError("wire sensitivity cycles must be non-negative")
    if len(set(requested)) != len(requested):
        raise ValueError("wire sensitivity cycles must be distinct")
    if len(requested) < 3:
        raise ValueError("wire sensitivity requires at least three distinct cycle levels")
    return requested


def build_sensitivity_vectors(base_vector: dict, cycles: list[int]) -> dict[int, dict]:
    """Return independent vectors that change only the injected wire latency."""
    requested = _validated_cycles(cycles)

    components = base_vector.get("components_cycles", {})
    required = ("l1d_cacti", "l1_pipeline", "l2_cacti", "l2_arbitration", "tsv")
    missing = [field for field in required if field not in components]
    if missing:
        raise ValueError(f"base latency vector is missing components: {', '.join(missing)}")
    if "gem5_overrides" not in base_vector:
        raise ValueError("base latency vector is missing gem5_overrides")

    result = {}
    for cycle in requested:
        vector = copy.deepcopy(base_vector)
        parts = vector["components_cycles"]
        parts["layout_wire"] = cycle
        vector["critical_l1d_to_l2_cycles"] = (
            parts["l1d_cacti"] + parts["l1_pipeline"] + parts["l2_cacti"]
            + parts["l2_arbitration"] + parts["tsv"] + cycle
        )
        vector["gem5_overrides"]["xbar_forward_latency"] = max(
            parts["l2_arbitration"] + parts["tsv"] + cycle, 1
        )
        vector["gem5_args"] = _gem5_args(vector["gem5_overrides"])
        result[cycle] = vector
    return result


def _normalized_vector(vector: dict) -> dict:
    """Mask the three injected values while retaining all other R2 inputs."""
    normalized = copy.deepcopy(vector)
    try:
        del normalized["components_cycles"]["layout_wire"]
        del normalized["critical_l1d_to_l2_cycles"]
        del normalized["gem5_overrides"]["xbar_forward_latency"]
    except KeyError as error:
        raise ValueError(f"latency vector lacks injected wire field: {error}") from error
    args = normalized.get("gem5_args")
    if not isinstance(args, list) or len(args) % 2:
        raise ValueError("latency vector has malformed gem5_args")
    expected = _gem5_args(vector["gem5_overrides"])
    if args != expected:
        raise ValueError("latency vector gem5_args are not regenerated from gem5_overrides")
    normalized_args = list(args)
    try:
        forward = normalized_args.index("--xbar-forward-latency")
    except ValueError as error:
        raise ValueError("latency vector lacks xbar-forward-latency argument") from error
    normalized_args[forward + 1] = "<injected-layout-wire>"
    normalized["gem5_args"] = normalized_args
    return normalized


def _assert_matched_vectors(vectors: dict[int, dict]) -> None:
    iterator = iter(vectors.items())
    _, first = next(iterator)
    expected = _normalized_vector(first)
    for cycle, vector in iterator:
        if _normalized_vector(vector) != expected:
            raise ValueError(
                f"wire sensitivity vector for cycle {cycle} changes a non-injected R2 input"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def _base_vector(r1_dir: Path, point_dir: Path, output_dir: Path,
                 config_path: Path) -> tuple[dict, dict]:
    """Build a fresh R2 vector from one completed lifted raw-power point."""
    metadata = read_json(_required_file(r1_dir / "r1_metadata.json", "R1 metadata"))
    point_config_path = _required_file(point_dir / "run_config.json", "lifting config")
    point_config = read_json(point_config_path).get("config")
    candidate_config = read_json(_required_file(config_path, "candidate config"))
    if point_config != candidate_config:
        raise ValueError("candidate config does not match the completed lifting point")
    modules = _required_file(point_dir / "modules.json", "lifted modules")
    cacti = _required_file(point_dir / "cacti/cacti_characterization.json", "CACTI result")
    layout = _required_file(point_dir / "hotspot/layout.json", "lifted layout")
    _required_file(point_dir / "performance.json", "lifting performance")
    _required_file(point_dir / "pipeline_summary.json", "lifting summary")
    delay = candidate_config.get("delay", {})
    base_path = output_dir / "base_latency.json"
    vector = build_vector(
        modules, cacti, base_path, layout_path=layout,
        wire_rounding=delay.get("wire_rounding", "nearest"),
        cycles_per_tsv=int(delay.get("cycles_per_tsv", 2)),
        l1_pipeline_cycles=int(delay.get("l1_pipeline_cycles", 1)),
        wire_aggregation=delay.get("wire_aggregation", "mean"),
    )
    module_model = read_json(modules)
    source_r1 = module_model.get("source_r1")
    if source_r1 is None or Path(source_r1).resolve() != r1_dir:
        raise ValueError("lifted modules do not originate from this exact R1 directory")
    if module_model.get("power_provenance", {}).get("postprocessing") != "none":
        raise ValueError("lifting point is not a completed raw-power point")
    if metadata.get("workload") != module_model.get("architecture", {}).get("workload"):
        raise ValueError("R1 workload does not match the lifted modules")
    return vector, candidate_config


def _input_hashes(r1_dir: Path, point_dir: Path, config_path: Path,
                  vector_path: Path) -> dict[str, str]:
    return {
        "r1_stats": _sha256(_required_file(r1_dir / "stats.txt", "R1 stats")),
        "r1_metadata": _sha256(_required_file(r1_dir / "r1_metadata.json", "R1 metadata")),
        "modules": _sha256(_required_file(point_dir / "modules.json", "lifted modules")),
        "cacti": _sha256(_required_file(
            point_dir / "cacti/cacti_characterization.json", "CACTI result")),
        "performance": _sha256(_required_file(
            point_dir / "performance.json", "lifting performance")),
        "candidate_config": _sha256(_required_file(config_path, "candidate config")),
        "vector": _sha256(_required_file(vector_path, "sensitivity vector")),
    }


def _resume_result(cycle_dir: Path, hashes: dict[str, str]) -> dict | None:
    result_path = cycle_dir / "r2_result.json"
    status_path = cycle_dir / "status.json"
    provenance_path = cycle_dir / "input_hashes.json"
    if not status_path.exists() and not provenance_path.exists() and not result_path.exists():
        return None
    if not status_path.exists() or not provenance_path.exists():
        raise ValueError(f"incomplete R2 resume state: {cycle_dir}")
    status = read_json(status_path)
    recorded = read_json(provenance_path)
    if recorded != hashes:
        raise ValueError(f"incompatible R2 resume state: {cycle_dir}")
    if status.get("state") != "success":
        return None
    if not result_path.is_file():
        raise ValueError(f"successful R2 resume is missing its result: {cycle_dir}")
    result = read_json(result_path)
    if Path(result.get("latency_vector", "")).resolve() != (cycle_dir / "latency.json"):
        raise ValueError(f"R2 result refers to a different sensitivity vector: {cycle_dir}")
    return result


def run_series(r1_dir: Path, point_dir: Path, output_dir: Path, cycles: list[int],
               config_path: Path, execute: bool = True) -> dict:
    """Prepare or execute one exact-R1, exact-lifting-point sensitivity series."""
    requested = _validated_cycles(cycles)
    r1_dir = Path(r1_dir).resolve()
    point_dir = Path(point_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base, _ = _base_vector(r1_dir, point_dir, output_dir, config_path)
    vectors = build_sensitivity_vectors(base, requested)
    _assert_matched_vectors(vectors)
    metadata = read_json(r1_dir / "r1_metadata.json")
    samples = []
    entries = []
    for cycle, vector in vectors.items():
        vector_path = output_dir / f"wire_{cycle}" / "latency.json"
        write_json(vector_path, vector)
        cycle_dir = vector_path.parent
        hashes = _input_hashes(r1_dir, point_dir, config_path, vector_path)
        result = _resume_result(cycle_dir, hashes)
        if result is None and execute:
            write_json(cycle_dir / "input_hashes.json", hashes)
            result = run_r2(r1_dir, vector_path, cycle_dir)
        if result is not None:
            samples.append((str(cycle), cycle_dir / "r2_result.json", vector_path))
        entries.append({
            "wire_cycles": cycle, "latency": str(vector_path.resolve()),
            "result": str((cycle_dir / "r2_result.json").resolve()) if result else None,
            "input_hashes": hashes, "state": "success" if result else "prepared",
        })

    calibration = None
    if len(samples) >= 3:
        performance = read_json(point_dir / "performance.json")
        calibration = calibrate_series(
            samples, float(read_json(point_dir / "modules.json")["ipc1"]),
            float(performance["sustainable_frequency_ghz"]),
        )
        write_json(output_dir / "lambda_wire_report.json", calibration)
    report = {
        "schema_version": 1,
        "workload": str(metadata["workload"]).upper(),
        "r1_dir": str(r1_dir), "point_dir": str(point_dir),
        "candidate_config": str(config_path), "cycles": list(vectors),
        "execute": execute, "samples": entries, "calibration": calibration,
        "recommendation": (calibration or {"accepted_for_this_workload": False})
            .get("recommendation", {"accepted_for_this_workload": False}),
    }
    write_json(output_dir / "wire_sensitivity_manifest.json", report)
    return report


def _parse_series(text: str) -> Path:
    if "=" not in text:
        raise argparse.ArgumentTypeError("series must be LABEL=wire_sensitivity_manifest.json")
    _, path = text.split("=", 1)
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise argparse.ArgumentTypeError(f"missing series report: {text}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--series", action="append", type=_parse_series)
    parser.add_argument("--output", type=Path,
                        help="required for --summary; series always write OUTPUT_DIR's manifest")
    parser.add_argument("--r1-dir", type=Path)
    parser.add_argument("--point-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wire-cycles", type=int, nargs="+")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.summary:
        if any((args.r1_dir, args.point_dir, args.config, args.output_dir, args.wire_cycles)):
            parser.error("--summary cannot be combined with series execution arguments")
        if args.output is None:
            parser.error("--summary requires --output")
        summary = summarize_workloads(args.series or [])
        write_json(args.output, summary)
        print(f"selected_lambda_wire={summary['selected_lambda_wire']}")
        return
    if args.output_dir is None or args.r1_dir is None or args.point_dir is None or args.config is None:
        parser.error("--r1-dir, --point-dir, --config, and --output-dir are required")
    report = run_series(args.r1_dir, args.point_dir, args.output_dir,
                        args.wire_cycles or [], args.config, not args.prepare_only)
    print(f"wire sensitivity {report['workload']}: {len(report['samples'])} levels")


if __name__ == "__main__":
    main()
