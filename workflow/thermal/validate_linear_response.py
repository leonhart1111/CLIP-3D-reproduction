#!/usr/bin/env python3
"""Validate linear-response predictions against full HotSpot probes.

This is the non-ML stage-one acceptance path.  It deliberately keeps the
baseline geometry, package and HotSpot contract unchanged and changes only
the module/source power vector for each candidate.  A candidate that passes
this check is *not* a final architecture result; it only demonstrates that
the cheap response model is accurate enough for ranking under this fixed
contract.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from workflow.common import read_json, sha256_file, write_json
from workflow.thermal.build_linear_response import (
    _copy_case,
    _power_trace_values,
    _temperature_vector,
)
from workflow.thermal.linear_response import LinearThermalResponse, parse_ptrace, soft_peak
from workflow.thermal.run_hotspot import DEFAULT_HOTSPOT, run_hotspot


def _candidate_document(path: Path) -> list[dict[str, Any]]:
    document = read_json(path)
    candidates = document.get("candidates")
    if document.get("schema_version") != 1 or not isinstance(candidates, list):
        raise ValueError("candidate document must have schema_version=1 and candidates")
    if not candidates:
        raise ValueError("candidate document must contain at least one candidate")
    return candidates


def _candidate_trace(response: LinearThermalResponse,
                     baseline_values: np.ndarray,
                     candidate_power: np.ndarray) -> np.ndarray:
    mappings = response.metadata.get("source_mappings")
    if not isinstance(mappings, list) or len(mappings) != response.source_count:
        raise ValueError("response metadata has no complete source_mappings")
    result = baseline_values.copy()
    delta = candidate_power - response.baseline_power_w
    for source_index, mapping in enumerate(mappings):
        entries = mapping.get("source_cells") if isinstance(mapping, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"source mapping {source_index} has no source_cells")
        for entry in entries:
            index = int(entry["index"])
            weight = float(entry["weight"])
            if index < 0 or index >= result.size or not math.isfinite(weight):
                raise ValueError("source mapping contains invalid cell index/weight")
            result[index] += delta[source_index] * weight
    if np.any(result < -1.0e-9):
        raise ValueError("candidate produces a negative HotSpot source cell")
    result[result < 0.0] = 0.0
    return result


def _validate_one(response: LinearThermalResponse, candidate: dict[str, Any],
                  baseline_case: Path, baseline_values: np.ndarray,
                  names: list[str], output_dir: Path, hotspot: Path,
                  force: bool) -> dict[str, Any]:
    identifier = candidate.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("each candidate needs a non-empty id")
    if Path(identifier).name != identifier or identifier in {".", ".."}:
        raise ValueError(f"candidate {identifier!r} is not a safe directory name")
    power = np.asarray(candidate.get("power_w", []), dtype=np.float64)
    if power.ndim != 1 or power.size != response.source_count:
        raise ValueError(f"candidate {identifier} has the wrong power vector")
    if not np.all(np.isfinite(power)) or np.any(power < -1.0e-12):
        raise ValueError(f"candidate {identifier} has invalid power")
    candidate_dir = output_dir / identifier
    result_path = candidate_dir / "validation.json"
    signature = {
        "candidate_id": identifier,
        "power_w": power.tolist(),
        "response_manifest": response.metadata.get("response_manifest"),
    }
    if result_path.is_file() and not force:
        existing = read_json(result_path)
        if existing.get("signature") == signature:
            return existing
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    trace = _candidate_trace(response, baseline_values, power)
    _copy_case(baseline_case, candidate_dir, _power_trace_values(names, trace))
    run_hotspot(candidate_dir, hotspot=hotspot)
    _, actual = _temperature_vector(candidate_dir)
    predicted = response.predict(power)
    if actual.size != predicted.size:
        raise ValueError(f"candidate {identifier} temperature vector length mismatch")
    difference = actual - predicted
    tau = float(response.metadata.get("validation_smooth_peak_tau_c", 1.0))
    report = {
        "schema_version": 1,
        "candidate_id": identifier,
        "signature": signature,
        "case_dir": str(candidate_dir.resolve()),
        "power_w": power.tolist(),
        "predicted_tmax_c": float(np.max(predicted)),
        "actual_tmax_c": float(np.max(actual)),
        "tmax_error_c": float(np.max(actual) - np.max(predicted)),
        "predicted_tsoft_c": float(soft_peak(predicted, tau)),
        "actual_tsoft_c": float(soft_peak(actual, tau)),
        "tsoft_error_c": float(soft_peak(actual, tau) - soft_peak(predicted, tau)),
        "temperature_mae_c": float(np.mean(np.abs(difference))),
        "temperature_max_abs_error_c": float(np.max(np.abs(difference))),
        "predicted_temperature_c": predicted.tolist(),
        "actual_temperature_c": actual.tolist(),
    }
    write_json(result_path, report)
    return report


def validate(response_manifest: Path, candidates_path: Path, output: Path,
             hotspot: Path = DEFAULT_HOTSPOT, workers: int = 1,
             max_candidates: int | None = None, force: bool = False,
             mae_limit_c: float = 1.0, max_error_limit_c: float = 2.0) -> dict[str, Any]:
    response = LinearThermalResponse.load(response_manifest)
    baseline_raw = response.metadata.get("baseline_case")
    if not isinstance(baseline_raw, str):
        raise ValueError("response metadata is missing baseline_case")
    baseline_case = Path(baseline_raw).resolve()
    if not baseline_case.is_dir():
        raise FileNotFoundError(f"baseline case does not exist: {baseline_case}")
    names, baseline_values = parse_ptrace(baseline_case / "power.ptrace")
    candidates = _candidate_document(candidates_path)
    if max_candidates is not None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        candidates = candidates[:max_candidates]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be positive")
    jobs = [(candidate, output / "cases") for candidate in candidates]
    if workers == 1:
        records = [_validate_one(response, candidate, baseline_case, baseline_values,
                                 names, case_root, hotspot, force)
                   for candidate, case_root in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(
                _validate_one, response, candidate, baseline_case, baseline_values,
                names, case_root, hotspot, force,
            ) for candidate, case_root in jobs]
            records = [future.result() for future in futures]
    for record in records:
        record["passed"] = (
            record["temperature_mae_c"] <= mae_limit_c
            and record["temperature_max_abs_error_c"] <= max_error_limit_c
        )
    report = {
        "schema_version": 1,
        "mode": "hotspot-held-out-linear-response-validation",
        "response_manifest": str(response_manifest.resolve()),
        "response_manifest_sha256": sha256_file(response_manifest),
        "candidate_document": str(candidates_path.resolve()),
        "candidate_document_sha256": sha256_file(candidates_path),
        "baseline_case": str(baseline_case),
        "candidate_count": len(records),
        "thresholds": {
            "temperature_mae_c_max": float(mae_limit_c),
            "temperature_max_abs_error_c_max": float(max_error_limit_c),
        },
        "passed_count": sum(1 for record in records if record["passed"]),
        "all_passed": all(record["passed"] for record in records),
        "candidates": records,
    }
    write_json(output / "validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hotspot", type=Path, default=DEFAULT_HOTSPOT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--mae-limit-c", type=float, default=1.0)
    parser.add_argument("--max-error-limit-c", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = validate(
        args.response.resolve(), args.candidates.resolve(), args.output.resolve(),
        args.hotspot.resolve(), args.workers, args.max_candidates, args.force,
        args.mae_limit_c, args.max_error_limit_c,
    )
    print(
        f"linear response validation: {report['passed_count']}/"
        f"{report['candidate_count']} passed; output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
